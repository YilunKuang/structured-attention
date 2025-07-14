"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.model import GPTConfig, GPT
from src.muP import muPify
from src.utils import compute_model_FLOPs, generate_out_dir
from src.lr_schedule import update_lrs, reset_lrs, get_lr_mult

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# debug
debug = False 

# wandb logging
wandb_log = False # disabled by default
wandb_entity = None
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)
data_dtype = np.uint16
vocab_size = 50304

gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
d_head = 64
d_qk_head = -1 # if -1, set to d_head
d_model = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
split_qkv = False
do_qk_ln = False
manual_disable_flash_att=False
token_mixing_struct = "low_rank"
link_function = "softmax"

# adamw optimizer
opt_name = "AdamW"
init_lr = 6e-4 # max learning rate
max_iters = 600_000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = init_lr / 10.0 # minimum learning rate, should be ~= init_lr/10 per Chinchilla

# Sliding Window Attention
sliding_block_size = 1024
use_swa_with_for_loop = False

# SWA + Global (GSWA)
gswa_rank_list = "32|32"

# Sequential GSWA
sequential_GSWA_layer_list = "MHA|SWA|SWA|MHA|SWA|SWA"

# Multi-Head Attention
mha_SP_attn_logits_scaling = False

# Multi-Level Low Rank Matrix (MLR)
mlr_rank_list = "16|16|16|16"
mlr_divide_by_num_levels=False
mlr_block_divide_by_num_levels=False
mlr_block_size_list="default"

# BilinearMLR
bilinear_mlr_muP_attn_logits_scaling=False

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# automatically update min_lr after passing in init_lr with new values
config.update({'min_lr': init_lr / 10.0})

# automatically update d_qk_head
if d_qk_head == -1:
    d_qk_head = d_head
    config.update({'d_qk_head': d_qk_head})
else:
    assert token_mixing_struct == "low_rank" # non-default d_qk_head only compactible with low rank for now

# update out_dir
if init_from == 'scratch':
    out_dir = generate_out_dir(out_dir)
elif init_from == 'resume':
    pass
else:
    raise ValueError

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
def get_batch(split, data_dtype):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=data_dtype, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=data_dtype, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 1
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, d_head=d_head, d_qk_head=d_qk_head, d_model=d_model, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, split_qkv=split_qkv, do_qk_ln=do_qk_ln, 
                  manual_disable_flash_att=manual_disable_flash_att, token_mixing_struct=token_mixing_struct,
                  mlr_rank_list=mlr_rank_list, mlr_divide_by_num_levels=mlr_divide_by_num_levels, 
                  mlr_block_divide_by_num_levels=mlr_block_divide_by_num_levels, link_function=link_function,
                  bilinear_mlr_muP_attn_logits_scaling=bilinear_mlr_muP_attn_logits_scaling, mha_SP_attn_logits_scaling=mha_SP_attn_logits_scaling,
                  sliding_block_size=sliding_block_size, use_swa_with_for_loop=use_swa_with_for_loop, gswa_rank_list=gswa_rank_list, 
                  sequential_GSWA_layer_list=sequential_GSWA_layer_list, mlr_block_size_list=mlr_block_size_list, debug=debug)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    print(f"defaulting to vocab_size of GPT-2 to {vocab_size}")

    if token_mixing_struct == "low_rank_with_sliding_window" or token_mixing_struct == "low_rank_with_gswa" or token_mixing_struct == "low_rank_with_sequential_gswa":
        model_args['use_swa_with_for_loop']=True # temporary changes for FLOPs counting

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")

    if token_mixing_struct == "low_rank_with_sliding_window" or token_mixing_struct == "low_rank_with_gswa" or token_mixing_struct == "low_rank_with_sequential_gswa":
        raise NotImplementedError

    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # for k in ['n_layer', 'n_head', 'd_model', 'block_size', 'bias', 'vocab_size']:
    #     model_args[k] = checkpoint_model_args[k]
    
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)
    print("model loaded!")

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError

if token_mixing_struct == "low_rank_with_sliding_window" or token_mixing_struct == "low_rank_with_gswa" or token_mixing_struct == "low_rank_with_sequential_gswa":
    assert init_from == 'scratch', "didn't implement for resumed training"
    model.to(device)
    
    # get the optimal FLOPs for SWA with slow implementation
    flops_per_token, non_emb_flops_per_token = compute_model_FLOPs(model, config, vocab_size, block_size, device)
    
    # switch back to fast implementation
    model_args['use_swa_with_for_loop']=False

    # refine the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # muP
    model, optimizer = muPify(
        model, 
        device, device_type,
        # optimization arguments
        opt_name, init_lr, weight_decay, beta1, beta2, init_from=init_from,
    )
    print(model)
else:
    # muP 
    model, optimizer = muPify(
        model, 
        device, device_type,
        # optimization arguments
        opt_name, init_lr, weight_decay, beta1, beta2, init_from=init_from,
    )

    # It's okay to recompute FLOPs for resumed training since they are the same anyway.
    flops_per_token, non_emb_flops_per_token = compute_model_FLOPs(model, config, vocab_size, block_size, device)
    print(model)

if debug==True:
    breakpoint()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer (deprecated)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("optimizer loaded!")

checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        eval_total_time = 0
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dtype)

            # time the evaluation 
            eval_start_time = time.time()
            with ctx:
                logits, loss = model(X, Y)
            eval_end_time = time.time()
            eval_iter_time = eval_end_time - eval_start_time
            eval_total_time += eval_iter_time

            losses[k] = loss.item()
        
        eval_avg_time = eval_total_time / eval_iters
        out[split] = losses.mean()
        out[f"{split}_time"] = eval_avg_time
    model.train()
    return out

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(entity=wandb_entity, project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train', data_dtype) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
init_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

for _ in (pbar := tqdm(range(max_iters))):
    mult = get_lr_mult(iter_num, init_lr, min_lr if decay_lr else init_lr, warmup_iters, lr_decay_iters)
    global_lr = init_lr * mult
    update_lrs(optimizer, mult)

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "compute": flops_per_token * tokens_per_iter * iter_num,
                "non_emb_compute": non_emb_flops_per_token * tokens_per_iter * iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "val/time":  losses['val_time'],
                "lr": global_lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', data_dtype)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    reset_lrs(optimizer, init_lrs)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    pbar.update(1)

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
