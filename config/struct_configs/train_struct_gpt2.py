# I/O
eval_interval = 200
log_interval = 100
always_save_checkpoint = False

# debug
debug = False 

# wandb
wandb_log = True
wandb_entity = "yilunkuang"
wandb_project = 'struct_sequence_mixing'
wandb_run_name='structured_gpt'

# data
dataset = 'openwebtext_small'
data_dir = os.path.join('/scratch/yk2516/repos/structure/struct_foundation_models/data', dataset)
data_dtype = np.uint8
vocab_size = 96

# model
n_layer = 6
n_head = -1
d_head = 64
d_qk_head = -1 # if -1, set to d_head
d_model = 768
split_qkv = True
do_qk_ln = True
manual_disable_flash_att=True
token_mixing_struct = "low_rank"
link_function = "softmax"

# optimizer
opt_name = "AdamW"
init_lr = 3e-3
min_lr = init_lr / 10.0

# compile
compile=False 

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 64
block_size = 256
gradient_accumulation_steps = 8

# this makes total number of tokens be 300B
decay_lr = False
max_iters = 100_000
lr_decay_iters = 100_000

# weight decay
weight_decay = 0.0

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