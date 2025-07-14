import os
import uuid
import math 
from random import randint

import argparse
from tqdm import tqdm
import torch
import yaml
import numpy as np
import wandb

from in_context_regression.eval import get_run_metrics, get_run_metrics_new
from in_context_regression.tasks import get_task_sampler
from in_context_regression.samplers import get_data_sampler
from in_context_regression.curriculum import Curriculum, CurriculumFullArgs
# from schema import schema
from in_context_regression.models import build_model, build_model_full_args
# from in_context_regression.model.gpt_fns import construct_configs
# from nn.cola_nn import cola_parameterize
# from in_context_regression.model.gpt_fns import get_lr_mult
# from in_context_regression.model.gpt_fns import update_lrs
# fromin_context_regression. model.gpt_fns import reset_lrs
# from nn.cola_nn import get_model_summary_and_flops

from src.lr_schedule import get_lr_mult, update_lrs, reset_lrs
from src.models.icl_models import GPTforICLRegressionConfig, GPTforICLRegression
from src.muP import muPify
from src.utils import compute_model_FLOPs_for_ICLRegression

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for the training script.")

    # Model schema
    parser.add_argument("--n_dims", type=int, required=True)
    parser.add_argument("--n_head", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--d_head", type=int, default=-1)
    parser.add_argument("--d_qk_head", type=int, default=-1)
    parser.add_argument("--n_positions", type=int, default=None)

    parser.add_argument('--bias', type=eval, choices=[True, False], default=False)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument('--split_qkv', type=eval, choices=[True, False], default=True)
    parser.add_argument('--do_qk_ln', type=eval, choices=[True, False], default=True)
    parser.add_argument('--manual_disable_flash_att', type=eval, choices=[True, False], default=True)

    # Optimization
    parser.add_argument("--opt_name", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--training_learning_rate", type=float, default=3e-4, help="Learning rate.")

    # Attention Specific
    parser.add_argument('--mha_SP_attn_logits_scaling', type=eval, choices=[True, False], default=False)

    # MLR Specific
    parser.add_argument("--token_mixing_struct", type=str, required=True)
    parser.add_argument("--mlr_rank_list", type=str, required=True)
    parser.add_argument('--mlr_divide_by_num_levels', type=eval, choices=[True, False], default=False)
    parser.add_argument('--mlr_block_divide_by_num_levels', type=eval, choices=[True, False], default=False)
    parser.add_argument("--link_function", type=str, default="softmax")
    parser.add_argument('--bilinear_mlr_muP_attn_logits_scaling', type=eval, choices=[True, False], default=False)

    # BTT schema
    parser.add_argument("--btt_tt_dim", type=int, default=2, help="btt_tt_dim.")
    parser.add_argument("--btt_tt_rank", type=int, default=1, help="btt_tt_rank.")
    parser.add_argument('--bilinear_btt_muP_attn_logits_scaling', type=eval, choices=[True, False], default=False)

    # Curriculum schema
    parser.add_argument("--training_curriculum_dims_start", type=int, default=4, help="Initial parameter for dims.")
    parser.add_argument("--training_curriculum_dims_end", type=int, default=None, help="Final value for dims.")
    parser.add_argument("--training_curriculum_dims_inc", type=int, default=None, help="Increment for dims.")
    parser.add_argument("--training_curriculum_dims_interval", type=int, default=2000, help="Interval for dims.")
    parser.add_argument("--training_curriculum_points_start", type=int, default=8, help="Initial parameter for points.")
    parser.add_argument("--training_curriculum_points_end", type=int, default=None, help="Final value for points.")
    parser.add_argument("--training_curriculum_points_inc", type=int, default=None, help="Increment for points.")
    parser.add_argument("--training_curriculum_points_interval", type=int, default=2000, help="Interval for points.")
    parser.add_argument('--training_curriculum_adaptive_inc', type=eval, choices=[True, False], default=True, help="use adaptive increments or not (default: True)")
    parser.add_argument('--training_curriculum_warmup_ratio', type=float, default=0.06, help="warm up ratio for training curriculum (default: 1.0)")

    # Training schema
    TASK_LIST = [
        "linear_regression",
        "sparse_linear_regression",
        "linear_classification",
        "relu_2nn_regression",
        "decision_tree",
    ]
    parser.add_argument("--training_task", type=str, choices=TASK_LIST, default="linear_regression", help="Task name.")
    parser.add_argument("--training_task_kwargs", type=dict, default={}, help="Task keyword arguments.")
    parser.add_argument("--training_num_tasks", type=int, default=None, help="Number of tasks.")
    parser.add_argument("--training_num_training_examples", type=int, default=None, help="Number of training examples.")
    parser.add_argument("--training_data", type=str, choices=["gaussian"], default="gaussian", help="Data type.")
    parser.add_argument("--training_batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--training_train_steps", type=int, default=500001, help="Training steps.")
    parser.add_argument("--training_save_every_steps", type=int, default=10000, help="Save every n steps.")
    parser.add_argument("--training_keep_every_steps", type=int, default=100000, help="Keep every n steps.")
    parser.add_argument("--training_resume_id", type=str, default=None, help="Resume ID.")
    
    # Evaluation schema
    parser.add_argument("--evaluation_interval", type=int, default=2500, help="evaluation_interval.")
    parser.add_argument("--evaluation_samples_multiplier", type=int, default=50, help="number of samples for evaluation = evaluation_samples_multiplier * training_batch_size.")

    # WandB schema
    parser.add_argument("--wandb_project", type=str, default="ICL-Functions", help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, default="in-context", help="WandB entity.")
    parser.add_argument("--wandb_notes", type=str, default="", help="WandB notes.")
    parser.add_argument("--wandb_name", type=str, default="linear_regression_standard", help="WandB run name.")
    parser.add_argument("--wandb_log_every_steps", type=int, default=10, help="Log every n steps.")

    # Global schema
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--test_run", action="store_true", help="Run a test.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")

    args = parser.parse_args()

    return args



def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

@torch.no_grad()
def eval_step(model, xs, ys, loss_func):
    model.eval()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    model.train()
    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, optimizer, args, device):
    init_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

    # initialize computes
    compute = 0
    non_emb_compute = 0

    curriculum = CurriculumFullArgs(
        args.training_curriculum_dims_start,
        args.training_curriculum_dims_end,
        args.training_curriculum_dims_inc,
        args.training_curriculum_dims_interval,
        args.training_curriculum_points_start,
        args.training_curriculum_points_end,
        args.training_curriculum_points_inc,
        args.training_curriculum_points_interval,
    )

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training_batch_size
    data_sampler = get_data_sampler(args.training_data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training_task,
        n_dims,
        bsize,
        num_tasks=args.training_num_tasks,
        **args.training_task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training_train_steps))

    num_training_examples = args.training_num_training_examples

    max_iters = args.training_train_steps-starting_step
    warmup_iters = 1000
    lr_decay_iters, min_lr = max_iters, args.training_learning_rate / 10.

    # get flops_per_token, non_emb_flops_per_token at initialization
    flops_per_token, non_emb_flops_per_token = compute_model_FLOPs_for_ICLRegression(model=model, curr_n_positions=model.block_size//2, curr_n_dims=n_dims, device=device)

    for i in pbar:
        iter_num = i+1

        # update LR
        mult = get_lr_mult(iter_num, args.training_learning_rate, args.training_learning_rate, warmup_iters, lr_decay_iters)
        global_lr = args.training_learning_rate * mult
        update_lrs(optimizer, mult)

        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training_task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()
        
        if args.debug:
            breakpoint()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)
        reset_lrs(optimizer, init_lrs)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        compute += flops_per_token * xs.shape[0] * xs.shape[1] * 2
        non_emb_compute += non_emb_flops_per_token * xs.shape[0] * xs.shape[1] * 2
             
        # eval loop
        if i % args.evaluation_interval == 0 and not args.test_run:            
            list_of_point_wise_loss = []
            point_wise_tags = list(range(curriculum.n_points))

            for eval_i in range(args.evaluation_samples_multiplier):
                xs = data_sampler.sample_xs(
                    curriculum.n_points,
                    bsize,
                    curriculum.n_dims_truncated,
                    **data_sampler_args,
                )
                task = task_sampler(**task_sampler_args)
                ys = task.evaluate(xs)

                loss_func = task.get_training_metric()
                
                loss, output = eval_step(model, xs.cuda(), ys.cuda(), loss_func)

                point_wise_loss_func = task.get_metric()
                point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

                list_of_point_wise_loss.append(point_wise_loss.cpu().numpy())

            wandb.log(
                {
                    "eval_pointwise/loss": dict(
                        zip(point_wise_tags, np.stack(list_of_point_wise_loss).mean(0))
                    ),
                    "eval_step": int(i/args.evaluation_interval),
                    "eval/compute": compute,
                    "eval/non_emb_compute": non_emb_compute,

                }
            )


        if i % args.wandb_log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "compute": compute,
                    "non_emb_compute": non_emb_compute,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training_save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
                "lr": global_lr,
            }
            torch.save(training_state, state_path)

        if (
            args.training_keep_every_steps > 0
            and i % args.training_keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    # ================================== default processing ================================== #
    if args.d_qk_head == -1:
        assert args.d_model != -1
        assert args.n_head != -1
        assert args.d_head == -1

        args.d_qk_head = args.d_model // args.n_head

    if args.d_head == -1:
        args.d_head = args.d_model // args.n_head
    
    if args.token_mixing_struct == "low_rank" or args.token_mixing_struct == "bilinear_BTT":
        args.mlr_rank_list = f"{args.d_head}|0"
    
    device = 'cuda'
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # we always set the number of (x_i, y_i) pairs to be double the number of the dimensionality of x_i
    args.n_positions = args.n_dims * 2

    # curriculum
    assert args.training_curriculum_dims_interval==args.training_curriculum_points_interval
    
    # autoset values in curriculum
    args.training_curriculum_dims_end = args.n_dims
    args.training_curriculum_points_end = args.n_positions
    # ================================== default processing ================================== #

    if args.test_run:
        args.training_curriculum_points_start = args.training_curriculum_points_end
        args.training_curriculum_dims_start = args.training_curriculum_dims_end
        args.training_train_steps = 100
    else:
        if args.training_curriculum_adaptive_inc:
            starting_inc = 1
            continue_update_inc = True

            # if training_train_steps and training_curriculum_dims_interval are fixed, we want the number of training steps involved in curriculum updates to be 6% of the total training_train_steps
            num_curriculum_updates = max(1, math.floor(args.training_curriculum_warmup_ratio * args.training_train_steps/args.training_curriculum_dims_interval))
            while continue_update_inc:
                if args.training_curriculum_dims_start + starting_inc * num_curriculum_updates >= args.training_curriculum_dims_end:
                    args.training_curriculum_dims_inc = starting_inc
                    continue_update_inc = False
                else:
                    starting_inc += 1

            args.training_curriculum_points_inc = args.training_curriculum_dims_inc * 2

            print(f"\n***** training_curriculum_dims_start={args.training_curriculum_dims_start}, training_curriculum_dims_end={args.training_curriculum_dims_end}, training_curriculum_dims_inc={args.training_curriculum_dims_inc}, training_curriculum_dims_interval={args.training_curriculum_dims_interval} *****\n")
            print(f"\n***** training_curriculum_points_start={args.training_curriculum_points_start}, training_curriculum_points_end={args.training_curriculum_points_end}, training_curriculum_points_inc={args.training_curriculum_points_inc}, training_curriculum_points_interval={args.training_curriculum_points_interval} *****\n")

        else:
            pass

        if args.debug:
            pass
        else:
            # save config
            run_id = args.training_resume_id
            if run_id is None:
                run_id = str(uuid.uuid4())

            out_dir = os.path.join(args.out_dir, run_id)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            args.out_dir = out_dir

            with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
                yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

            # initialize wandb
            wandb.init(
                dir=args.out_dir,
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=args.__dict__,
                notes=args.wandb_notes,
                name=args.wandb_name,
                resume=True,
            )
    
    # print args
    print(f"Running with: {args}")
    

    if True:
        # ! We set block_size = args.n_positions*2 ! #
        model_args = dict(n_dims=args.n_dims, n_layer=args.n_layer, n_head=args.n_head, d_head=args.d_head, 
                        d_qk_head=args.d_qk_head, d_model=args.d_model, block_size=args.n_positions*2, bias=args.bias,
                        vocab_size=args.vocab_size, dropout=args.dropout, split_qkv=args.split_qkv, do_qk_ln=args.do_qk_ln, 
                        manual_disable_flash_att=args.manual_disable_flash_att, token_mixing_struct=args.token_mixing_struct, 
                        mlr_rank_list=args.mlr_rank_list, mlr_divide_by_num_levels=args.mlr_divide_by_num_levels, 
                        mlr_block_divide_by_num_levels=args.mlr_block_divide_by_num_levels, link_function=args.link_function,
                        bilinear_mlr_muP_attn_logits_scaling=args.bilinear_mlr_muP_attn_logits_scaling, btt_tt_dim=args.btt_tt_dim, 
                        btt_tt_rank=args.btt_tt_rank, bilinear_btt_muP_attn_logits_scaling=args.bilinear_btt_muP_attn_logits_scaling, 
                        mha_SP_attn_logits_scaling=args.mha_SP_attn_logits_scaling, debug=args.debug)

        print("Initializing a new model from scratch")
        gptconf = GPTforICLRegressionConfig(**model_args)
        model = GPTforICLRegression(gptconf)

        # muP 
        model, optimizer = muPify(
            model, 
            device, device_type,
            # optimization arguments
            args.opt_name, args.training_learning_rate, args.weight_decay, args.beta1, args.beta2
        )
    else:
        raise ValueError("Let's just use structgpt for our experiments")

    model.train()
    print(model)

    train(model, optimizer, args, device)

    # if not args.test_run:
    #     all_metrics = get_run_metrics_new(args.out_dir, model=model)  # precompute metrics for eval
        # _ = get_run_metrics(args.out_dir)  # precompute metrics for eval

if __name__ == "__main__":
    args = parse_args()
    main(args)
