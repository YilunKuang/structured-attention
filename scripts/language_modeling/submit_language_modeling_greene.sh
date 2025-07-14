#!/bin/bash

##### ? Standard Softmax Low-Rank Attention ? #####
# token_mixing_struct="low_rank"
# list_of_d_model=(512) # (256 512 768 1024 1536 2048)
# block_size=1024
# d_qk_head=-1 # (16 for 16|0|0|0); (32 for 32|0); (96 for 32|32); (240 for 16|16|16|16); (-1 for d_qk_head set to d_head)
# list_of_mlr_rank_list=("64|0")
# list_of_mlr_divide_by_num_levels=(False)
# mlr_block_divide_by_num_levels=False
# list_of_init_lr=(3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4) # (3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4 3e-5)
# link_function="softmax"
# bilinear_mlr_muP_attn_logits_scaling=False
# mha_SP_attn_logits_scaling=False
# batch_size=4
# sliding_block_size=512
# gswa_rank_list="32|32"
# init_from="scratch"
# out_dir="out"
##### ? ################################### ? #####

##### ? Standard Softmax Low-Rank Sliding Window Attention ? #####
# token_mixing_struct="low_rank_with_sliding_window"
# list_of_d_model=(256 384 512 768)
# block_size=1024
# d_qk_head=-1
# list_of_mlr_rank_list=("64|0")
# list_of_mlr_divide_by_num_levels=(False)
# mlr_block_divide_by_num_levels=False
# list_of_init_lr=(3e-3 1.65e-3) # (3e-2 1.65e-2 3e-3 1.65e-3)
# link_function="softmax"
# bilinear_mlr_muP_attn_logits_scaling=False
# mha_SP_attn_logits_scaling=False
# batch_size=4
# sliding_block_size=512
# gswa_rank_list="32|32"
# init_from="scratch"
# out_dir="out"
##### ? ################################### ? #####

##### ? Standard Softmax Low-Rank GSWA Attention ? #####
# token_mixing_struct="low_rank_with_gswa"
# list_of_d_model=(256 384 512 768)
# block_size=1024
# d_qk_head=-1
# list_of_mlr_rank_list=("64|0")
# list_of_mlr_divide_by_num_levels=(False)
# mlr_block_divide_by_num_levels=False
# list_of_init_lr=(3e-2 1.65e-2 3e-3 1.65e-3) # (3e-2 1.65e-2 3e-3 1.65e-3)
# link_function="softmax"
# bilinear_mlr_muP_attn_logits_scaling=False
# mha_SP_attn_logits_scaling=False
# batch_size=4
# sliding_block_size=512
# gswa_rank_list="32|32"
# init_from="scratch"
# out_dir="out"
##### ? ################################### ? #####

##### ? Standard Softmax Low-Rank Sequential GSWA Attention ? #####
# token_mixing_struct="low_rank_with_sequential_gswa"
# list_of_d_model=(256 384 512 768)
# block_size=1024
# d_qk_head=-1
# list_of_mlr_rank_list=("64|0")
# list_of_mlr_divide_by_num_levels=(False)
# mlr_block_divide_by_num_levels=False
# # list_of_init_lr=(3e-2 1.65e-2 3e-3 1.65e-3) # (3e-2 1.65e-2 3e-3 1.65e-3)
# list_of_init_lr=(3e-2 1.65e-2) # (3e-2 1.65e-2 3e-3 1.65e-3)
# link_function="softmax"
# bilinear_mlr_muP_attn_logits_scaling=False
# mha_SP_attn_logits_scaling=False
# batch_size=4
# sliding_block_size=128
# gswa_rank_list="32|32"
# init_from="scratch"
# out_dir="out"
##### ? ################################### ? #####

##### ? Softmax Multi-Level Low-Rank Attention ? #####
token_mixing_struct="multi_level_low_rank"
list_of_d_model=(768) # (256 384 512 768 1024 1536 2048)
block_size=1024
d_qk_head=-1
# list_of_mlr_rank_list=("28|24|8|4" "24|20|12|8" "20|18|14|12" "16|16|16|16" "12|14|18|20" "8|12|20|24" "4|8|24|28")
# list_of_mlr_rank_list=("56|8" "48|16" "40|24" "32|32" "24|40" "16|48" "8|56")
# list_of_mlr_rank_list=("32|8|6|4|4|4|4|2" "24|12|8|6|4|4|4|2" "14|12|8|8|8|8|4|2")
# list_of_mlr_rank_list=("16|16|12|8|8|4" "20|16|8|8|8|4" "24|12|8|8|8|4")  # "36|12|8|4|2|2"
list_of_mlr_rank_list=("32|8|6|4|4|4|4|2")

# list_of_mlr_rank_list=("48|8|4|4" "40|16|4|4" "32|20|8|4")
# list_of_mlr_rank_list=("32|8|6|4|4|4|4|2" "32|10|8|8|4|2" "32|16|12|4" "32|32")
# list_of_mlr_rank_list=("24|12|8|6|4|4|4|2" "40|6|4|4|4|2|2|2")

# list_of_mlr_rank_list=("48|8|4|4" "40|16|4|4" "32|20|8|4")

# list_of_mlr_rank_list=("32|32" "16|16|16|16")
list_of_mlr_divide_by_num_levels=(False) # True
mlr_block_divide_by_num_levels=True # True

# list_of_init_lr=(1.65e-3)

# list_of_init_lr=(3e-3) # 3e-4)
list_of_init_lr=(1.65e-2) # 3e-4)
# list_of_init_lr=(3e-2 1.65e-2 3e-3 1.65e-3) # 3e-4)
# list_of_init_lr=(3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4) # 3e-4)
# list_of_init_lr=(1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4) # (3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4 3e-5)
# list_of_init_lr=(3e-3) # (3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4 3e-5)
link_function="softmax" # identity or softmax
bilinear_mlr_muP_attn_logits_scaling=False
mha_SP_attn_logits_scaling=False
batch_size=4
sliding_block_size=128
gswa_rank_list="32|32"
init_from="resume" # "scratch"
out_dir="out/checkpoint_2025-01-18_21-11-11_59X6qO" # "out"
##### ? ###################################### ? #####

##### ? Softmax Bilinear MLR Attention ? #####
# token_mixing_struct="bilinear_MLR"
# list_of_d_model=(1024) # (256 512 768 1024 1536 2048)
# block_size=256
# d_qk_head=-1
# # list_of_mlr_rank_list=("28|24|8|4" "24|20|12|8" "20|18|14|12" "16|16|16|16" "12|14|18|20" "8|12|20|24" "4|8|24|28")
# # list_of_mlr_rank_list=("56|8" "48|16" "40|24" "32|32" "24|40" "16|48" "8|56")
# # list_of_mlr_rank_list=("32|8|6|4|4|4|4|2" "24|12|8|6|4|4|4|2" "14|12|8|8|8|8|4|2")
# # list_of_mlr_rank_list=("16|16|12|8|8|4" "20|16|8|8|8|4" "24|12|8|8|8|4")  # "36|12|8|4|2|2"

# # list_of_mlr_rank_list=("48|8|4|4" "40|16|4|4" "32|20|8|4")
# list_of_mlr_rank_list=("56|8" "48|16" "40|24" "48|8|4|4" "40|16|4|4" "32|20|8|4")

# # list_of_mlr_rank_list=("32|32" "16|16|16|16")
# list_of_mlr_divide_by_num_levels=(False) # True
# mlr_block_divide_by_num_levels=False
# list_of_init_lr=(3e-2 1.65e-2 3e-3 1.65e-3 3e-4) # (3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4 3e-5)
# # list_of_init_lr=(1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4) # (3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4 3e-5)
# # list_of_init_lr=(3e-3) # (3e-1 1.65e-1 3e-2 1.65e-2 3e-3 1.65e-3 3e-4 1.65e-4 3e-5)
# link_function="softmax" # identity or softmax
# bilinear_mlr_muP_attn_logits_scaling=True
# mha_SP_attn_logits_scaling=False
# batch_size=64
# sliding_block_size=512
# gswa_rank_list="32|32"
# init_from="scratch"
# out_dir="out"
##### ? ###################################### ? #####


for init_lr in "${list_of_init_lr[@]}"; do 
        for d_model in "${list_of_d_model[@]}"; do 
                for mlr_rank_list in "${list_of_mlr_rank_list[@]}"; do 
                        for mlr_divide_by_num_levels in "${list_of_mlr_divide_by_num_levels[@]}"; do 
                                echo "init_lr: $init_lr, token_mixing_struct: $token_mixing_struct, d_model: $d_model, mlr_rank_list: $mlr_rank_list, mlr_divide_by_num_levels: $mlr_divide_by_num_levels, mlr_block_divide_by_num_levels: $mlr_block_divide_by_num_levels, d_qk_head: $d_qk_head, link_function: $link_function"

                                ##### struct_foundation_models #####
                                sbatch --job-name=struct_foundation_models \
                                        --nodes=1 \
                                        --ntasks-per-node=1 \
                                        --cpus-per-task=4 \
                                        --gres=gpu:a100:1 \
                                        --time=47:59:59 \
                                        --mem=32G \
                                        --mail-type=ALL \
                                        --mail-user=yk2516@nyu.edu \
                                        --error=/scratch/yk2516/slurm/struct_foundation_models/%j_%a_%N.err \
                                        --output=/scratch/yk2516/slurm/struct_foundation_models/%j_%a_%N.out \
                                        --wrap="singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-50G-10M-struct3.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c 'source /ext3/env.sh; conda activate struct3; cd /scratch/yk2516/repos/structure/struct_foundation_models; export WANDB__SERVICE_WAIT=300; python train.py config/struct_configs/train_struct_gpt2.py --d_model=\"$d_model\" --block_size=\"$block_size\" --token_mixing_struct=\"$token_mixing_struct\" --mlr_rank_list=\"$mlr_rank_list\" --mlr_divide_by_num_levels=\"$mlr_divide_by_num_levels\" --mlr_block_divide_by_num_levels=\"$mlr_block_divide_by_num_levels\" --mha_SP_attn_logits_scaling=\"$mha_SP_attn_logits_scaling\" --batch_size=\"$batch_size\" --init_lr=\"$init_lr\" --d_qk_head=\"$d_qk_head\" --link_function=\"$link_function\" --bilinear_mlr_muP_attn_logits_scaling=\"$bilinear_mlr_muP_attn_logits_scaling\" --sliding_block_size=\"$sliding_block_size\" --gswa_rank_list=\"$gswa_rank_list\" --init_from=\"$init_from\" --out_dir=\"$out_dir\"'"
                        done
                done
        done
done
