#!/bin/bash

##### ? Standard Softmax Low-Rank Attention ? #####
# list_of_n_dims=(16) # 16 32 64 128
# list_of_d_model=(16 32 64 128 256 512)
# n_head=8 #8
# token_mixing_struct="low_rank"
# list_of_mlr_rank_list=("8|0") # will be updated in the code
# list_of_training_learning_rate=(0.001 0.0005 0.0001 0.00005)
# mlr_block_divide_by_num_levels=False
# mha_SP_attn_logits_scaling=True
##### ? ################################### ? #####

##### ? Softmax Bilinear MLR Attention ? #####
list_of_n_dims=(64) # 64 256
list_of_d_model=(256) # (32 64 128 256 512 1024)
n_head=8
token_mixing_struct="bilinear_MLR"
# list_of_mlr_rank_list=("16|16|12|8|8|4" "20|16|8|8|8|4" "24|12|8|8|8|4") 
list_of_mlr_rank_list=("8|4|4|16" "8|8|8|8" "16|8|4|4") 

list_of_training_learning_rate=(0.01 0.001 0.0001)
mlr_block_divide_by_num_levels=False
mha_SP_attn_logits_scaling=False
##### ? Softmax Bilinear MLR Attention ? #####

##### ? Softmax Bilinear BTT Attention ? #####
# list_of_n_dims=(256) # 64 256
# list_of_d_model=(16 32 64 128 256 512 1024)

# n_head=8
# token_mixing_struct="bilinear_BTT"
# list_of_mlr_rank_list=("8|0") # will be updated in the code
# # list_of_training_learning_rate=(0.0005 0.0001 0.00005)
# list_of_training_learning_rate=(0.0001 0.00005)

# mlr_block_divide_by_num_levels=False
# mha_SP_attn_logits_scaling=False
##### ? Softmax Bilinear BTT Attention ? #####

for n_dims in "${list_of_n_dims[@]}"; do 
        for d_model in "${list_of_d_model[@]}"; do 
                for training_learning_rate in "${list_of_training_learning_rate[@]}"; do 
                        for mlr_rank_list in "${list_of_mlr_rank_list[@]}"; do 
                                echo "n_dims: $n_dims, d_model: $d_model, n_head: $n_head, token_mixing_struct: $token_mixing_struct, mlr_rank_list: $mlr_rank_list, training_learning_rate: $training_learning_rate"

                                ##### struct_foundation_models #####
                                sbatch --job-name=ICLRegression \
                                        --nodes=1 \
                                        --ntasks-per-node=1 \
                                        --cpus-per-task=4 \
                                        --gres=gpu:1 \
                                        --time=47:59:59 \
                                        --mem=32G \
                                        --mail-type=ALL \
                                        --mail-user=yk2516@nyu.edu \
                                        --error=/scratch/yk2516/slurm/struct_foundation_models/%j_%a_%N.err \
                                        --output=/scratch/yk2516/slurm/struct_foundation_models/%j_%a_%N.out \
                                        --wrap="singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-50G-10M-struct3.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c 'source /ext3/env.sh; conda activate struct3; cd /scratch/yk2516/repos/structure/struct_foundation_models; export WANDB__SERVICE_WAIT=300; python train_ICL.py --n_dims=\"$n_dims\" --n_head=\"$n_head\" --d_model=\"$d_model\" --token_mixing_struct=\"$token_mixing_struct\" --mlr_rank_list=\"$mlr_rank_list\" --mlr_divide_by_num_levels=False --mlr_block_divide_by_num_levels=\"$mlr_block_divide_by_num_levels\" --bilinear_mlr_muP_attn_logits_scaling=False --mha_SP_attn_logits_scaling=\"$mha_SP_attn_logits_scaling\" --training_learning_rate=\"$training_learning_rate\" --training_curriculum_adaptive_inc=True --training_num_training_examples=32000064 --wandb_entity=yilunkuang --out_dir=/scratch/yk2516/repos/structure/struct_foundation_models/out/ICL_regression'"
                        done
                done
        done
done
