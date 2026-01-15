#!/bin/bash

# set your environment variables here
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_PATH="${SCRIPT_DIR}/../.."
DATASET_DIR="${ROOT_PATH}/datasets"
# CHECKPOINT_DIR="outputs/egoclip/stage1_providellm_1b_stage_1_connector_num_tokens_5_hand_2_obj_4_wt_1.0_num_samples_4_epochs_2_run2/checkpoint-14000"
# CHECKPOINT_DIR="outputs/egoclip/stage1_providellm_1b_stage_1_connector_num_tokens_5_hand_2_obj_4_wt_1.0_num_samples_4_epochs_2_run2/checkpoint-21000"
# CHECKPOINT_DIR="outputs/egoclip/stage1_providellm_1b_stage_1_connector_num_tokens_5_hand_2_obj_4_wt_1.0_num_samples_4_epochs_2_run2/checkpoint-28000"


CHECKPOINT_DIR="outputs/ego4d/debug"


N_GPUS=1
BATCH_SIZE_PER_GPU=1

cd $ROOT_PATH

EFFECTIVE_BATCH_SIZE=8
EFFECTIVE_BATCH_SIZE_PER_GPU=$((EFFECTIVE_BATCH_SIZE / N_GPUS))
GRAD_STEPS=$((EFFECTIVE_BATCH_SIZE_PER_GPU / BATCH_SIZE_PER_GPU))

export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled # prevent syncing
# ego4d_onlinestep_val
# --eval_datasets ego4d_onlinestep_val \
torchrun --nproc_per_node=$N_GPUS --standalone debug.py \
    --model_variant interleave_providellm_1b \
    --eval_datasets ego4d_onlinestep_val \
    --dataset_dir $DATASET_DIR \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --bf16 True --tf32 False --fp16 False \
    --fine_tune connector \
    --override_output_dir $CHECKPOINT_DIR \
    --resume_from_checkpoint $CHECKPOINT_DIR