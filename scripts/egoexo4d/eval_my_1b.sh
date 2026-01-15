#!/bin/bash

# set your environment variables here
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_PATH="${SCRIPT_DIR}/../.."
DATASET_DIR="${ROOT_PATH}/datasets"
CHECKPOINT_DIR="outputs/egoexo4d/keystep_providellm_1b_stage_2_lora_128_num_tokens_5_hand_2_obj_4_num_samples_16_epochs_20_run1"


N_GPUS=1
BATCH_SIZE_PER_GPU=1

cd $ROOT_PATH

EFFECTIVE_BATCH_SIZE=1
EFFECTIVE_BATCH_SIZE_PER_GPU=$((EFFECTIVE_BATCH_SIZE / N_GPUS))
GRAD_STEPS=$((EFFECTIVE_BATCH_SIZE_PER_GPU / BATCH_SIZE_PER_GPU))

export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled # prevent syncing

# --eval_datasets egoexo4d_keystep_val egoexo4d_keystep_test \
torchrun --nproc_per_node=$N_GPUS --standalone evaluate.py \
    --model_variant providellm_1b --stage 2 \
    --eval_datasets egoexo4d_keystep_val \
    --dataset_dir $DATASET_DIR \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --bf16 True --tf32 False --fp16 False \
    --num_samples 16 \
    --fine_tune "lora" --lora_r 128 \
    --override_output_dir $CHECKPOINT_DIR \
    --resume_from_checkpoint $CHECKPOINT_DIR