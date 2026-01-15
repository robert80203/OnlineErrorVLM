#!/bin/bash

# set your environment variables here
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_PATH="${SCRIPT_DIR}/../.."
DATASET_DIR="${ROOT_PATH}/datasets"
PRETRAINED_DIR="outputs/egoclip/stage1_providellm_1b_stage_1_connector_num_tokens_5_hand_2_obj_4_wt_1.0_num_samples_4_epochs_2_run2/checkpoint-28000"



N_GPUS=1
BATCH_SIZE_PER_GPU=4 #64

cd $ROOT_PATH

EFFECTIVE_BATCH_SIZE=4 #128
EFFECTIVE_BATCH_SIZE_PER_GPU=$((EFFECTIVE_BATCH_SIZE / N_GPUS))
GRAD_STEPS=$((EFFECTIVE_BATCH_SIZE_PER_GPU / BATCH_SIZE_PER_GPU))

export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled # prevent syncing

# num_samples = num of frames for each clip
# --model_variant providellm_1b \

torchrun --nproc_per_node=$N_GPUS --standalone debug.py \
    --deepspeed configs/deepspeed/zero1.json \
    --model_variant stream_providellm_1b \
    --train_datasets egoexo4d_keystep_train --eval_datasets egoexo4d_keystep_val egoexo4d_keystep_test \
    --dataset_dir $DATASET_DIR \
    --num_train_epochs 30 --per_device_train_batch_size $BATCH_SIZE_PER_GPU --per_device_eval_batch_size 1 \
    --bf16 True --tf32 False --fp16 False \
    --num_samples 8 \
    # --gradient_accumulation_steps $GRAD_STEPS --gradient_checkpointing False \
    # --eval_strategy no --prediction_loss_only False \
    # --save_strategy no --save_steps 1000 \
    # --learning_rate 1.5e-4 --optim adamw_torch --lr_scheduler_type cosine --warmup_ratio 0.05 \
    # --logging_steps 50 --dataloader_num_workers 6 \
    # --bf16 True --tf32 False --fp16 False \
    # --report_to wandb \
    # --fine_tune "lora" --lora_r 128 \
    # --num_samples 16 \
    # --resume_from_checkpoint $PRETRAINED_DIR