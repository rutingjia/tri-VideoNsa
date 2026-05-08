#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export VIDEO_MAX_PIXELS=50176
export IMAGE_MAX_PIXELS=50176
export MAX_PIXELS=50176

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0

export SEQUENCE_PARALLEL_IMPL=ulysses
export CELOSS_PARALLEL_SIZE=16384

# Qwen3 decoder-side VideoNSA.
export VIDEONSA_ENABLE=1
export VIDEONSA_BLOCK_SIZE=64
export VIDEONSA_BLOCK_COUNTS=16
export VIDEONSA_WINDOW_SIZE=512
export VIDEONSA_PRINT_ONCE=1
export VIDEONSA_DEBUG_SHAPES=1
export VIDEONSA_FALLBACK_ON_ERROR=0

swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --model_type videonsa \
    --deepspeed zero2 \
    --freeze_vit false \
    --freeze_aligner false \
    --freeze_llm false \
    --attn_impl flash_attn \
    --sequence_parallel_size 1 \
    --use_hf true \
    --dataset datasets/overfit_10.jsonl \
    --train_dataloader_shuffle false \
    --dataset_shuffle false \
    --data_seed 42 \
    --dataset_num_proc 1 \
    --split_dataset_ratio 0 \
    --save_strategy steps \
    --save_steps 50 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 0 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --max_length 32768 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 1 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --loss_scale default \
    --load_from_cache_file false \
    --report_to tensorboard \
    --logging_dir logs/qwen3_videonsa_overfit_10 \
    --output_dir output/qwen3_videonsa_overfit_10 \
    --strict false \
    --max_steps 200