#!/usr/bin/env bash
set -euo pipefail

# ===== 基础路径 =====
MODEL=/data/vjuicefs_ai_camera_album_ql/public_data/rutingjia/Qwen3-VL-2B-Instruct
DATA_DIR=/data/vjuicefs_ai_camera_album_ql/public_data/rutingjia/qwen3vl_video_sft
TRAIN_JSON=${DATA_DIR}/train.jsonl
VAL_JSON=${DATA_DIR}/val.jsonl
OUT=output/qwen3vl2b_llava_video_20k_lora

# ===== GPU 设置 =====
# 单卡就改成 0；两卡就 0,1；四卡就 0,1,2,3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# ===== Qwen3-VL 视频 token 控制 =====
# 这几个变量很重要，先保守一点，防止爆显存。
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-16}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 如果你装好了 flash-attn，可以运行：
#   ATTN_IMPL=flash_attn bash train_qwen3vl_video_lora.sh
# 没装 flash-attn 就默认 sdpa，更稳。
ATTN_IMPL=${ATTN_IMPL:-sdpa}

# 多卡时用 zero2；单卡时不加 deepspeed。
DS_ARGS=()
if [ "${NPROC_PER_NODE}" -gt 1 ]; then
  DS_ARGS=(--deepspeed zero2)
fi

swift sft \
  --model "${MODEL}" \
  --dataset "${TRAIN_JSON}" \
  --val_dataset "${VAL_JSON}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --gradient_accumulation_steps 8 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 4096 \
  --attn_impl "${ATTN_IMPL}" \
  --output_dir "${OUT}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  "${DS_ARGS[@]}"
