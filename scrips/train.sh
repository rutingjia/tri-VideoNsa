PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1 
export VIDEO_MAX_PIXELS=50176
export IMAGE_MAX_PIXELS=50176
export MAX_PIXELS=50176

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0

export SEQUENCE_PARALLEL_IMPL=ulysses
export CELOSS_PARALLEL_SIZE=16384 

# Use a Swift-supported model_type that matches Qwen3-VL architecture.
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
    --dataset /data/vjuicefs_ai_camera_album_ql/public_data/rutingjia/VideoNSA2.5/ms-swift/datasets/frames/LLaVA-Video-178K/ActivityNet-QA/activitynet/train \
    --train_dataloader_shuffle false \
    --dataset_shuffle false \
    --data_seed 42 \
    --dataset_num_proc 4 \
    --split_dataset_ratio 0 \
    --save_strategy steps \
    --save_steps 200 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-06 \
    --gradient_accumulation_steps 8 \
    --eval_steps 0 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 32768 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 1 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --loss_scale default \
    --load_from_cache_file true \
    --report_to tensorboard \
    --logging_dir logs/ \
    --output_dir output/ \
    --strict true 
