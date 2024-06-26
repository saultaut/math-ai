#!/bin/bash

export WANDB_API_KEY=
export WANDB_PROJECT=GSM8K-Generator-Finetune
export WANDB_ENTITY=


# model_name_or_path=facebook/opt-1.3b
model_name_or_path=facebook/opt-125m
save_generator_id=facebook/opt-125m-ep2

save_dir=~/models/gsm8k/generators/${save_generator_id}/
export WANDB_NAME=${save_generator_id}




accelerate launch train_generator.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset gsm8k \
  --data_dir data/gsm8k \
  --target_set train_small \
  --save_dir ${save_dir} \
  --num_train_epoches 3 \
  --eval_steps 200 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_steps 200 \
  --save_best False \
  --save_total_limit 0 \
  --logging_dir ./wandb \
  --logging_steps 8 \
  --seed 42
  
