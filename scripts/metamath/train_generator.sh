#!/bin/bash

export WANDB_API_KEY=
export WANDB_PROJECT=Opt-125m-Generator-Finetune
export WANDB_ENTITY=

WANDB_MODE=online

model_name_or_path=facebook/opt-125m
save_generator_id=opt-125m-generator

save_dir=~/models/metamath/generators/${save_generator_id}/
export WANDB_NAME=${save_generator_id}




accelerate launch \
  --config_file ./configs/zero1_metamath.yaml \
  train_generator_metamath.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset metamath \
  --data_dir data/metamath \
  --target_set train_500 \
  --save_dir ${save_dir} \
  --num_train_epoches 2 \
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
  --logging_steps 4 \
  --seed 42
  
