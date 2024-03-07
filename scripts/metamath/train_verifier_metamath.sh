#!/bin/bash

export WANDB_API_KEY=
export WANDB_PROJECT=MetaMath-Verifier
export WANDB_ENTITY=


n_solution=2
generator_id=MetaMath-Mistral-7B
save_verifier_id=n${n_solution}-scahead-mse-lm-token


checkpoint_dir=facebook/opt-125m

experimentID=1
final_id=${generator_id}-${save_verifier_id}
save_dir=~/models/metamath/verifiers/${generator_id}-${experimentID}
export WANDB_NAME=${generator_id}-${experimentID}



accelerate launch \
  --config_file ./configs/zero1_metamath.yaml \
  train_verifier_metamath.py \
  --model_name_or_path ${checkpoint_dir} \
  --data_dir data/metamath/model_generation \
  --target_set train_500 \
  --save_dir ${save_dir} \
  --generator_id ${generator_id} \
  --dedup True \
  --per_problem_sampling_solution ${n_solution} \
  --loss_level token \
  --loss_on_llm True \
  --num_train_epoches 1 \
  --eval_steps 1000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_epoches 1 \
  --save_best False \
  --save_total_limit 0 \
  --logging_dir ./wandb \
  --logging_steps 20 \
  --seed 42

