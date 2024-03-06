#!/bin/bash

generator_id=facebook/opt-125m-ep2
n_solutions=4

model_name_or_path=~/models/gsm8k/generators/${generator_id}

accelerate launch generate_paths_and_eval.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset gsm8k \
  --data_dir data/gsm8k \
  --output_dir data/gsm8k/model_generation \
  --metric_output_dir eval_results/gsm8k/generator \
  --target_set train_small \
  --n_solutions ${n_solutions} \
  --batch_size 4 \
  --do_sample True \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 1.0 \
  --max_new_tokens 400
