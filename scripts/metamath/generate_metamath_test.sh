#!/bin/bash

n_solutions=2

model_name_or_path=meta-math/MetaMath-Mistral-7B

accelerate launch \
  --config_file ./configs/multi_gpu_inference.yaml \
  generate_metamath.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset metamath \
  --data_dir data/metamath \
  --output_dir data/metamath/model_generation \
  --metric_output_dir eval_results/metamath/generator \
  --target_set train_50 \
  --n_solutions ${n_solutions} \
  --batch_size 4 \
  --do_sample True \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 1.0 \
  --max_new_tokens 1000
