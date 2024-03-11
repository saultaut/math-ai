#!/bin/bash

generator_id=MetaMath-Mistral-7B
verifier_id=n100-scahead-mse-lm-token

n_beam=4
n_sampling_steps=4


model_name_or_path=meta-math/MetaMath-Mistral-7B
verifier_model_name_or_path=./models/metamath/verifiers/MetaMath-Mistral-7B-1


accelerate launch \
  --config_file ./configs/multi_gpu_inference.yaml \
  eval_generator_by_step_metamath.py \
  --model_name_or_path ${model_name_or_path} \
  --verifier_model_name_or_path ${verifier_model_name_or_path} \
  --dataset metamath \
  --data_dir data/metamath \
  --output_dir eval_results/metamath/generator_with_verifier \
  --target_set train_50 \
  --inference_mode beam \
  --batch_size 8 \
  --vs_batch_size 8 \
  --n_beam ${n_beam} \
  --n_sampling_steps ${n_sampling_steps} \
  --max_n_step 10 \
  --max_step_length 100 \
  --dedup_mode 0 \
  --do_sample True \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 1.0 \
  --max_new_tokens 1000 \
  --seed 42
