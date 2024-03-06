#!/bin/bash

generator_id=facebook/opt-125m-ep2
verifier_id=n4-scahead-mse-lm-token
verifier_model_name_or_path=~/models/gsm8k/verifiers/${generator_id}-${verifier_id}


accelerate launch eval_with_verifier.py \
  --model_name_or_path ${verifier_model_name_or_path} \
  --data_dir data/gsm8k/model_generation \
  --verifier_output_dir eval_results/gsm8k/verifier \
  --generator_metric_dir eval_results/gsm8k/generator_with_verifier \
  --generator_id ${generator_id} \
  --target_set test_small \
  --batch_size 4 \
  --seed 42

