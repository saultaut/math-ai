# MATH OVM code - Outcome-supervised Value Models for Planning in Mathematical Reasoning

## Prepare the MetaMath dataset for trainging verifier in correct format

```
python prepare_metamath_data.py --data_size 1000
```


## Generate training labels for Verifier:
Run the commands:
```
git clone https://github.com/saultaut/math-ai.git
cd math-ai/
pip install -r requirements_runpod.txt
bash scripts/metamath/generate_metamath.sh
```
The output will be saved to `data/metamath/model_generation/train_500/` and file should be like responses_n1_*.jsonl

## Debuging the Verifier code

Connected remotly using VS Code to RunPod instace with GPU 3080. Everything worked. This uses small Opt-125m model.

```
python train_verifier_debug_metamath.py
```

## Train Verifier on MetaMath dataset:
Run the commands:
```
git clone https://github.com/saultaut/math-ai.git
cd math-ai/
pip install -r requirements_runpod.txt
bash scripts/metamath/train_verifier_metamath.sh
```

Output will be save in `/models/metamath/verifiers/`