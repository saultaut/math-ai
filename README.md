# MATH OVM code - Outcome-supervised Value Models for Planning in Mathematical Reasoning

## Generate verifier training set from MATH dataset

Run the commands:
```
git clone https://github.com/saultaut/OVM.git
cd OVM/
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
git clone https://github.com/saultaut/OVM.git
cd OVM/
pip install -r requirements_runpod.txt
bash scripts/metamath/train_verifier_metamath.sh
```

Output will be save in `/models/metamath/verifiers/`