import json
import os
import re
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, List, Set, Dict, Any, Union
import transformers
import logging
from dataclasses import dataclass
import pathlib

from utils.datasets import read_jsonl, get_few_shot_prompt, left_pad_sequences, right_pad_sequences, mask_labels
from utils.constants import DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, IGNORE_INDEX
from utils.metamath.decoding import extract_answer, extract_test_answer


TEST_PROBLEM_PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )

def get_examples(data_dir, split):
    read_file = {
		'train_500': 'train_500.jsonl',
        'train_5000': 'train_5000.jsonl',
        'train_50': 'train_50.jsonl',
        'train': 'train.jsonl',
        'test_100': 'test_100.jsonl',
        'MATH_test_100': 'MATH_test_100.jsonl'
    }[split]
    
    path = os.path.join(data_dir, read_file)
    examples = read_jsonl(path)

    data = []
    for ex in examples:
        data.append({'question' : ex["query"] + "\n", 'answer': ex["response"]})

    print(f"{len(data)} {split} examples")
    return data



def get_test_examples(data_dir, split):
    read_file = {
        'test': 'test.jsonl',
        'MATH_test_100': 'MATH_test_100.jsonl'
    }[split]
    
    path = os.path.join(data_dir, read_file)
    examples = read_jsonl(path)

    
    data = []
    for ex in examples:
        temp_instr = TEST_PROBLEM_PROMPT.format(instruction=ex["instruction"])
        data.append({'question' : temp_instr, 'answer': ex["output"]})

    print(f"{len(data)} {split} examples")
    return data


def make_finetuning_generator_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass) -> Dict:
    train_dataset = FineTuningGeneratorDataset(
                        tokenizer=tokenizer, 
                        data_dir=data_args.data_dir, 
                        target_set=data_args.target_set,
                        loss_on_prefix=data_args.loss_on_prefix,
                    )
    val_dataset = None
    return dict(train_dataset=train_dataset, val_dataset=val_dataset)


def make_test_generator_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass, inference_args: dataclass) -> Dict:
    test_dataset = TestGeneratorDataset(
        tokenizer=tokenizer, 
        data_dir=data_args.data_dir,
        target_set=data_args.target_set
    )
    return test_dataset




class FineTuningGeneratorDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/metamath', 
        target_set: str = 'train',
        loss_on_prefix=True,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.loss_on_prefix = loss_on_prefix
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        print("+ [Dataset] Loading Training Data")
        self.examples = get_examples(self.data_dir, target_set)
        qns_str = [ex["question"] for ex in self.examples]
        ans_str = [ex["answer"] for ex in self.examples]
        
        print("+ [Dataset] Tokenizing Training Data")
        qns_tokens = tokenizer(qns_str, padding=False).input_ids
        ans_tokens = tokenizer(ans_str, padding=False, add_special_tokens=False).input_ids

        self.qns_str = qns_str
        self.ans_str = ans_str
        self.qns_tokens = qns_tokens
        self.ans_tokens = ans_tokens

        self.max_len = max([
                len(qns_tokens[i]) + len(ans_tokens[i]) + 1
                for i in range(len(qns_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        ans_tokens = self.ans_tokens[idx]

        input_ids = qn_tokens + ans_tokens + [self.eos_token_id]
        labels = input_ids

        masks = (
            ([1] if self.loss_on_prefix else [0]) * len(qn_tokens)
            + ([1] * len(ans_tokens))
            + ([1])
        )
        labels = mask_labels(labels, masks)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        return dict(input_ids=input_ids, labels=labels)

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids, attention_mask = right_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        labels = right_pad_sequences(labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )




class TestGeneratorDataset(torch.utils.data.Dataset):
    """Left Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/gsm8k', 
        target_set: str = None,
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.pad_token_id = tokenizer.pad_token_id

        print("+ [Dataset] Loading Training Data")
        self.examples = get_examples(self.data_dir, target_set)
        qns_str = [ex["question"] for ex in self.examples]
        ans_str = [ex["answer"] for ex in self.examples]
        
        gts_str = [extract_answer(ans) for ans in ans_str]

        print("+ [Dataset] Tokenizing Testing Data")
        qns_tokens = tokenizer(qns_str, padding=False).input_ids
        ans_tokens = tokenizer(ans_str, padding=False, add_special_tokens=False).input_ids

        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.ans_str = ans_str
        self.gts_str = gts_str

        self.max_len = max([
                len(qns_tokens[i]) + len(ans_tokens[i]) + 1
                for i in range(len(qns_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        qn_str = self.qns_str[idx]
        ans_str = self.ans_str[idx]
        gt = self.gts_str[idx]

        input_ids = torch.tensor(qn_tokens)
        return dict(
            idx=idx, 
            input_ids=input_ids, 
            input=qn_str,
            question=qn_str,
            reference=gt,
            record_data=dict(answer=ans_str, ground_truth=gt),
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        idx, input_ids, input, question, reference, record_data = tuple([instance[key] for instance in instances] for key in ("idx", "input_ids", "input", "question", "reference", "record_data"))
        record_data = {k: [instance[k] for instance in record_data] for k in record_data[0].keys()}

        input_ids, attention_mask = left_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        
        return dict(
            idx=idx,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input=input,
            question=question,
            reference=reference,
            record_data=record_data,
        )








