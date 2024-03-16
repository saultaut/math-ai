"""
This file is used to transoform original MATH test data set to MetaMathQA data set format.
"""

import json
import io


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


path ='/content/MATH_test_100.jsonl'
examples = read_jsonl(path)

data = []
for ex in examples:
    output_string = ex["output"]
    answer = extract_test_answer(output_string)
    final_output = output_string + "\nThe answer is: " + answer

    data.append({'query' : ex["instruction"], 'response': final_output})

print(f"{len(data)} {path} examples")

##### Save file to correct format

data_list = [item for item in data]

# Write to a JSONL file
with open('test_100.jsonl', 'w', encoding='utf-8') as jsonl_file:
    for item in data_list:
        json_string = json.dumps(item, ensure_ascii=False)
        jsonl_file.write(json_string + '\n')