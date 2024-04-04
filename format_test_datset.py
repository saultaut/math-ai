"""
This file is used to transoform original MATH test data set to MetaMathQA data set format.
"""

import json
import io
from utils.metamath.decoding import extract_answers, extract_answer, extract_test_answer, get_answer_label


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


path ='data/metamath/MATH_test_100.jsonl'
examples = read_jsonl(path)

data = []
for ex in examples:
    output_string = ex["output"]
    answer = extract_test_answer(output_string)
    final_output = output_string + "\nThe answer is: " + answer

    data.append({'query' : ex["instruction"], 'response': final_output})

print(f"{len(data)} {path} examples")

######### Save file to correct format

data_list = [item for item in data]

save_path = 'data/metamath/test_100.jsonl'
# Write to a JSONL file
with open(save_path, 'w', encoding='utf-8') as jsonl_file:
    for item in data_list:
        json_string = json.dumps(item, ensure_ascii=False)
        jsonl_file.write(json_string + '\n')



##################################################################################################
### Asser that saved file accuracy is 100 %
##################################################################################################

# load from orignal data format
ans_str = [ex["output"] for ex in examples]
gts = [extract_test_answer(ans) for ans in ans_str]


### parse formated response
model_completetions = [ex["response"] for ex in data]
answers = extract_answers(model_completetions)

corrs = [float(get_answer_label(answer, gt) == True) for answer, gt in zip(answers, gts)]

acc = (sum(corrs) / len(corrs))

print(f'Accuracy: {acc}')