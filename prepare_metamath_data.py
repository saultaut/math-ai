from datasets import load_dataset
import argparse
import json
import io


# Create the parser
parser = argparse.ArgumentParser(description="Dataset size.")
# Define the --data_size argument
parser.add_argument('--data_size', type=int, default=10, help='The size of the data')
# Parse the arguments
args = parser.parse_args()
# Now you can use args.data_size in your script
print(f"Data size is set to: {args.data_size}")
      
dataset = load_dataset("meta-math/MetaMathQA")

n = args.data_size #number of datapoints

# Select a subset of the dataset
subset = dataset['train'].select(range(n))



# Convert the dataset to a list of dictionaries
data_list = [item for item in subset]

# Write to a JSONL file
file_path=f'./data/metamath/train_{n}.jsonl'
with open(file_path, 'w', encoding='utf-8') as jsonl_file:
    for item in data_list:
        json_string = json.dumps(item, ensure_ascii=False)
        jsonl_file.write(json_string + '\n')