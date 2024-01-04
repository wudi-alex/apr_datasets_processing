import os

os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'
from datasets import load_dataset, load_from_disk

dataset = load_from_disk('data/apr_rlhf_coconut')

print(dataset['train'][0])['prompt']

def process(sample):
    pre, suffix = sample['prompt'].split('<FILL_ME>')[0], sample['prompt'].split('<FILL_ME>')[1]
    sample['prompt'] = "<PRE> " + pre + " <SUF>" + suffix + " <MID>"
    return sample


dataset['train'] = dataset['train'].map(process)
dataset['test'] = dataset['train'].map(process)

dataset.save_to_disk(f"data/apr_rlhf_coconut")

print(dataset['train'][0])['prompt']
