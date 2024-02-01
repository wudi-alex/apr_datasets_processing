from datasets import *

dataset = load_from_disk('data/classinfo_mutation_full')

import os

os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'
from transformers import AutoTokenizer

MODEL_NAME = 'codellama/CodeLlama-7b-hf'
TOKEN = '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'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)


def get_token_len(input):
    input_tokens = tokenizer(input, return_tensors="pt")["input_ids"]
    return len(input_tokens[0])


dataset = dataset.filter(
    lambda x: get_token_len(x['input']) < 1024, num_proc=40)

dataset = dataset.train_test_split(test_size=0.05)

selected_features = ['input', 'output']


# 使用 map 函数选择特定的特征
def select_features(example):
    return {feature: example[feature] for feature in selected_features}


training_dataset = dataset.map(select_features)

training_dataset.save_to_disk('data/classinfo_mutation')
print('done')
