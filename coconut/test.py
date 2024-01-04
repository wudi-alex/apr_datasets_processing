import os

os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

MODEL_NAME = 'codellama/CodeLlama-7b-hf'
TOKEN = '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'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)


def get_token_len(input):
    input_tokens = tokenizer(input, return_tensors="pt")["input_ids"]
    return len(input_tokens[0])


reformat_dataset = load_from_disk('data/reformatted_coconut')

print('filter patch')
rm_dataset = reformat_dataset.filter(lambda x: x['add_patch'] != None).filter(
    lambda x: not x['rem_patch'].startswith('{'))
print('filter infill context')
rm_dataset = reformat_dataset.filter(lambda x: x['infill_context'] != None).filter(
    lambda x: x['infill_context'].count('<INFILL>') == 1)
print('filter lenghth')
rm_dataset = rm_dataset.filter(
    lambda x: get_token_len(x['prompt']) <= 500 and get_token_len(x['prompt']) > 20 and get_token_len(
        x['chosen']) <= 100 and get_token_len(x['rejected']) <= 100)
print('filter protected')
rm_dataset = rm_dataset.filter(
    lambda x: 'protected' not in x['rem_patch'] and 'private' not in x['rem_patch'] and 'public' not in x['rem_patch'])


def process_dschat(sample):
    pre, suffix = sample['infill_context'].split('<INFILL>')[0], sample['infill_context'].split('<INFILL>')[1]
    # format as "<PRE> {pre} <SUF>{suf} <MID>"
    sample['prompt'] = "<PRE> " + pre + ' <SUF>' + suffix + ' <MID>'
    return sample


rm_dataset = rm_dataset.map(process_dschat)
rm_dataset = rm_dataset.rename_columns({'rem_patch': 'rejected', 'add_patch': 'chosen'})
rm_dataset = rm_dataset.train_test_split(test_size=0.05)
rm_dataset.save_to_disk(f"data/apr_rlhf_coconut")
print(rm_dataset['train'][0])
