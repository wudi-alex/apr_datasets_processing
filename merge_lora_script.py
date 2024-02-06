import os
os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'
import argparse
import os
import sys
import time
import shutil
import torch
import transformers

sys.path.append("..")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_SPECIAL_TOKENS = {
    "gpt_neox": {

        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",

    },
    "llama": {

        "eos_token": "</s>",
        "pad_token": "<unk>",

    },
    "baichuan": {

        "eos_token": "</s>",
        "pad_token": "<unk>",

    },
    "starcoder": {

        "eos_token": "<|endoftext|>",
        "pad_token": "<fim_pad>",

    },
    "qwen": {

        "eos_token": "<|endoftext|>",
        "pad_token": "<|extra_1|>",

    },
    "chatglm2": {

        "eos_token": "</s>",
        "pad_token": "<unk>",

    },
}


def main(args):
    t0 = time.time()
    config = {"model_type": args.model_type}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        return_dict=True,
        device_map="auto"
    )
    print(base_model)

    # DEAL with eos_token_id and pad_token_id
    eos_token = MODEL_SPECIAL_TOKENS[config['model_type']]['eos_token']
    pad_token = MODEL_SPECIAL_TOKENS[config['model_type']]['pad_token']
    base_model.config.eos_token = eos_token
    base_model.config.pad_token = pad_token
    base_model.config.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    base_model.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    print(f"Finetuned eos_token: {eos_token}, eos_token_id: {tokenizer.convert_tokens_to_ids(eos_token)}")
    print(f"Finetuned pad_token: {pad_token}, pad_token_id: {tokenizer.convert_tokens_to_ids(pad_token)}")

    # merge, save model and tokenizer
    model_to_merge = PeftModel.from_pretrained(base_model, args.lora_adapter)
    merged_model = model_to_merge.merge_and_unload()
    print(merged_model.config)
    merged_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Merge finished: {args.save_path} saved, Cost {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to merge base model with LoRA adapter.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_adapter", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save merged model")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model type (e.g., llama/gpt_neox/qwen/chatglm2/starcoder)")

    args = parser.parse_args()
    main(args)
