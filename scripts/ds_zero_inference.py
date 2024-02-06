"""
Run OPT with huggingface or deepspeed.

Reference:
https://github.com/FMInference/FlexGen/blob/main/benchmark/hf_ds/hf_opt.py
"""

import os

os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'

import gc
import os

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from accelerate import init_empty_weights

from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, CodeLlamaTokenizer,
                          BloomForCausalLM, OPTForCausalLM, LlamaForCausalLM, )
from transformers.deepspeed import HfDeepSpeedConfig

from utils import (GB, get_quant_config, meta_to_cpu, )
from datasets import load_from_disk

deepspeed.init_distributed()

local_rank = dist.get_rank()
world_size = dist.get_world_size()

TOKEN = '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'


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def get_model_config(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=TOKEN)
    return config


def get_ds_model(
        model_name,
        batch_size,
        cpu_offload,
        disk_offload,
        offload_dir,
        dummy_weights,
        quant_bits,
        quant_group_size,
        pin_memory,
):
    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")

    if getattr(config, 'torch_dtype', None) is None:
        dtype = torch.float16
    else:
        dtype = config.torch_dtype

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size,  # 0,
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": batch_size,
        "wall_clock_breakdown": False,
    }

    if quant_bits:
        quant_config = get_quant_config(config, bits=quant_bits, group_size=quant_group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )

    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=pin_memory,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=9 * GB if config.model_type == 'bloom' else 2 * GB,
        )
        ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
        }

    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    get_accelerator().empty_cache()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        dummy_weights or model_name, torch_dtype=dtype, token=TOKEN, )

    model = model.eval()

    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(f"model.config = {model.config}")

    return model


def run_generation(
        model_name,
        batch_size,
        dataset,
        input_name,
        output_path,
        prompt_len,
        gen_len,
        num_outputs=1,
        temperature=1.0,
        do_sample=False,
        early_stopping=False,
        pin_memory=True,
        offload_dir="~/offload_dir",
        dummy=False,
        kv_offload=False,
        quant_bits=None,
        quant_group_size=64,
        pin_kv_cache=False,
        async_kv_offload=False,
        cpu_offload=True,
        disk_offload=False,
        skip_special_tokens=False,
        process_func=None,
):
    # Load tokenizer
    config = get_model_config(model_name)

    tokenizer = get_tokenizer(model_name)

    if dummy:
        filename = os.path.join(
            offload_dir, f"{model_name.replace('/', '-')}-hf-weights/"
        )
        if not os.path.exists(filename):
            print("create dummy weights")
            with init_empty_weights():
                if config.model_type == 'opt':
                    model = OPTForCausalLM(config)
                elif config.model_type in ["bloom", "bloom-7b1"]:
                    model = BloomForCausalLM(config)
                elif config.model_type == "llama":
                    model = LlamaForCausalLM(config)
                else:
                    raise ValueError(f"Unexpected model type: {config.model_type}")
            model.save_pretrained(
                filename, state_dict=meta_to_cpu(model.state_dict(), torch.float16)
            )
        dummy_weights = filename
    else:
        dummy_weights = None

    print(f"{local_rank} load model")
    with torch.no_grad():
        model = get_ds_model(
            model_name=model_name,
            batch_size=batch_size,
            cpu_offload=cpu_offload,
            disk_offload=disk_offload,
            offload_dir=offload_dir,
            dummy_weights=dummy_weights,
            quant_bits=quant_bits,
            quant_group_size=quant_group_size,
            pin_memory=pin_memory
        )

    if kv_offload:
        model.set_kv_cache_offload(True, gen_len, pin_kv_cache, async_kv_offload)

    model.config.pad_token = tokenizer.pad_token

    generate_kwargs = dict(max_new_tokens=gen_len, do_sample=do_sample, num_beams=num_outputs, temperature=temperature,
                           num_return_sequences=num_outputs, early_stopping=early_stopping,
                           eos_token_id=tokenizer.eos_token_id,
                           pad_token_id=tokenizer.pad_token_id)

    def _batch_encode(inputs):
        input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding="max_length",
                                                   max_length=prompt_len, truncation=True)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
        return input_tokens

    def _batch_gen(batch):
        inputs = batch[input_name]
        input_tokens = _batch_encode(inputs).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(**input_tokens, **generate_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
        if process_func:
            outputs = process_func(outputs)
        batch['gen'] = [outputs[i:i + num_outputs] for i in range(0, len(outputs), num_outputs)]
        # print(batch['gen_patch'])
        return batch

    print(f"start generating...")
    updated_dataset = dataset.map(_batch_gen, batched=True, batch_size=batch_size)
    updated_dataset.save_to_disk(f"{output_path}")
    print(f"{output_path} dataset saved")


if __name__ == '__main__':
    MODEL_PATH = '/projects/ksun3/dwu25/trained_models/classinfo_mutation_merged'
    DATASET_PATH = '/projects/ksun3/dwu25/apr_datasets_processing/java_mutation/data/classinfo_mutation'
    OUTPUT_PATH = '/projects/ksun3/dwu25/datasets/classinfo_mutation_test'
    dataset = load_from_disk(DATASET_PATH)['test']
    run_generation(
        model_name=MODEL_PATH,
        batch_size=64,
        dataset=dataset,
        input_name='input',
        output_path=OUTPUT_PATH,
        prompt_len=1024,
        gen_len=100,
        # num_outputs=10,
        # early_stopping=True,
        # do_sample=True,
        skip_special_tokens=True,
    )
