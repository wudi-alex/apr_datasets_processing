import os
from tqdm import tqdm

os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'
from datasets import load_dataset, load_from_disk
import subprocess
import tempfile


def format_java_code(java_code):
    # 创建一个临时文件来保存Java代码
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.java', delete=False) as temp_file:
        temp_file_name = temp_file.name
        # print(java_code)
        temp_file.write('public class A {  ' + java_code + '}')

    # 使用java命令行工具格式化代码
    try:
        subprocess.run(
            ['java', '-jar', '/home/dwu25/google-java-format-1.19.1-all-deps.jar', '--replace', temp_file_name],
            check=True)

        # 读取格式化后的代码
        with open(temp_file_name, 'r') as file:
            formatted_code = file.read()
        return formatted_code[19:-3]
    except subprocess.CalledProcessError as e:
        # print(java_code)
        return None
    finally:
        # 删除临时文件
        if temp_file_name:
            os.remove(temp_file_name)


def remove_spaces_newlines_and_get_indices(java_code):
    cleaned_code = ""
    indices = []
    for index, char in enumerate(java_code):
        if char not in [' ', '\n', '\r', '\t']:
            cleaned_code += char
            indices.append(index)
    return indices, cleaned_code


def find_substring_indices(main_string, substring):
    start_index = main_string.find(substring)

    # 如果找不到子字符串，则返回-1
    if start_index == -1:
        return -1, -1

    end_index = start_index + len(substring) - 1
    return start_index, end_index


def replace_pacth(java_code, java_patch):
    rm_code_ind_lst, rm_java_code = remove_spaces_newlines_and_get_indices(java_code)
    _, rm_java_patch = remove_spaces_newlines_and_get_indices(java_patch)
    start_ind, end_ind = find_substring_indices(rm_java_code, rm_java_patch)
    code_start_ind, code_end_ind = rm_code_ind_lst[start_ind], rm_code_ind_lst[end_ind]
    patch = java_code[code_start_ind:code_end_ind + 1]
    return patch


def get_infill_code(code, rem_patch, add_patch):
    rem_patch = replace_pacth(code, rem_patch)
    return rem_patch, code.replace(rem_patch, '<INFILL>'), code.replace(rem_patch, add_patch)


def reformat(sample):
    rem_patch, add_patch, java_code = sample['rem'], sample['add'], sample['formatted_context']
    sample['rem_patch'], sample['infill_context'], add_context = get_infill_code(java_code, rem_patch, add_patch)
    formatted_add_context = format_java_code(add_context)
    if formatted_add_context:
        add_patch = replace_pacth(formatted_add_context, add_patch)
        sample['add_patch'] = add_patch
    else:
        sample['add_patch'] = None
    return sample


formatted_dataset = load_from_disk('data/formatted_coconut')
formatted_dataset = formatted_dataset.filter(lambda x: x['formatted_context'] != None)
reformat_dataset = formatted_dataset.map(reformat, num_proc=12)
reformat_dataset.save_to_disk(f"data/reformatted_coconut")
