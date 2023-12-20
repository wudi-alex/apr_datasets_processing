import os

os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'
from tqdm import tqdm
from datasets import load_dataset
import subprocess
import tempfile
import re

dataset = load_dataset('h4iku/coconut_java2006_preprocessed', split='train')


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


## 去掉注释
def remove_comments(text):
    # 正则表达式匹配以 // 开头，后面跟任意字符（除了换行符），并以两个或更多空格结尾的子字符串
    pattern = r'//.*?(  +|\t)'
    # 使用空字符串替换匹配的文本
    return re.sub(pattern, '', text)


def format_context(sample):
    context = sample['context']
    context = remove_comments(context)

    if sample['rem'].replace('{', '').replace('}', '') == '':
        sample['formatted_context'] = None
        return sample

    if len(context) > 1000:
        sample['formatted_context'] = None
        return sample

    sample['formatted_context'] = format_java_code(context)
    return sample


format_dataset = dataset.map(format_context, num_proc=12)
format_dataset.save_to_disk(f"formatted_coconut")
