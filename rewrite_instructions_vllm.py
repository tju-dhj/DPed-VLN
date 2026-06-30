#!/usr/bin/env python3
"""
使用Qwen3.5-VL模型批量重写DPed数据集的instruction
将包含行人相关内容的instruction转换为L1语言指令（不包含行人/躲避指导）

Usage:
    python rewrite_instructions_vllm.py --input /path/to/input --output /path/to/output --limit 10
"""

import os
import gzip
import json
import glob
import argparse
from tqdm import tqdm
from datetime import datetime
import re

# 输入输出目录
INPUT_DIR = "/share/home/u19666033/dhj/DPed_pro/dped_pro/train"
OUTPUT_DIR = "/share/home/u19666033/dhj/DPed_pro/dped_pro/train_rewritten"

# 模型路径
MODEL_PATH = "/share/home/u19666033/dhj/models/Qwen3.6-27B"

SYSTEM_PROMPT = """You are a navigation instruction rewriter. Your task is to rewrite navigation instructions by REMOVING all references to pedestrians, people, or avoidance instructions.

Requirements:
1. Remove ALL mentions of pedestrians, people, persons, man, woman, avoid, wait for, etc.
2. Keep ONLY the navigation directions (turn left, turn right, go straight, move forward, etc.)
3. Make it a coherent, natural language instruction
4. Keep the same general structure and number of steps if possible
5. If the instruction only contains navigation without pedestrians, keep it mostly unchanged

Original instruction: {instruction}

Rewritten instruction (L1 language, no pedestrian references):"""


def init_vllm_model():
    """初始化vLLM模型"""
    try:
        from vllm import LLM, SamplingParams
        print("使用vLLM加速...")
        
        # 初始化vLLM模型
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=8192,
            tensor_parallel_size=1,  # 根据GPU数量调整
            trust_remote_code=True,
            dtype="bfloat16",
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        return llm, sampling_params, "vllm"
    except ImportError:
        print("vLLM未安装，使用transformers...")
        return None, None, "transformers"


def init_transformers_model():
    """初始化transformers模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("正在加载Qwen模型 (transformers)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )
    print("模型加载完成!")
    return model, tokenizer


def rewrite_instruction_vllm(instruction, llm, sampling_params):
    """使用vLLM重写instruction"""
    prompt = SYSTEM_PROMPT.format(instruction=instruction)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = llm.generate([text], sampling_params)
    return outputs[0].outputs[0].text.strip()


def rewrite_instruction_transformers(instruction, model, tokenizer):
    """使用transformers重写instruction"""
    prompt = SYSTEM_PROMPT.format(instruction=instruction)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def has_pedestrian_reference(text):
    """检查文本是否包含行人相关引用"""
    pedestrian_keywords = [
        'pedestrian', 'pedestrians', 'person', 'persons', 'people',
        'man', 'woman', 'men', 'women', 'avoid', 'wait for',
        'walking near', 'walking towards', 'walking down', 'walking away',
        'standing by', 'near center', 'near wall', 'on staircase',
        'at the top of', 'on the right side', 'on the left side',
        'human', 'crowd', 'walker'
    ]
    
    text_lower = text.lower()
    return any(kw in text_lower for kw in pedestrian_keywords)


def load_episodes_from_file(filepath):
    """从json.gz文件加载episodes"""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('episodes', []), data


def save_episodes_to_file(filepath, episodes, original_data):
    """保存episodes到json.gz文件"""
    original_data['episodes'] = episodes
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(original_data, f, indent=2, ensure_ascii=False)


def process_single_instruction(instruction, model, tokenizer, backend):
    """处理单条instruction"""
    if not has_pedestrian_reference(instruction):
        return instruction  # 不需要重写
    
    try:
        if backend == "vllm":
            return rewrite_instruction_vllm(instruction, model, tokenizer)
        else:
            return rewrite_instruction_transformers(instruction, model, tokenizer)
    except Exception as e:
        print(f"Error rewriting: {e}")
        return instruction  # 出错时返回原指令


def process_file(input_path, output_path, model, tokenizer, backend):
    """处理单个json.gz文件"""
    episodes, original_data = load_episodes_from_file(input_path)
    
    modified_count = 0
    for episode in episodes:
        original_instruction = episode.get('instruction', '')
        if original_instruction and has_pedestrian_reference(original_instruction):
            new_instruction = process_single_instruction(
                original_instruction, model, tokenizer, backend
            )
            episode['instruction'] = new_instruction
            episode['original_instruction'] = original_instruction
            modified_count += 1
    
    save_episodes_to_file(output_path, episodes, original_data)
    return modified_count


def main():
    parser = argparse.ArgumentParser(description='重写DPed数据集的instruction')
    parser.add_argument('--input', default=INPUT_DIR, help='输入目录')
    parser.add_argument('--output', default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--limit', type=int, default=None, help='限制处理文件数量')
    parser.add_argument('--test', action='store_true', help='测试模式，只处理3个文件')
    parser.add_argument('--sample', type=int, default=5, help='每个文件展示的样本数')
    args = parser.parse_args()
    
    # 初始化模型
    llm = None
    sampling_params = None
    backend = "transformers"
    
    llm, sampling_params, backend = init_vllm_model()
    
    if llm is None:
        model, tokenizer = init_transformers_model()
    else:
        model, tokenizer = llm, sampling_params  # vLLM使用不同的接口
    
    # 获取所有文件
    json_files = sorted(glob.glob(os.path.join(args.input, "*.json.gz")))
    
    if args.test:
        json_files = json_files[:3]
    elif args.limit:
        json_files = json_files[:args.limit]
    
    print(f"\n找到 {len(json_files)} 个文件需要处理")
    print(f"后端: {backend}")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print("-" * 60)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 统计信息
    total_files = len(json_files)
    total_modified = 0
    start_time = datetime.now()
    
    # 处理每个文件
    for i, input_path in enumerate(json_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output, filename)
        
        try:
            modified = process_file(input_path, output_path, model, tokenizer, backend)
            total_modified += modified
            
            # 进度显示
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (total_files - i - 1)
            
            print(f"[{i+1}/{total_files}] {filename}: {modified} 条instruction已重写 "
                  f"(预计剩余: {remaining/60:.1f}分钟)")
            
        except Exception as e:
            print(f"[{i+1}/{total_files}] {filename}: 错误 - {e}")
    
    # 最终统计
    elapsed_total = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"总修改instruction数: {total_modified}")
    print(f"总耗时: {elapsed_total/60:.2f} 分钟")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
