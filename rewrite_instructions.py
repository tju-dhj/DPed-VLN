#!/usr/bin/env python3
"""
使用Qwen模型批量重写DPed数据集的instruction
将包含行人相关内容的instruction转换为L1语言指令（不包含行人/躲避指导）
"""

import os
import gzip
import json
import glob
import time
from tqdm import tqdm
import threading
from queue import Queue

# 模型路径
MODEL_PATH = "/share/home/u19666033/dhj/models/Qwen3.6-27B"

# 输入输出目录
INPUT_DIR = "/share/home/u19666033/dhj/DPed_pro/dped_pro/train"
OUTPUT_DIR = "/share/home/u19666033/dhj/DPed_pro/dped_pro/train_rewritten"

# 批量处理大小
BATCH_SIZE = 50

def load_model():
    """加载Qwen模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("正在加载Qwen模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"
    )
    print("模型加载完成!")
    return model, tokenizer

def rewrite_instruction(instruction, model, tokenizer, client=None):
    """使用Qwen模型重写instruction，去除行人相关内容"""
    
    prompt = f"""You are a helpful instruction rewriter. Your task is to rewrite navigation instructions by REMOVING all references to pedestrians, people, or avoidance instructions.

Original instruction:
\"{instruction}\"

Requirements:
1. Remove ALL mentions of pedestrians, people, persons, man, woman, avoid, wait for pedestrians, etc.
2. Keep ONLY the navigation directions (turn left, turn right, go straight, move forward, etc.)
3. Make it a coherent, natural language instruction
4. Keep the same general structure and number of steps if possible
5. If the instruction only contains navigation without pedestrians, keep it mostly unchanged

Rewrite the instruction:"""
    
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

def process_file(input_path, output_path, model, tokenizer):
    """处理单个json.gz文件"""
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    episodes = data.get('episodes', [])
    
    for episode in episodes:
        original_instruction = episode.get('instruction', '')
        if original_instruction:
            # 检查是否包含行人相关关键词
            pedestrian_keywords = ['pedestrian', 'person', 'people', 'man', 'woman', 
                                   'avoid', 'wait for', 'walk around']
            
            has_pedestrian = any(kw in original_instruction.lower() for kw in pedestrian_keywords)
            
            if has_pedestrian:
                try:
                    new_instruction = rewrite_instruction(original_instruction, model, tokenizer)
                    episode['instruction'] = new_instruction
                    episode['original_instruction'] = original_instruction  # 保存原指令
                except Exception as e:
                    print(f"Error rewriting instruction: {e}")
                    episode['instruction_rewrite_error'] = str(e)
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=INPUT_DIR)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 获取所有文件
    json_files = sorted(glob.glob(os.path.join(args.input, "*.json.gz")))
    
    if args.limit:
        json_files = json_files[:args.limit]
    
    print(f"找到 {len(json_files)} 个文件需要处理")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理每个文件
    for i, input_path in enumerate(tqdm(json_files, desc="Processing files")):
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output, filename)
        
        try:
            process_file(input_path, output_path, model, tokenizer)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        
        # 每处理10个文件保存一次进度
        if (i + 1) % 10 == 0:
            print(f"\n进度: {i+1}/{len(json_files)}")

if __name__ == "__main__":
    main()
