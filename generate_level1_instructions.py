#!/usr/bin/env python3
"""
从第二层指令（包含行人运动信息）生成第一层指令（去除行人运动信息）
使用本地部署的Qwen-3 API
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ================== 配置 ==================
# Qwen-3本地API地址（根据你的实际部署修改）
QWEN_API_URL = "http://localhost:8000/v1/chat/completions"  # 修改为你的API地址
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 修改为你的模型名称

# 数据路径
DATA_ROOT = "data/collect_data/train"
OUTPUT_DIR_NAME = "instruction_vl_level_1"  # 新生成的第一层指令目录名
SOURCE_DIR_NAME = "instruction_vl_level_2"  # 第二层指令源目录名

# 并发数量
MAX_WORKERS = 4

# ================== Prompt ==================
SYSTEM_PROMPT = """You are an expert in visual-language navigation instruction processing.
Your task is to remove pedestrian-related movement information from navigation instructions while preserving all other spatial and landmark details.

Guidelines:
1. Remove any phrases about avoiding, going around, steering clear of pedestrians
2. Remove phrases like "taking a left/right to go around the person"
3. Keep all landmark descriptions, spatial relationships, and stopping points
4. Keep the instruction natural and fluent
5. Maintain the same tone and style as the original

Return ONLY the processed instruction text without any explanation."""

def create_prompt(instruction_level_2):
    """创建用于Qwen-3的prompt"""
    user_message = f"""Original instruction with pedestrian information:
"{instruction_level_2}"

Please generate the instruction without pedestrian information:"""
    
    return user_message


def call_qwen_api(instruction_level_2, max_retries=3):
    """调用本地Qwen-3 API"""
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": QWEN_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": create_prompt(instruction_level_2)}
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                QWEN_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                instruction_level_1 = result['choices'][0]['message']['content'].strip()
                # 去除可能的引号
                instruction_level_1 = instruction_level_1.strip('"').strip("'")
                return instruction_level_1
            else:
                print(f"API request failed (attempt {attempt+1}): {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            print(f"Error calling API (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return None


def process_single_episode(episode_path):
    """处理单个episode的指令"""
    source_dir = episode_path / SOURCE_DIR_NAME
    output_dir = episode_path / OUTPUT_DIR_NAME
    
    # 检查源目录是否存在
    if not source_dir.exists():
        return None, "source_not_found"
    
    # 检查输出目录是否已存在
    if output_dir.exists():
        output_file = output_dir / "0.txt"
        if output_file.exists():
            return str(episode_path), "already_exists"
    
    # 读取第二层指令
    source_file = source_dir / "0.txt"
    if not source_file.exists():
        return str(episode_path), "source_file_missing"
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            instruction_level_2 = f.read().strip()
        
        if not instruction_level_2:
            return str(episode_path), "empty_instruction"
        
        # 调用Qwen-3生成第一层指令
        instruction_level_1 = call_qwen_api(instruction_level_2)
        
        if instruction_level_1 is None:
            return str(episode_path), "api_failed"
        
        # 保存第一层指令
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "0.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(instruction_level_1)
        
        return str(episode_path), "success"
        
    except Exception as e:
        return str(episode_path), f"error: {str(e)}"


def find_all_episodes(data_root):
    """找到所有包含第二层指令的episode路径"""
    data_root = Path(data_root)
    episodes = []
    
    # 遍历所有场景
    for scene_dir in data_root.iterdir():
        if not scene_dir.is_dir():
            continue
        
        # 遍历该场景下的所有episode
        for episode_dir in scene_dir.iterdir():
            if not episode_dir.is_dir():
                continue
            
            # 检查是否有第二层指令
            if (episode_dir / SOURCE_DIR_NAME).exists():
                episodes.append(episode_dir)
    
    return episodes


def main():
    print("=" * 80)
    print("从第二层指令生成第一层指令（去除行人运动信息）")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  数据根目录: {DATA_ROOT}")
    print(f"  源指令目录: {SOURCE_DIR_NAME}")
    print(f"  输出目录: {OUTPUT_DIR_NAME}")
    print(f"  API地址: {QWEN_API_URL}")
    print(f"  模型: {QWEN_MODEL_NAME}")
    print(f"  并发数: {MAX_WORKERS}")
    
    # 测试API连接
    print(f"\n测试API连接...")
    test_instruction = "Walk forward and turn left to avoid a person walking nearby."
    test_result = call_qwen_api(test_instruction)
    if test_result is None:
        print("✗ API连接失败！请检查:")
        print(f"  1. Qwen-3 API服务是否启动")
        print(f"  2. API地址是否正确: {QWEN_API_URL}")
        print(f"  3. 模型名称是否正确: {QWEN_MODEL_NAME}")
        return
    else:
        print("✓ API连接成功！")
        print(f"  测试输入: {test_instruction}")
        print(f"  测试输出: {test_result}")
    
    # 查找所有episode
    print(f"\n查找所有episodes...")
    episodes = find_all_episodes(DATA_ROOT)
    print(f"✓ 找到 {len(episodes)} 个episodes")
    
    if len(episodes) == 0:
        print("没有找到任何包含第二层指令的episodes！")
        return
    
    # 处理所有episodes
    print(f"\n开始处理...")
    results = {
        "success": [],
        "already_exists": [],
        "api_failed": [],
        "errors": []
    }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_episode, ep): ep for ep in episodes}
        
        with tqdm(total=len(episodes), desc="Processing") as pbar:
            for future in as_completed(futures):
                episode_path, status = future.result()
                
                if status == "success":
                    results["success"].append(episode_path)
                elif status == "already_exists":
                    results["already_exists"].append(episode_path)
                elif status == "api_failed":
                    results["api_failed"].append(episode_path)
                else:
                    results["errors"].append((episode_path, status))
                
                pbar.update(1)
    
    # 输出统计
    print("\n" + "=" * 80)
    print("处理完成！统计:")
    print("=" * 80)
    print(f"  ✓ 成功生成: {len(results['success'])}")
    print(f"  ⊙ 已存在: {len(results['already_exists'])}")
    print(f"  ✗ API失败: {len(results['api_failed'])}")
    print(f"  ✗ 其他错误: {len(results['errors'])}")
    print(f"  总计: {len(episodes)}")
    
    # 保存结果日志
    log_file = Path(DATA_ROOT).parent.parent / "generate_level1_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细日志已保存到: {log_file}")
    
    # 显示失败的案例（如果有）
    if results["api_failed"]:
        print(f"\nAPI失败的episodes (前10个):")
        for ep in results["api_failed"][:10]:
            print(f"  - {ep}")
    
    if results["errors"]:
        print(f"\n错误的episodes (前10个):")
        for ep, err in results["errors"][:10]:
            print(f"  - {ep}: {err}")


if __name__ == "__main__":
    main()

