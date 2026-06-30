#!/usr/bin/env python3
"""
从第二层指令（包含行人运动信息）生成第一层指令（去除行人运动信息）
使用本地部署的Qwen模型
"""

import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================== 配置 ==================
# Qwen模型配置（参考NavComposer的Qwen30Summary配置）
# 对应: summary_config.name = Qwen30Summary
#       summary_config.qwen30_config.model_name
#       summary_config.qwen30_config.enable_thinking
#       run_precision = "bfloat16"

# 模型选择
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # Qwen3.0 32B模型（最佳质量）
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Qwen2.5 7B（更快）

# 运行配置（参考NavComposer）
RUN_PRECISION = "bfloat16"  # 对应 run_precision
ENABLE_THINKING = False  # 对应 qwen30_config.enable_thinking

# Temperature（如果NavComposer中有配置，可以调整）
TEMPERATURE = 0.8  # 参考NavComposer typical value

# System Prompt（用于去除行人信息的专用prompt）
SYSTEM_PROMPT_CUSTOM = """As an expert in vision-and-language navigation, your task is to remove pedestrian-related information from navigation instructions while keeping all other details intact.

Remove any mentions of:
- Avoiding, going around, or steering clear of pedestrians/people/persons
- Waiting for pedestrians
- Navigation adjustments due to people
- Any safety instructions related to human obstacles

Keep all other information:
- Landmarks and objects
- Spatial relationships
- Directional instructions
- Stopping points

Return ONLY the cleaned instruction without any explanation."""

# 数据路径
DATA_ROOT = "data/collect_data/val"
OUTPUT_DIR_NAME = "instruction_vl_level_1"  # 新生成的第一层指令目录名
SOURCE_DIR_NAME = "instruction_vl_level_2"  # 第二层指令源目录名

# 生成参数
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 1024  # 增大以支持更长的指令

# 批处理配置
BATCH_SIZE = 1  # 每次处理的指令数（可以增大以提高效率）
MAX_EPISODES = None  # None表示处理所有，或设置数字限制处理数量

# ================== Qwen模型封装 ==================
class Qwen3LocalModel:
    """本地Qwen-3模型封装"""
    
    def __init__(self, model_name, run_precision="bfloat16", enable_thinking=False):
        print(f"Loading Qwen model: {model_name}")
        print(f"Precision: {run_precision}")
        
        if run_precision == "bfloat16":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.enable_thinking = enable_thinking
        
        print(f"✓ Model loaded successfully on device: {self.model.device}")
    
    def generate(self, system_prompt, user_message, temperature=0.7, max_new_tokens=1024):
        """生成响应（参考qwen30_summary.py的实现）"""
        # 将system prompt和user message合并
        # 参考qwen30_summary.py的方式
        full_prompt = system_prompt + "\n\n" + user_message
        
        messages = [
            {
                "role": "user",
                "content": full_prompt,
            },
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 处理thinking content（如果启用）
        if self.enable_thinking:
            try:
                # 查找 </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking_content = self.tokenizer.decode(
                output_ids[:index], skip_special_tokens=True
            ).strip("\n")
            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
        else:
            thinking_content = ""
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        # 清理输出（参考qwen30_summary.py）
        import re
        content = content.replace('"', '')
        content = re.sub("^.*?:", "", content)  # 移除开头的标签（如 "Instruction:"）
        content = content.strip()
        
        return content, thinking_content


# ================== Prompt ==================
# 使用上面定义的SYSTEM_PROMPT_CUSTOM，或者使用默认的
SYSTEM_PROMPT = SYSTEM_PROMPT_CUSTOM if 'SYSTEM_PROMPT_CUSTOM' in globals() else """You are an expert in visual-language navigation instruction processing.
Your task is to remove pedestrian-related movement information from navigation instructions while preserving all other spatial and landmark details.

Guidelines:
1. Remove any phrases about avoiding, going around, steering clear of pedestrians or people
2. Remove phrases like "taking a left/right to go around the person/pedestrians"
3. Keep all landmark descriptions, spatial relationships, and stopping points
4. Keep the instruction natural and fluent
5. Maintain the same tone and style as the original
6. If there's no pedestrian information, return the instruction as is

Return ONLY the processed instruction text without any explanation or quotation marks."""

def create_prompt(instruction_level_2):
    """创建用于Qwen的prompt"""
    return f"""Original navigation instruction:
{instruction_level_2}

Please generate the instruction without pedestrian-related information:"""


# ================== 处理函数 ==================
def process_single_episode(episode_path, model):
    """处理单个episode的指令"""
    source_dir = episode_path / SOURCE_DIR_NAME
    output_dir = episode_path / OUTPUT_DIR_NAME
    
    # 检查源目录是否存在
    if not source_dir.exists():
        return str(episode_path), "source_not_found"
    
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
        
        # 使用本地模型生成第一层指令
        instruction_level_1, thinking = model.generate(
            SYSTEM_PROMPT,
            create_prompt(instruction_level_2),
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS
        )
        
        if not instruction_level_1:
            return str(episode_path), "empty_output"
        
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
    
    print("Scanning episodes...")
    # 遍历所有场景
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        # 遍历该场景下的所有episode
        for episode_dir in sorted(scene_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            
            # 检查是否有第二层指令
            if (episode_dir / SOURCE_DIR_NAME).exists():
                episodes.append(episode_dir)
    
    return episodes


def main():
    print("=" * 80)
    print("从第二层指令生成第一层指令（去除行人运动信息）")
    print("使用本地Qwen模型")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  数据根目录: {DATA_ROOT}")
    print(f"  源指令目录: {SOURCE_DIR_NAME}")
    print(f"  输出目录: {OUTPUT_DIR_NAME}")
    print(f"  模型: {MODEL_NAME}")
    print(f"  精度: {RUN_PRECISION}")
    print(f"  Temperature: {TEMPERATURE}")
    
    # 加载模型
    print(f"\n{'='*80}")
    print("加载Qwen模型...")
    print(f"{'='*80}")
    model = Qwen3LocalModel(
        MODEL_NAME,
        run_precision=RUN_PRECISION,
        enable_thinking=ENABLE_THINKING
    )
    
    # 测试模型
    print(f"\n{'='*80}")
    print("测试模型生成...")
    print(f"{'='*80}")
    test_instruction = "Walk forward and turn left to avoid a person walking nearby. Stop at the door."
    test_result, _ = model.generate(
        SYSTEM_PROMPT,
        create_prompt(test_instruction),
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS
    )
    print(f"测试输入: {test_instruction}")
    print(f"测试输出: {test_result}")
    
    # 查找所有episode
    print(f"\n{'='*80}")
    print("查找所有episodes...")
    print(f"{'='*80}")
    episodes = find_all_episodes(DATA_ROOT)
    
    if MAX_EPISODES is not None:
        episodes = episodes[:MAX_EPISODES]
        print(f"✓ 限制处理前 {MAX_EPISODES} 个episodes")
    
    print(f"✓ 找到 {len(episodes)} 个episodes需要处理")
    
    if len(episodes) == 0:
        print("没有找到任何包含第二层指令的episodes！")
        return
    
    # 处理所有episodes（单线程，因为模型已经在GPU上）
    print(f"\n{'='*80}")
    print("开始处理...")
    print(f"{'='*80}")
    
    results = {
        "success": [],
        "already_exists": [],
        "errors": []
    }
    
    with tqdm(total=len(episodes), desc="Processing episodes") as pbar:
        for episode_path in episodes:
            episode_str, status = process_single_episode(episode_path, model)
            
            if status == "success":
                results["success"].append(episode_str)
            elif status == "already_exists":
                results["already_exists"].append(episode_str)
            else:
                results["errors"].append((episode_str, status))
            
            pbar.update(1)
            
            # 每100个episode显示一次进度
            if (len(results["success"]) + len(results["already_exists"]) + len(results["errors"])) % 100 == 0:
                pbar.set_postfix({
                    'success': len(results["success"]),
                    'exists': len(results["already_exists"]),
                    'errors': len(results["errors"])
                })
    
    # 输出统计
    print("\n" + "=" * 80)
    print("处理完成！统计:")
    print("=" * 80)
    print(f"  ✓ 成功生成: {len(results['success'])}")
    print(f"  ⊙ 已存在: {len(results['already_exists'])}")
    print(f"  ✗ 错误: {len(results['errors'])}")
    print(f"  总计: {len(episodes)}")
    
    # 保存结果日志
    log_file = Path(DATA_ROOT).parent.parent / "generate_level1_local_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细日志已保存到: {log_file}")
    
    # 显示错误的案例（如果有）
    if results["errors"]:
        print(f"\n错误的episodes (前10个):")
        for ep, err in results["errors"][:10]:
            print(f"  - {ep}")
            print(f"    {err}")
    
    # 随机展示几个成功案例
    if results["success"]:
        import random
        print(f"\n{'='*80}")
        print("随机展示3个生成结果:")
        print(f"{'='*80}")
        samples = random.sample(results["success"], min(3, len(results["success"])))
        
        for i, ep_str in enumerate(samples, 1):
            ep_path = Path(ep_str)
            level_2_file = ep_path / SOURCE_DIR_NAME / "0.txt"
            level_1_file = ep_path / OUTPUT_DIR_NAME / "0.txt"
            
            if level_2_file.exists() and level_1_file.exists():
                level_2 = level_2_file.read_text(encoding='utf-8').strip()
                level_1 = level_1_file.read_text(encoding='utf-8').strip()
                
                print(f"\n示例 {i}: {ep_path.parent.name}/{ep_path.name}")
                print(f"  Level 2 (with pedestrians):")
                print(f"    {level_2}")
                print(f"  Level 1 (without pedestrians):")
                print(f"    {level_1}")


if __name__ == "__main__":
    main()

