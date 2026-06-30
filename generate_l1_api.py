#!/usr/bin/env python3
"""
L1 Instruction Generator using Qwen API (DashScope)
Optimizes navigation instructions to remove pedestrian-related content.
Processes train, val, and test datasets with anomaly detection.
"""

import os
import sys
import json
import gzip
import time
import random
import logging
import argparse
import re
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('l1_generation.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


API_KEY = "sk-6adc738411b94388a93069728032d39e"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen-plus"
MAX_TOKENS = 256
TEMPERATURE = 0.3

# ----异常检测阈值----
MAX_INSTRUCTION_CHARS = 2000    # 指令最大字符数，超过视为异常
MIN_INSTRUCTION_CHARS = 5       # 指令最小字符数
MAX_REPEAT_RATIO = 0.5          # 单个词出现超过此比例视为重复异常
FORBIDDEN_WORDS = [             # 处理后仍不应出现的关键词
    'person', 'people', 'pedestrian', 'avoid', 'waiting', 'wait for',
    'approaching', 'walking', 'standing', 'man', 'woman', 'human',
    'crowd', 'dodge', 'evade', 'watch out for', 'look out for',
    'make way', 'give way',
]

SYSTEM_PROMPT = """You are a strict instruction rewriter for indoor navigation tasks.
Convert navigation instructions to pedestrian-free L1 instructions.

RULES:
1. REMOVE ALL references to: person, people, pedestrian, man, woman, human, individual, crowd
2. REMOVE ALL action verbs: avoid, wait for, watch out for, dodge, evade, navigating around, waiting
3. REMOVE ALL descriptions of human movement: walking, standing, approaching, entering, exiting
4. KEEP ALL static environment descriptions (rooms, doors, hallways, furniture, objects, landmarks, directions)
5. Preserve the full navigation intent (turn, go, move toward target)
6. EXTRACT and PRESERVE the original stop condition (e.g. "Stop at the [X]", "Enter the [X]", "Reach the [X]", "Proceed to the [X]") - do NOT invent a new stop target
7. Output ONLY the rewritten instruction, nothing else.

FORBIDDEN in output: person, people, pedestrian, avoid, waiting, wait, approaching,
walking, standing, human, man, woman, crowd, dodge, evade, watch out, look out,
make way, give way

Example:
Input: "Turn left into the bedroom, avoid the bed on the right. Proceed straight, no obstacles or pedestrians. Stop immediately in the empty room."
Output: "Turn left into the bedroom. Proceed straight. Stop in the empty room."

Input: "Go straight and wait for the man to pass. Enter the bedroom on the right."
Output: "Go straight and enter the bedroom on the right."

Input: "Turn right at the doorway. Avoid the person near the kitchen counter. Proceed to the dining table."
Output: "Turn right at the doorway. Proceed to the dining table."

Output ONLY the rewritten instruction."""


# ── 异常检测 ────────────────────────────────────────────────────────────────

def compute_instr_fingerprint(instr: str) -> str:
    """标准化指令文本，抹去数字/物体颜色等差异，用于比较重复。"""
    import re
    # 去掉数字、颜色词、物体具体属性，保留动作结构
    normalized = instr.lower()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def detect_forbidden_words(instruction: str) -> Optional[str]:
    """检测指令是否含禁止词，是则返回描述，否则返回 None。"""
    instr_lower = instruction.lower()
    found = [w for w in FORBIDDEN_WORDS if w in instr_lower]
    if found:
        return f"含禁止词: {found}"
    return None


def detect_anomaly(instruction: str, other_instructions: List[str] = None) -> Optional[str]:
    """检测指令异常：仅检查指令过长或为空（不含禁止词检测）。"""
    if not instruction or len(instruction.strip()) < MIN_INSTRUCTION_CHARS:
        return f"指令过短或为空 (len={len(instruction)})"

    if len(instruction) > MAX_INSTRUCTION_CHARS:
        return f"指令过长 (len={len(instruction)})"

    return None


# ── API 调用 ────────────────────────────────────────────────────────────────

def call_qwen_api(instruction: str, model: str = DEFAULT_MODEL) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    user_content = f"""Rewrite this indoor navigation instruction to remove ALL pedestrian-related content.
Keep the full navigation intent. Output ONLY the rewritten instruction.

Original:
{instruction}

Rewritten:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_BASE, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return content.strip('"\'').strip()
            elif resp.status_code == 429:
                wait_time = (attempt + 1) * 5 + random.uniform(1, 3)
                logger.warning(f"Rate limited, wait {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            else:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return ""
    return ""


def post_clean(result: str) -> Tuple[str, List[str]]:
    """对生成结果进行后处理，移除残留禁止词，返回 (清理后文本, 被移除的词列表)。"""
    cleaned = result
    removed = []
    # 逐个词清理
    for word in FORBIDDEN_WORDS:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        if pattern.search(cleaned):
            removed.append(word)
            cleaned = pattern.sub('', cleaned)
    # 清理多余空格、标点
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    cleaned = re.sub(r'\s*\.\s*', '. ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned = cleaned.rstrip(',') + '.'
    if len(cleaned) < 5:
        cleaned = "Proceed forward to the destination."
    return cleaned, removed


def rewrite_instruction_api(instruction: str, model: str = DEFAULT_MODEL) -> Tuple[str, bool, Optional[str]]:
    """
    Rewrite via API with post-cleaning.
    Returns (result, api_success, post_clean_warning).
    """
    result = call_qwen_api(instruction, model=model)
    if not result:
        result = call_qwen_api(instruction, model=model)
    if not result:
        return fallback_cleanup(instruction), False, "API failed, used fallback"

    # Post-clean to remove any residual forbidden words
    result, removed = post_clean(result)

    # If still has forbidden words, try strict re-generation once
    remaining = [w for w in FORBIDDEN_WORDS if w in result.lower()]
    if remaining:
        strict_result = call_qwen_api(instruction, model=model)
        if strict_result:
            result, _ = post_clean(strict_result)
            remaining = [w for w in FORBIDDEN_WORDS if w in result.lower()]

    if remaining:
        warning = f"残留禁止词 after clean: {remaining}"
    else:
        warning = None

    return result, True, warning


def fallback_cleanup(instruction: str) -> str:
    import re
    patterns = [
        r',?\s*avoid\s+(the\s+)?(person|pedestrian|people|man|woman|human)[^.]*',
        r',?\s*wait\s+for\s+(the\s+)?(person|pedestrian|pedestrians)[^.]*',
        r',?\s*(person|pedestrian|people|man|woman)\s+(walking|moving|standing|approaching)[^.]*',
        r',?\s*watch\s+out\s+for[^.]*',
        r',?\s*look\s+out\s+for[^.]*',
        r'\.?\s*No\s+pedestrians\.?',
        r'\.?\s*No\s+people\.?',
    ]
    result = instruction
    for p in patterns:
        result = re.sub(p, '', result, flags=re.IGNORECASE)
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*,\s*', ', ', result)
    result = result.strip()
    if result and not result.endswith('.'):
        result += '.'
    if len(result) < 10:
        result = "Proceed forward to the destination."
    return result


# ── 数据加载/保存 ───────────────────────────────────────────────────────────

def load_episodes(filepath: str) -> List[Dict[str, Any]]:
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        if 'episodes' in data:
            return data['episodes']
        elif isinstance(data, list):
            return data
        return [data]
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return []


def save_episodes(filepath: str, episodes: List[Dict[str, Any]]):
    data = {"episodes": episodes}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── 核心处理 ────────────────────────────────────────────────────────────────

def _rewrite_episode_worker(args: Tuple[int, Dict[str, Any], str, List[str]]):
    """线程 worker：对单条 episode 进行 rewrite。
    仅当原始指令含禁止词时才调用 API，否则保留原 instruction 不变，不添加任何 l1_* 字段。"""
    idx, episode, model, all_originals = args
    raw_instr = episode.get('instruction')
    # 容错：instruction 可能为 int 或其他非字符串类型
    if not isinstance(raw_instr, str):
        if raw_instr is None:
            original = ""
        else:
            original = str(raw_instr).strip()
    else:
        original = raw_instr.strip()
    if not original:
        return idx, episode, original, "", None, None

    anomaly_info = detect_anomaly(original, other_instructions=all_originals)
    forbidden_info = detect_forbidden_words(original)

    # ── 仅当原始指令含禁止词时才重写 ─────────────────────────────────────
    if not forbidden_info:
        # 无禁止词，原样保留，不添加任何 l1_ 字段
        return idx, episode, original, original, None, None

    new_instruction, success, post_warning = rewrite_instruction_api(original, model=model)

    # 检测生成结果是否仍含禁止词
    gen_anomaly = detect_anomaly(new_instruction)
    gen_forbidden = detect_forbidden_words(new_instruction)

    episode['instruction'] = new_instruction
    episode['l1_changed'] = True
    episode['l1_original'] = original
    episode['l1_api_success'] = success
    episode['l1_original_anomaly'] = forbidden_info  # 记录原始含禁止词
    if anomaly_info:
        episode['l1_original_anomaly'] = f"{forbidden_info} | {anomaly_info}"
    if post_warning:
        episode['l1_post_warning'] = post_warning
    if gen_anomaly:
        episode['l1_generated_anomaly'] = gen_anomaly
    if gen_forbidden:
        existing = episode.get('l1_generated_anomaly', '')
        episode['l1_generated_anomaly'] = f"{gen_forbidden}{(' | ' + existing) if existing else ''}"

    return idx, episode, original, new_instruction, forbidden_info, gen_anomaly


def process_file(
    input_path: str,
    output_path: str,
    model: str,
    max_workers: int,
    pbar: Optional[tqdm] = None,
) -> Dict[str, Any]:
    """处理单个场景文件：并发 rewrite 所有 episodes。"""
    start = time.time()
    episodes = load_episodes(input_path)
    if not episodes:
        return {"status": "empty", "episodes_processed": 0}

    results: Dict[int, Tuple[str, str, Optional[str], Optional[str]]] = {}
    anomalies_original: List[Tuple[str, str]] = []
    anomalies_generated: List[Tuple[str, str]] = []

    # 收集所有原始指令，用于跨 episode 重复检测
    def _to_str(v):
        return str(v).strip() if v is not None else ""
    all_originals = [_to_str(ep.get('instruction')) for ep in episodes]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_rewrite_episode_worker, (i, ep, model, all_originals)): i
            for i, ep in enumerate(episodes)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                _, ep, orig, new_instr, orig_anomaly, gen_anomaly = future.result()
                episodes[idx] = ep
                results[idx] = (orig, new_instr, orig_anomaly, gen_anomaly)
                if orig_anomaly:
                    anomalies_original.append((ep.get('episode_id', '?'), orig, orig_anomaly))
                if gen_anomaly:
                    anomalies_generated.append((ep.get('episode_id', '?'), new_instr, gen_anomaly))
            except Exception as e:
                logger.warning(f"Episode {idx} failed: {e}")

    # ── 打印结果 ──────────────────────────────────────────────────────────
    changed = 0
    for idx, (orig, new_instr, orig_anomaly, gen_anomaly) in sorted(results.items()):
        if orig == new_instr:
            continue
        changed += 1
        ep_id = episodes[idx].get('episode_id', '?')
        warnings = []
        if orig_anomaly:
            warnings.append(f"[原异常: {orig_anomaly}]")
        if gen_anomaly:
            warnings.append(f"[生成异常: {gen_anomaly}]")
        warn_str = "  ".join(warnings)
        print(f"\n{'─'*72}")
        print(f"  Scene: {os.path.basename(input_path)}  Episode: {ep_id}  {warn_str}")
        print(f"  ── BEFORE ──")
        orig_display = orig if len(orig) <= 300 else orig[:300] + " ... [TRUNCATED]"
        print(f"  {orig_display}")
        print(f"  ── AFTER ──")
        new_display = new_instr if len(new_instr) <= 300 else new_instr[:300] + " ..."
        print(f"  {new_display}")
        print(f"{'─'*72}")

    # ── 异常汇总 ─────────────────────────────────────────────────────────
    if anomalies_original:
        print(f"\n  [原指令异常] 共 {len(anomalies_original)} 条:")
        for ep_id, instr, reason in anomalies_original[:5]:
            print(f"    ep={ep_id}: {reason}")
            print(f"    {instr[:120]}{'...' if len(instr) > 120 else ''}")
        if len(anomalies_original) > 5:
            print(f"    ... 还有 {len(anomalies_original)-5} 条")
    if anomalies_generated:
        print(f"\n  [生成结果异常] 共 {len(anomalies_generated)} 条:")
        for ep_id, instr, reason in anomalies_generated[:5]:
            print(f"    ep={ep_id}: {reason}")
            print(f"    {instr[:120]}{'...' if len(instr) > 120 else ''}")
        if len(anomalies_generated) > 5:
            print(f"    ... 还有 {len(anomalies_generated)-5} 条")

    save_episodes(output_path, episodes)
    elapsed = time.time() - start
    return {
        "status": "success",
        "input": input_path,
        "output": output_path,
        "episodes": len(episodes),
        "changed": changed,
        "elapsed": round(elapsed, 2),
    }


# ── 主入口 ─────────────────────────────────────────────────────────────────

DATASET_MAP = {
    "train":       ("train",                              "train"),
    "val_seen":    ("val/seen",                           "val_seen"),
    "val_unseen":  ("val/unseen",                         "val_unseen"),
    "test_seen":   ("test/seen_test",                     "test_seen"),
    "test_unseen": ("test/unseen_test",                   "test_unseen"),
}


def main():
    parser = argparse.ArgumentParser(description="L1 instruction generation via Qwen API")
    parser.add_argument("--base-input",  type=str, default="/share/home/u19666033/dhj/DPed_pro/dped_pro")
    parser.add_argument("--base-output", type=str, default="/share/home/u19666033/dhj/DPed_pro/dped_pro_l1")
    parser.add_argument("--model",        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-workers",  type=int, default=16,
                        help="并发 API 请求数 (每个场景内并行)")
    parser.add_argument("--max-files",   type=int, default=0,
                        help="每个数据集最大文件数 (0=全部)")
    parser.add_argument("--resume",      action="store_true",
                        help="跳过已存在的输出文件")
    parser.add_argument("--datasets",    type=str, nargs='+',
                        choices=list(DATASET_MAP.keys()),
                        default=list(DATASET_MAP.keys()))
    args = parser.parse_args()

    # ── 统计总文件数 ─────────────────────────────────────────────────────
    dataset_files: Dict[str, List[str]] = {}
    for name in args.datasets:
        input_sub = os.path.join(args.base_input, DATASET_MAP[name][0])
        if not os.path.isdir(input_sub):
            logger.warning(f"跳过不存在的目录: {input_sub}")
            continue
        files = sorted([
            f for f in os.listdir(input_sub)
            if f.endswith('.json') or f.endswith('.json.gz')
        ])
        if args.max_files > 0:
            files = files[:args.max_files]
        dataset_files[name] = files

    total_files = sum(len(v) for v in dataset_files.values())
    total_episodes = 0

    logger.info(f"待处理数据集: {list(dataset_files.keys())}")
    logger.info(f"总场景文件数: {total_files}")

    # 预统计 episodes 数量（用于进度条）
    logger.info("正在统计 episodes 数量...")
    for name, files in dataset_files.items():
        input_sub = os.path.join(args.base_input, DATASET_MAP[name][0])
        for f in files:
            fp = os.path.join(input_sub, f)
            total_episodes += len(load_episodes(fp))
    logger.info(f"总指令数: {total_episodes}")

    # ── 全局进度条 ────────────────────────────────────────────────────────
    all_files = []
    for name, files in dataset_files.items():
        input_sub = os.path.join(args.base_input, DATASET_MAP[name][0])
        output_sub = os.path.join(args.base_output, DATASET_MAP[name][1])
        for f in files:
            all_files.append((name, input_sub, output_sub, f))

    pbar_total = tqdm(total=total_files, desc="总进度", unit="scene",
                      bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_noinv}]")
    pbar_episodes = tqdm(total=total_episodes, desc="指令数", unit="instr",
                         bar_format="{l_bar}{bar}| {n}/{total}")

    all_stats = []
    total_changed = 0

    for name, input_sub, output_sub, filename in all_files:
        input_path = os.path.join(input_sub, filename)
        output_path = os.path.join(output_sub, filename.replace('.gz', ''))

        if args.resume and os.path.exists(output_path):
            eps = load_episodes(output_path)
            skipped_changed = sum(1 for e in eps if e.get('l1_changed', False))
            pbar_total.update(1)
            pbar_episodes.update(len(eps))
            total_changed += skipped_changed
            continue

        try:
            stats = process_file(input_path, output_path, args.model, args.max_workers)
            stats["dataset"] = name
            all_stats.append(stats)
            pbar_total.update(1)
            pbar_episodes.update(stats["episodes"])
            total_changed += stats.get("changed", 0)
            pbar_total.set_postfix({
                "changed": total_changed,
                "last_scene": f"{stats.get('changed', 0)}/{stats.get('episodes', 0)}"
            })
        except Exception as e:
            logger.error(f"处理失败 {filename}: {e}")
            pbar_total.update(1)

    pbar_total.close()
    pbar_episodes.close()

    # ── 汇总 ──────────────────────────────────────────────────────────────
    summary_path = os.path.join(args.base_output, "processing_summary.json")
    os.makedirs(args.base_output, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "args": vars(args),
            "stats": all_stats,
            "summary": {
                "files_processed": len(all_stats),
                "total_episodes": total_episodes,
                "total_changed": total_changed,
            }
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info("处理完成!")
    logger.info(f"场景文件: {len(all_stats)}")
    logger.info(f"总指令数: {total_episodes}")
    logger.info(f"已修改:   {total_changed}")
    logger.info(f"汇总:     {summary_path}")
    logger.info(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
