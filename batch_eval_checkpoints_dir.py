#!/usr/bin/env python3
"""
批量评估脚本 - 批量评估文件夹中的所有checkpoint并保存详细结果
用法: 
    python batch_eval_checkpoints_dir.py                                    # 评估所有checkpoint
    python batch_eval_checkpoints_dir.py --start 5                          # 从第5个checkpoint开始评估
    python batch_eval_checkpoints_dir.py --start 5 --end 50                 # 评估第5到50个checkpoint
    python batch_eval_checkpoints_dir.py --ckpt-dir /path/to/checkpoints     # 指定checkpoint目录
    python batch_eval_checkpoints_dir.py --config-name DPed_pro/eval/DPed_rl_val_6action_normalized_50  # 指定config

功能:
    1. 批量eval指定目录下的所有.pth文件
    2. 支持从指定checkpoint开始
    3. 支持指定结束checkpoint
    4. 每个checkpoint的结果保存为独立的JSON文件
    5. 最终汇总所有结果为CSV和JSON
    6. 自动分析最佳checkpoint
"""

import os
import sys
import json
import glob
import shutil
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
# CHECKPOINT_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_social_eq/hm3d/checkpoints"
CHECKPOINT_DIR="/share/home/u19666033/dhj/dped-vln/evaluation-vln/dped_pro_clip_rl_v2_6actions/hm3d/checkpoints"
# ============== 配置区域 ==============
# DEFAULT_CKPT_DIR = "/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_social_eq/hm3d/checkpoints"
DEFAULT_CKPT_DIR = "/share/home/u19666033/dhj/dped-vln/evaluation-vln/dped_pro_clip_rl_v2_6actions/hm3d/checkpoints"
DEFAULT_OUTPUT_BASE = "/share/home/u19666033/dhj/dped-vln/evaluation-vlnce-dpedpro2/rl/4a-base-start/hm3d/eval_fast_2"
DEFAULT_CONFIG = "DPed_VLN/eval/eval_rl_v1_eval_fast.yaml"
# ======================================


def natural_sort_key(s):
    """自然排序key，支持 ckpt.5.pth 和 ckpt.43.pth 混合排序"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]


def parse_args():
    parser = argparse.ArgumentParser(description="批量评估checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR,
                        help=f"checkpoint目录 (默认: {DEFAULT_CKPT_DIR})")
    parser.add_argument("--output-base", type=str, default=DEFAULT_OUTPUT_BASE,
                        help=f"输出根目录 (默认: {DEFAULT_OUTPUT_BASE})")
    parser.add_argument("--config-name", type=str, default=DEFAULT_CONFIG,
                        help=f"eval配置文件名 (默认: {DEFAULT_CONFIG})")
    parser.add_argument("--start", type=int, default=None,
                        help="从第几个checkpoint开始评估 (默认: 所有)")
    parser.add_argument("--end", type=int, default=None,
                        help="评估到第几个checkpoint结束 (默认: 所有)")
    parser.add_argument("--dataset", type=str, default="val_evalfast",
                        help="评估数据集 (默认: val_evalfast)")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="单个checkpoint超时时间，秒 (默认: 7200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出要评估的checkpoint，不实际运行")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有结果 (默认跳过已评估的checkpoint)")
    return parser.parse_args()


def get_checkpoint_list(ckpt_dir, start=None, end=None):
    """获取指定范围内的checkpoint文件列表"""
    all_files = glob.glob(os.path.join(ckpt_dir, "ckpt.*.pth"))
    
    ckpt_infos = []
    for f in all_files:
        basename = os.path.basename(f)
        # 支持两种格式:
        # 1. ckpt.N.pth (如 ckpt.5.pth)
        # 2. ckpt.epoch_N.step_M.pth (如 ckpt.epoch_1.step_28365.pth)
        match = re.search(r'ckpt\.epoch_(\d+)\.step_\d+\.pth', basename)
        if match:
            # 格式2: ckpt.epoch_N.step_M.pth
            ckpt_num = int(match.group(1))
            ckpt_infos.append({
                'num': ckpt_num,
                'path': f,
                'basename': basename
            })
        else:
            match = re.search(r'ckpt\.(\d+)\.pth', basename)
            if match:
                # 格式1: ckpt.N.pth
                ckpt_num = int(match.group(1))
                ckpt_infos.append({
                    'num': ckpt_num,
                    'path': f,
                    'basename': basename
                })
    
    # 自然排序
    ckpt_infos.sort(key=lambda x: x['num'])
    
    # 过滤范围
    if start is not None:
        ckpt_infos = [c for c in ckpt_infos if c['num'] >= start]
    if end is not None:
        ckpt_infos = [c for c in ckpt_infos if c['num'] <= end]
    
    return ckpt_infos


def build_eval_command(ckpt_path, ckpt_num, config_name, output_base, dataset):
    """构建评估命令"""
    # 每个checkpoint的输出目录
    output_dir = os.path.join(output_base, f"ckpt.{ckpt_num}", "checkpoints")
    tensorboard_dir = os.path.join(output_base, f"ckpt.{ckpt_num}", "tb")
    
    cmd = [
        "python", "-u", "-m", "habitat-baselines.habitat_baselines.run",
        "--config-name", config_name,
        f"habitat_baselines.eval_ckpt_path_dir={ckpt_path}",
        f"habitat_baselines.checkpoint_folder={output_dir}",
        f"habitat_baselines.tensorboard_dir={tensorboard_dir}",
    ]
    
    return cmd, output_dir


def run_eval(ckpt_info, config_name, output_base, dataset, timeout):
    """运行单个checkpoint的评估"""
    ckpt_num = ckpt_info['num']
    ckpt_path = ckpt_info['path']
    
    # 检查是否已有结果
    result_json = os.path.join(output_base, f"ckpt.{ckpt_num}", "result.json")
    if os.path.exists(result_json) and not args.overwrite:
        print(f"  [跳过] ckpt.{ckpt_num} 已存在评估结果")
        with open(result_json, 'r') as f:
            return json.load(f)
    
    print(f"\n{'='*70}")
    print(f"  评估 ckpt.{ckpt_num}")
    print(f"  路径: {ckpt_path}")
    print(f"{'='*70}")
    
    cmd, output_dir = build_eval_command(ckpt_path, ckpt_num, config_name, output_base, dataset)
    print(f"执行命令: {' '.join(cmd)}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = datetime.now()
    full_output_lines = []
    
    try:
        # 使用Popen实现流式输出，实时打印到终端
        process = subprocess.Popen(
            cmd,
            cwd="/share/home/u19666033/dhj/dped-vln",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并stderr到stdout
            text=True,
            bufsize=1  # 行缓冲，实时输出
        )
        
        # 实时读取并打印每一行，同时保存
        for line in iter(process.stdout.readline, ''):
            if line:
                full_output_lines.append(line)
                print(line, end='')  # 实时打印到终端
        
        process.wait()
        returncode = process.returncode
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # 合并所有输出
        full_output = ''.join(full_output_lines)
        
        # 提取指标
        metrics = extract_all_metrics(full_output)
        metrics['ckpt_num'] = ckpt_num
        metrics['ckpt_path'] = ckpt_path
        metrics['elapsed_seconds'] = elapsed
        metrics['returncode'] = returncode
        metrics['success'] = returncode == 0
        metrics['stdout_snippet'] = full_output[-3000:] if len(full_output) > 3000 else full_output
        metrics['full_output'] = full_output[-50000:]  # 保留最近50000字符
        
    except Exception as e:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        print(f"  [错误] ckpt.{ckpt_num}: {e}")
        metrics = {
            'ckpt_num': ckpt_num,
            'ckpt_path': ckpt_path,
            'elapsed_seconds': elapsed,
            'returncode': -1,
            'success': False,
            'error': str(e),
        }
    
    # 保存单个checkpoint结果
    ckpt_result_dir = os.path.join(output_base, f"ckpt.{ckpt_num}")
    os.makedirs(ckpt_result_dir, exist_ok=True)
    
    with open(os.path.join(ckpt_result_dir, "result.json"), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 打印关键指标
    sr = metrics.get('success_rate') or metrics.get('sr') or metrics.get('Success Rate')
    spl = metrics.get('spl') or metrics.get('SPL')
    if sr is not None:
        print(f"\n  >>> ckpt.{ckpt_num} SR: {sr:.4f}", end="")
        if spl is not None:
            print(f"  SPL: {spl:.4f}", end="")
        print(f"  耗时: {elapsed:.0f}s")
    elif not metrics.get('success', True):
        print(f"\n  >>> ckpt.{ckpt_num} 评估失败，耗时: {elapsed:.0f}s")
    else:
        print(f"\n  >>> ckpt.{ckpt_num} 完成，耗时: {elapsed:.0f}s (未能解析SR)")
    
    return metrics


def extract_all_metrics(output):
    """从eval输出中提取所有关键指标"""
    metrics = {}
    
    # 常见的指标模式
    patterns = {
        # 成功率
        'success_rate': [r'Success Rate[:\s]+([0-9]*\.[0-9]+)', r'SR[:\s]+([0-9]*\.[0-9]+)',
                         r'success_rate[:\s]+([0-9]*\.[0-9]+)', r'success[:\s]+([0-9]*\.[0-9]+)',
                         r'Success[:\s]+([0-9]*\.[0-9]+)', r'"success_rate"\s*:\s*([0-9]*\.[0-9]+)'],
        # SPL
        'spl': [r'SPL[:\s]+([0-9]*\.[0-9]+)', r'spl[:\s]+([0-9]*\.[0-9]+)',
                r'"spl"\s*:\s*([0-9]*\.[0-9]+)', r'SPL \( Success weighted by Path Length \)[:\s]+([0-9]*\.[0-9]+)'],
        # 导航误差
        'navigation_error': [r'Navigation Error[:\s]+([0-9]*\.[0-9]+)', r'navigation_error[:\s]+([0-9]*\.[0-9]+)'],
        # 动作准确率
        'action_acc': [r'Action Accuracy[:\s]+([0-9]*\.[0-9]+)', r'action_acc[:\s]+([0-9]*\.[0-9]+)',
                       r'action_accuracy[:\s]+([0-9]*\.[0-9]+)'],
        # 总reward
        'total_reward': [r'Total Reward[:\s]+([0-9]*\.[0-9]+)', r'total_reward[:\s]+([0-9]*\.[0-9]+)'],
        # episodes数
        'num_episodes': [r'num_episodes[:\s]+([0-9]+)', r'Number of episodes[:\s]+([0-9]+)'],
        # 距离
        'distance_to_goal': [r'Distance to Goal[:\s]+([0-9]*\.[0-9]+)', r'distance_to_goal[:\s]+([0-9]*\.[0-9]+)'],
        # 步骤数
        'steps': [r'Steps[:\s]+([0-9]+)', r'steps[:\s]+([0-9]+)'],
    }
    
    for metric_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                metrics[metric_name] = float(match.group(1))
                # 也保存别名
                if metric_name == 'success_rate':
                    metrics['sr'] = metrics['success_rate']
                break
    
    # 尝试从JSON块中提取
    json_match = re.search(r'\{[^{}]*"success_rate"[^{}]*\}', output)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            for k, v in parsed.items():
                if isinstance(v, (int, float)) and k not in metrics:
                    metrics[k] = v
        except:
            pass
    
    return metrics


def save_summary(all_results, output_base, start_time):
    """保存汇总结果"""
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    
    # 汇总数据
    summary = {
        'eval_start_time': start_time.isoformat(),
        'eval_end_time': end_time.isoformat(),
        'total_elapsed_seconds': total_elapsed,
        'total_checkpoints': len(all_results),
        'successful_evals': sum(1 for r in all_results if r.get('success')),
        'failed_evals': sum(1 for r in all_results if not r.get('success')),
    }
    
    # 提取有SR的结果
    sr_results = [r for r in all_results if r.get('success_rate') is not None or r.get('sr') is not None]
    if sr_results:
        sr_results_sorted = sorted(sr_results, key=lambda x: x.get('success_rate') or x.get('sr') or 0, reverse=True)
        summary['best_checkpoint'] = sr_results_sorted[0]['ckpt_num']
        summary['best_sr'] = sr_results_sorted[0].get('success_rate') or sr_results_sorted[0].get('sr')
        summary['best_spl'] = sr_results_sorted[0].get('spl')
    
    # 详细结果表格
    table = []
    for r in sorted(all_results, key=lambda x: x['ckpt_num']):
        row = {
            'ckpt_num': r['ckpt_num'],
            'SR': r.get('success_rate') or r.get('sr'),
            'SPL': r.get('spl'),
            'Nav_Error': r.get('navigation_error'),
            'Action_Acc': r.get('action_acc'),
            'Steps': r.get('steps'),
            'Success': r.get('success', False),
            'Elapsed_s': r.get('elapsed_seconds'),
            'Error': r.get('error', ''),
        }
        table.append(row)
    
    summary['details'] = table
    
    # 保存JSON
    summary_json_path = os.path.join(output_base, "summary.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存CSV
    csv_path = os.path.join(output_base, "summary.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        # header
        f.write("ckpt_num,SR,SPL,Nav_Error,Action_Acc,Steps,Success,Elapsed_s,Error\n")
        for row in table:
            sr = row['SR'] if row['SR'] is not None else ''
            spl = row['SPL'] if row['SPL'] is not None else ''
            ne = row['Nav_Error'] if row['Nav_Error'] is not None else ''
            aa = row['Action_Acc'] if row['Action_Acc'] is not None else ''
            steps = row['Steps'] if row['Steps'] is not None else ''
            err = row['Error'] if row['Error'] else ''
            f.write(f"{row['ckpt_num']},{sr},{spl},{ne},{aa},{steps},{row['Success']},{row['Elapsed_s']},{err}\n")
    
    return summary


def print_summary(summary):
    """打印汇总表格"""
    print(f"\n{'='*70}")
    print(f"  批量评估完成")
    print(f"{'='*70}")
    print(f"  总checkpoint数: {summary['total_checkpoints']}")
    print(f"  成功评估: {summary['successful_evals']}  |  失败: {summary['failed_evals']}")
    
    if 'best_checkpoint' in summary:
        print(f"\n  ★ 最佳模型: ckpt.{summary['best_checkpoint']}  (SR={summary['best_sr']:.4f})", end="")
        if summary.get('best_spl') is not None:
            print(f"  SPL={summary['best_spl']:.4f}", end="")
        print()
    
    print(f"\n  详细结果表格:")
    print(f"  {'ckpt':>6} {'SR':>8} {'SPL':>8} {'Nav_Err':>8} {'Action_Acc':>10} {'Steps':>6} {'状态':>6}")
    print(f"  {'-'*60}")
    
    for row in summary['details']:
        sr = f"{row['SR']:.4f}" if row['SR'] is not None else "N/A"
        spl = f"{row['SPL']:.4f}" if row['SPL'] is not None else "N/A"
        ne = f"{row['Nav_Error']:.2f}" if row['Nav_Error'] is not None else "N/A"
        aa = f"{row['Action_Acc']:.4f}" if row['Action_Acc'] is not None else "N/A"
        steps = str(row['Steps']) if row['Steps'] is not None else "N/A"
        status = "✓" if row['Success'] else "✗"
        print(f"  ckpt.{row['ckpt_num']:>3} {sr:>8} {spl:>8} {ne:>8} {aa:>10} {steps:>6} {status:>6}")
    
    print(f"\n  结果已保存到:")
    print(f"    - {os.path.join(summary.get('_output_base', ''), 'summary.json')}")
    print(f"    - {os.path.join(summary.get('_output_base', ''), 'summary.csv')}")
    print(f"    - 每个checkpoint的独立结果: {{output_base}}/ckpt.{{N}}/result.json")
    print(f"{'='*70}")


def main():
    global args
    args = parse_args()
    
    print("=" * 70)
    print("  批量Checkpoint评估脚本")
    print("=" * 70)
    print(f"  Checkpoint目录: {args.ckpt_dir}")
    print(f"  输出目录: {args.output_base}")
    print(f"  Eval配置: {args.config_name}")
    print(f"  数据集: {args.dataset}")
    
    if args.start is not None:
        print(f"  起始checkpoint: {args.start}")
    if args.end is not None:
        print(f"  结束checkpoint: {args.end}")
    
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(args.output_base, exist_ok=True)
    
    # 保存本次运行配置
    config_info = {
        'ckpt_dir': args.ckpt_dir,
        'output_base': args.output_base,
        'config_name': args.config_name,
        'dataset': args.dataset,
        'start': args.start,
        'end': args.end,
        'timeout': args.timeout,
        'start_time': datetime.now().isoformat(),
    }
    config_path = os.path.join(args.output_base, "eval_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    # 获取checkpoint列表
    checkpoints = get_checkpoint_list(args.ckpt_dir, args.start, args.end)
    
    if not checkpoints:
        print("未找到任何checkpoint文件!")
        return
    
    print(f"\n找到 {len(checkpoints)} 个checkpoint待评估:")
    for c in checkpoints:
        print(f"  ckpt.{c['num']:>3}: {c['basename']}")
    
    if args.dry_run:
        print("\n[dry-run模式，仅列出待评估checkpoint]")
        return
    
    # 批量评估
    all_results = []
    start_time = datetime.now()

    for i, ckpt_info in enumerate(checkpoints):
        # 两个checkpoint之间清理GPU显存，避免显存碎片累积导致malloc崩溃
        if i > 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"\n  [GPU缓存已清理，准备评估下一个checkpoint]\n")
            except ImportError:
                pass

        print(f"\n[{i+1}/{len(checkpoints)}] ", end="", flush=True)
        result = run_eval(ckpt_info, args.config_name, args.output_base, args.dataset, args.timeout)
        all_results.append(result)
        
        # 每评估完一个就保存汇总
        summary = save_summary(all_results, args.output_base, start_time)
        summary['_output_base'] = args.output_base
    
    # 最终汇总
    summary = save_summary(all_results, args.output_base, start_time)
    summary['_output_base'] = args.output_base
    print_summary(summary)
    
    print(f"\n✓ 所有评估完成！结果目录: {args.output_base}")


if __name__ == "__main__":
    main()
