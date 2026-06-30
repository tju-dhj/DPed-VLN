#!/bin/bash
# -*- coding: utf-8 -*-
"""
批量评估执行脚本 - 一键运行批量评估并生成可视化报告

使用方法:
    bash run_batch_eval.sh [OPTIONS]

示例:
    # 评估所有 checkpoints
    bash run_batch_eval.sh

    # 只评估第 10-20 个 checkpoints
    bash run_batch_eval.sh --start_idx 10 --end_idx 20

    # 强制重新评估所有 checkpoints
    bash run_batch_eval.sh --force
"""

# ============================================================
# 配置区域 - 根据您的环境修改这些路径
# ============================================================

# Checkpoint 目录（包含 ckpt.*.pth 文件的目录）
CHECKPOINT_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_clip_rl_v2_6actions/hm3d/checkpoints"

# 评估配置文件模板
CONFIG_TEMPLATE="/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines/config/DPed_pro/eval/DPed_rl_val_6action_normalized.yaml"

# 批量评估结果输出目录（留空自动生成）
OUTPUT_DIR=""

# ============================================================
# 命令行参数解析
# ============================================================

# 解析参数
FORCE_FLAG=""
START_IDX=""
END_IDX=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --start_idx)
            START_IDX="--start_idx $2"
            shift 2
            ;;
        --end_idx)
            END_IDX="--end_idx $2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --help|-h)
            echo "用法: bash run_batch_eval.sh [OPTIONS]"
            echo ""
            echo "选项:"
            echo "  --force           强制重新评估所有 checkpoints"
            echo "  --start_idx N     从第 N 个 checkpoint 开始评估"
            echo "  --end_idx N       评估到第 N 个 checkpoint 结束"
            echo "  --dry_run         仅显示将要执行的操作，不实际运行"
            echo "  --help, -h        显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  bash run_batch_eval.sh --start_idx 10 --end_idx 20"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================
# 主执行流程
# ============================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT="$(pwd)"

echo "========================================"
echo "批量评估执行脚本"
echo "========================================"
echo "项目根目录: $PROJECT_ROOT"
echo "Checkpoint 目录: $CHECKPOINT_DIR"
echo "配置模板: $CONFIG_TEMPLATE"
echo ""

# 设置默认输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/batch_eval_results"
fi
echo "输出目录: $OUTPUT_DIR"
echo "========================================"
echo ""

# 检查依赖
echo "[1/3] 检查依赖..."
if [ ! -f "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] Checkpoint 目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_TEMPLATE" ]; then
    echo "[ERROR] 配置模板不存在: $CONFIG_TEMPLATE"
    exit 1
fi

# 检查 Python 脚本
if [ ! -f "$PROJECT_ROOT/batch_eval.py" ]; then
    echo "[ERROR] 缺少 batch_eval.py 脚本"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/plot_results.py" ]; then
    echo "[ERROR] 缺少 plot_results.py 脚本"
    exit 1
fi

echo "[OK] 依赖检查通过"
echo ""

# 统计 checkpoints 数量
NUM_CKPTS=$(ls "$CHECKPOINT_DIR"/ckpt.*.pth 2>/dev/null | grep -v "latest" | wc -l)
echo "[INFO] 发现 $NUM_CKPTS 个 checkpoints"
echo ""

# ============================================================
# Step 1: 运行批量评估
# ============================================================

echo "[2/3] 开始批量评估..."
echo ""

# 构造命令
CMD="python $PROJECT_ROOT/batch_eval.py \
    --ckpt_dir \"$CHECKPOINT_DIR\" \
    --config_template \"$CONFIG_TEMPLATE\" \
    --output_dir \"$OUTPUT_DIR\" \
    $FORCE_FLAG \
    $START_IDX \
    $END_IDX \
    $DRY_RUN"

echo "执行命令:"
echo "$CMD"
echo ""

if [ -n "$DRY_RUN" ]; then
    echo "[DRY RUN] 跳过实际评估"
else
    eval "$CMD"
    EVAL_EXIT_CODE=$?

    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "[ERROR] 批量评估失败，退出码: $EVAL_EXIT_CODE"
        exit $EVAL_EXIT_CODE
    fi

    echo "[OK] 批量评估完成"
fi

echo ""

# ============================================================
# Step 2: 生成可视化报告
# ============================================================

echo "[3/3] 生成可视化报告..."
echo ""

PLOT_CMD="python $PROJECT_ROOT/plot_results.py \
    --input_dir \"$OUTPUT_DIR\" \
    --output_dir \"$OUTPUT_DIR/plots\" \
    --metrics SR SPL PSC H-Coll Total"

echo "执行命令:"
echo "$PLOT_CMD"
echo ""

eval "$PLOT_CMD"
PLOT_EXIT_CODE=$?

if [ $PLOT_EXIT_CODE -ne 0 ]; then
    echo "[WARNING] 可视化生成失败，退出码: $PLOT_EXIT_CODE"
else
    echo "[OK] 可视化生成完成"
fi

echo ""
echo "========================================"
echo "批量评估全部完成！"
echo "========================================"
echo ""
echo "结果目录: $OUTPUT_DIR"
echo "图表目录: $OUTPUT_DIR/plots"
echo ""
echo "查看结果:"
echo "  - 汇总报告: $OUTPUT_DIR/batch_summary.json"
echo "  - 统计报告: $OUTPUT_DIR/plots/statistics_report.txt"
echo "  - 指标曲线: $OUTPUT_DIR/plots/"
echo ""
