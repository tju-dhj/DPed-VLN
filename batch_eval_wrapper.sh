#!/bin/bash
# 用法:
#   bash batch_eval_wrapper.sh 5 10 --ckpt-dir=... --output-base=...
#   bash batch_eval_wrapper.sh --config=DPed_pro/new_data/eval/dped_eval_4a_start_eval_fast.yaml 5 10 --ckpt-dir=... --output-base=...
#
# 可选参数:
#   start_num / end_num      起始/结束 ckpt
#   --dry-run                仅列出
#   --overwrite              覆盖旧结果
#   --config=NAME            eval 配置名（相对于 habitat-baselines 的 config 根目录）
#   --ckpt-dir=PATH          checkpoint 目录
#   --output-base=PATH       输出根目录
#   --dataset=NAME           数据集 split 名字
#   --timeout=SEC            单 ckpt 超时

cd /share/home/u19666033/dhj/dped-vln

START_ARG=""
END_ARG=""
DRY_RUN=""
OVERWRITE=""
CKPT_DIR_ARG=""
OUTPUT_BASE_ARG=""
CONFIG_ARG="--config-name=DPed_VLN/eval/eval_rl_v1_eval_fast.yaml"
DATASET_ARG="--dataset=val_evalfast"
TIMEOUT_ARG="--timeout=7200"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"; shift ;;
        --overwrite)
            OVERWRITE="--overwrite"; shift ;;
        --config=*)
            CONFIG_ARG="--config-name=${1#*=}"; shift ;;
        --ckpt-dir=*)
            CKPT_DIR_ARG="--ckpt-dir=${1#*=}"; shift ;;
        --output-base=*)
            OUTPUT_BASE_ARG="--output-base=${1#*=}"; shift ;;
        --dataset=*)
            DATASET_ARG="--dataset=${1#*=}"; shift ;;
        --timeout=*)
            TIMEOUT_ARG="--timeout=${1#*=}"; shift ;;
        --help|-h)
            cat <<EOF
用法: $0 [start_num] [end_num] [options]

位置参数:
  start_num                 从第几个 checkpoint 开始 (默认: 1)
  end_num                   评估到第几个 checkpoint (默认: 全部)

选项:
  --dry-run                 仅列出待评估的 checkpoint，不实际运行
  --overwrite               覆盖已有评估结果
  --config=NAME             eval 配置名 (Hydra config name)
  --ckpt-dir=PATH           checkpoint 目录 (必填)
  --output-base=PATH        输出根目录 (必填)
  --dataset=NAME            数据集 split 名 (默认: val_evalfast)
  --timeout=SEC             单个 checkpoint 超时时间，秒 (默认: 7200)

示例:
  $0 5 10 \\
    --ckpt-dir=/share/home/u19666033/dhj/DPed_pro/evaluation-vlnce-dpedpro2/rl/4a-base-start/hm3d/checkpoints \\
    --output-base=/share/home/u19666033/dhj/DPed_pro/evaluation-vlnce-dpedpro2/rl/4a-base-start/hm3d/eval_fast_2 \\
    --config=DPed_pro/new_data/eval/dped_eval_4a_start_eval_fast.yaml
EOF
            exit 0 ;;
        *)
            if [[ -z "$START_ARG" ]]; then
                START_ARG="--start=$1"
            elif [[ -z "$END_ARG" ]] && [[ "$1" =~ ^[0-9]+$ ]]; then
                END_ARG="--end=$1"
            fi
            shift ;;
    esac
done

# 必填校验
if [[ -z "$CKPT_DIR_ARG" || -z "$OUTPUT_BASE_ARG" ]]; then
    echo "ERROR: 必须指定 --ckpt-dir 和 --output-base"
    exit 1
fi

echo "[wrapper] CONFIG     : $CONFIG_ARG"
echo "[wrapper] CKPT_DIR   : $CKPT_DIR_ARG"
echo "[wrapper] OUTPUT_BASE: $OUTPUT_BASE_ARG"
echo "[wrapper] DATASET    : $DATASET_ARG"
echo "[wrapper] TIMEOUT    : $TIMEOUT_ARG"

# 执行
python -u batch_eval_checkpoints_dir.py \
    $START_ARG \
    $END_ARG \
    $DRY_RUN \
    $OVERWRITE \
    $CKPT_DIR_ARG \
    $OUTPUT_BASE_ARG \
    $CONFIG_ARG \
    $DATASET_ARG \
    $TIMEOUT_ARG
