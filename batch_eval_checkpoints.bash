#!/bin/bash
# 批量评估便捷启动脚本
# 功能：监控训练过程中的checkpoint并自动评估，实时绘制SR曲线

# 默认参数
CHECKPOINT_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vln/dped_pro_social_eq/hm3d/checkpoints"
CONFIG_NAME="DPed_pro/eval/DPed_rl_val_6action"
POLL_INTERVAL=300
EVAL_TIMEOUT=7200
ONCE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --poll)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --timeout)
            EVAL_TIMEOUT="$2"
            shift 2
            ;;
        --once)
            ONCE="true"
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --checkpoint_dir <路径>  checkpoint目录"
            echo "  --config <名称>          Hydra配置名称"
            echo "  --poll <秒数>            轮询间隔"
            echo "  --timeout <秒数>         评估超时"
            echo "  --once                   只运行一次"
            echo "  --help, -h               显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")"

CMD="python -u batch_eval_checkpoints.py \
    --checkpoint_dir '$CHECKPOINT_DIR' \
    --config_name '$CONFIG_NAME' \
    --poll_interval $POLL_INTERVAL \
    --eval_timeout $EVAL_TIMEOUT"

if [ "$ONCE" = "true" ]; then
    CMD="$CMD --once"
fi

echo "=============================================="
echo "批量评估启动脚本"
echo "=============================================="
echo "Checkpoint目录: $CHECKPOINT_DIR"
echo "配置文件: $CONFIG_NAME"
echo "轮询间隔: ${POLL_INTERVAL}秒"
echo "评估超时: ${EVAL_TIMEOUT}秒"
echo "模式: ${ONCE:+单次评估}else{持续监控}"
echo "=============================================="
echo ""
echo "按 Ctrl+C 中断运行"
echo "=============================================="
echo ""

eval $CMD
