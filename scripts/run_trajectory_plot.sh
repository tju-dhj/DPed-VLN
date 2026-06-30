#!/bin/bash
# 快速生成轨迹长度分布图（≤500步）

echo "=========================================="
echo "生成轨迹长度分布可视化"
echo "=========================================="
echo ""

cd /share/home/u14004/dhj/Falcon-main

echo "运行轨迹分布绘图脚本..."
echo "  - 数据范围: ≤500 步"
echo "  - 训练集 + 验证集"
echo ""

python scripts/plot_trajectory_distribution.py \
    --csv dataset_analysis_academic/all_episodes.csv \
    --output dataset_analysis_academic/trajectory_length_distribution.pdf \
    --max_steps 500

echo ""
echo "=========================================="
echo "完成！检查输出文件："
echo "=========================================="

if [ -f "dataset_analysis_academic/trajectory_length_distribution.pdf" ]; then
    ls -lh dataset_analysis_academic/trajectory_length_distribution.pdf
    echo ""
    echo "✓ 轨迹长度分布PDF已生成"
    echo "  - 仅包含 ≤500 步的轨迹"
    echo "  - 图注已优化，避免重叠"
    echo "  - 训练集和验证集分别展示"
else
    echo "✗ 未找到输出文件，请检查错误信息"
fi

echo ""

