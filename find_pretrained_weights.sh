#!/bin/bash
# 查找预训练权重文件

echo "================================================================================"
echo "查找预训练权重文件"
echo "================================================================================"

echo -e "\n1. 查找所有.pth文件（可能需要几分钟）..."
find /share/home/u14004/dhj -name "*.pth" -type f 2>/dev/null | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  $file ($size)"
done | head -30

echo -e "\n2. 查找包含'falcon'或'pretrain'的.pth文件..."
find /share/home/u14004/dhj -name "*.pth" -type f 2>/dev/null | grep -i -E "falcon|pretrain" | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  $file ($size)"
done

echo -e "\n3. 检查常见目录..."
dirs=(
    "/share/home/u14004/dhj/Falcon-main/data/models"
    "/share/home/u14004/dhj/Falcon-main/pretrained"
    "/share/home/u14004/dhj/Falcon-main/checkpoints"
    "/share/home/u14004/dhj/Falcon-main/data/pretrained"
    "/share/home/u14004/dhj/Falcon/pretrained_model"
    "/share/home/u14004/dhj/Falcon/data/models"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "\n  检查: $dir"
        ls -lh "$dir"/*.pth 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
    fi
done

echo -e "\n4. 当前配置文件中指定的路径:"
echo "  /share/home/u14004/dhj/Falcon/pretrained_model/falcon_pretrained_25.pth"
if [ -f "/share/home/u14004/dhj/Falcon/pretrained_model/falcon_pretrained_25.pth" ]; then
    size=$(du -h "/share/home/u14004/dhj/Falcon/pretrained_model/falcon_pretrained_25.pth" | cut -f1)
    echo "  ✓ 文件存在 ($size)"
else
    echo "  ✗ 文件不存在"
fi

echo -e "\n================================================================================"
echo "完成！"
echo "================================================================================"
echo -e "\n如果找到了合适的.pth文件，请更新配置文件中的路径："
echo "  文件: habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_direct_il_train_v2.yaml"
echo "  参数: habitat_baselines.il.model.pretrained_weights"
echo -e "\n如果没有找到预训练权重，建议："
echo "  1. 设置 pretrained: False"
echo "  2. 降低学习率到 1e-4 或 2.5e-5"
echo "  3. 考虑解冻更多参数"

