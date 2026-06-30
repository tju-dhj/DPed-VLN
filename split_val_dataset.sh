#!/bin/bash
# 将验证集数据分成5份，通过软连接创建5个子目录

BASE_DIR="/share/home/u14004/dhj/Falcon-main/data/datasets/pointnav/social-hm3d"
VAL_DIR="$BASE_DIR/val"
CONTENT_DIR="$VAL_DIR/content"

echo "================================================================================"
echo "将验证集分成5份（通过软连接）"
echo "================================================================================"

# 检查源目录是否存在
if [ ! -d "$CONTENT_DIR" ]; then
    echo "错误: 源目录不存在: $CONTENT_DIR"
    exit 1
fi

# 获取所有场景文件（已排序）
cd "$CONTENT_DIR"
scenes=($(ls *.json.gz | sort))
total=${#scenes[@]}

echo "找到 $total 个场景文件"
echo ""

# 计算每份的场景数量
scenes_per_split=$(( (total + 4) / 5 ))  # 向上取整
echo "每份约 $scenes_per_split 个场景"
echo ""

# 创建5个子目录
for i in {1..5}; do
    split_dir="$BASE_DIR/val_split_$i"
    split_content="$split_dir/content"
    
    echo "创建 val_split_$i ..."
    
    # 如果目录已存在，先询问是否删除
    if [ -d "$split_dir" ]; then
        echo "  警告: $split_dir 已存在"
        read -p "  是否删除并重新创建? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$split_dir"
        else
            echo "  跳过 val_split_$i"
            continue
        fi
    fi
    
    # 创建目录结构
    mkdir -p "$split_content"
    
    # 计算这一份的场景范围
    start=$(( (i - 1) * scenes_per_split ))
    end=$(( i * scenes_per_split ))
    if [ $end -gt $total ]; then
        end=$total
    fi
    
    count=0
    echo "  分配场景 $((start + 1)) 到 $end (共 $((end - start)) 个)"
    
    # 创建软连接
    for (( j=start; j<end; j++ )); do
        scene="${scenes[$j]}"
        ln -sf "$CONTENT_DIR/$scene" "$split_content/$scene"
        count=$((count + 1))
    done
    
    echo "  ✓ 创建了 $count 个软连接"
    echo ""
done

# 显示统计信息
echo "================================================================================"
echo "分割完成！统计信息："
echo "================================================================================"
for i in {1..5}; do
    split_dir="$BASE_DIR/val_split_$i"
    if [ -d "$split_dir/content" ]; then
        count=$(ls "$split_dir/content"/*.json.gz 2>/dev/null | wc -l)
        echo "val_split_$i: $count 个场景"
        
        # 显示前3个场景作为示例
        echo "  示例场景:"
        ls "$split_dir/content"/*.json.gz 2>/dev/null | head -3 | xargs -n1 basename | sed 's/^/    - /'
    fi
done

echo ""
echo "================================================================================"
echo "使用方法："
echo "================================================================================"
echo "在配置文件中设置以下路径进行分别采集："
echo ""
for i in {1..5}; do
    echo "  采集任务 $i: $BASE_DIR/val_split_$i/content/{scene}.json.gz"
done
echo ""
echo "或者在配置中使用通配符："
echo "  data_path: \"$BASE_DIR/val_split_X/content/{scene}.json.gz\""
echo "  (将X替换为1-5)"
echo "================================================================================"

