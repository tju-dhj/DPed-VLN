#!/bin/bash
# 修复可编辑安装的路径问题
# 此脚本将重新安装habitat-baselines和habitat-lab，指向正确的路径

echo "========================================="
echo "修复可编辑安装路径"
echo "========================================="

# 激活conda环境
source ~/.bashrc 2>/dev/null || true
conda activate falcon

# 获取当前项目路径
FALCON_ROOT="/share/home/u14004/dhj/Falcon-main"
HABITAT_BASELINES_DIR="$FALCON_ROOT/habitat-baselines"
HABITAT_LAB_DIR="$FALCON_ROOT/habitat-lab"

echo "项目根目录: $FALCON_ROOT"
echo "habitat-baselines目录: $HABITAT_BASELINES_DIR"
echo "habitat-lab目录: $HABITAT_LAB_DIR"

# 检查目录是否存在
if [ ! -d "$HABITAT_BASELINES_DIR" ]; then
    echo "错误: $HABITAT_BASELINES_DIR 不存在！"
    exit 1
fi

if [ ! -d "$HABITAT_LAB_DIR" ]; then
    echo "错误: $HABITAT_LAB_DIR 不存在！"
    exit 1
fi

# 先卸载旧的安装（如果存在）
echo ""
echo "步骤1: 卸载旧的habitat-baselines和habitat-lab..."
pip uninstall -y habitat-baselines habitat-lab 2>/dev/null || true

# 重新安装habitat-baselines（可编辑模式）
echo ""
echo "步骤2: 重新安装habitat-baselines（可编辑模式）..."
cd "$HABITAT_BASELINES_DIR"
pip install -e . --no-deps

# 重新安装habitat-lab（可编辑模式）
echo ""
echo "步骤3: 重新安装habitat-lab（可编辑模式）..."
cd "$HABITAT_LAB_DIR"
pip install -e . --no-deps

echo ""
echo "========================================="
echo "安装完成！"
echo "========================================="
echo ""
echo "验证安装:"
python -c "import habitat_baselines; print('habitat_baselines location:', habitat_baselines.__file__)" 2>&1 | head -5
python -c "import habitat; print('habitat location:', habitat.__file__)" 2>&1 | head -5







