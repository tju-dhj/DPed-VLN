#!/bin/bash
# ==============================================================================
# 文件: download_yolo.sh
# 描述: 下载 YOLO 行人检测模型（n/s/m 三种精度）
# ==============================================================================

SAVE_DIR="/share/home/u19666033/dhj/DPed_pro/pretrained_model"
YOLO_DIR="${SAVE_DIR}/yolo_models"
mkdir -p "${YOLO_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  YOLO 行人检测模型下载${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "保存路径: ${GREEN}${YOLO_DIR}${NC}"
echo ""

echo -e "${BLUE}检查磁盘空间...${NC}"
available_space=$(df -h "${SAVE_DIR}" | tail -1 | awk '{print $4}')
echo -e "可用空间: ${available_space}"
echo ""
echo -e "${YELLOW}开始下载 YOLO 模型...${NC}"
echo ""

python -u << EOF
import os
import shutil
from ultralytics import YOLO

models = {
    "yolov8n-seg": "yolov8n-seg.pt",  # nano: 最轻量
    "yolov8s-seg": "yolov8s-seg.pt",  # small: 推荐
    "yolov8m-seg": "yolov8m-seg.pt",  # medium: 更高精度
}

for name, filename in models.items():
    save_path = os.path.join("${YOLO_DIR}", filename)
    if os.path.exists(save_path):
        print("Skip (exists): " + save_path)
    else:
        print("Downloading: " + name + " ...", flush=True)
        model = YOLO(name + ".pt")
        src = name + ".pt"
        if os.path.exists(src):
            shutil.move(src, save_path)
            print("Saved: " + save_path)

print()
print("All YOLO models downloaded!")
print()
import os
for f in os.listdir("${YOLO_DIR}"):
    full = os.path.join("${YOLO_DIR}", f)
    if os.path.isfile(full):
        size_mb = os.path.getsize(full) / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  YOLO 模型下载完成！${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "模型目录: ${YOLO_DIR}"
    echo ""
    ls -lh "${YOLO_DIR}"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  YOLO 模型下载失败！${NC}"
    echo -e "${RED}============================================${NC}"
    exit 1
fi
