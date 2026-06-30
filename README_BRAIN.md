# -*- coding: utf-8 -*-
# ==============================================================================
# 文件: README_BRAIN.md
# 描述: Brain模块使用说明文档
# ==============================================================================

# Brain模块 - 基于行人检测和Gemma大模型的"外接大脑"系统

## 目录

1. [系统概述](#系统概述)
2. [模块架构](#模块架构)
3. [快速开始](#快速开始)
4. [配置说明](#配置说明)
5. [代码结构](#代码结构)
6. [训练指南](#训练指南)
7. [评估指南](#评估指南)
8. [扩展开发](#扩展开发)
9. [常见问题](#常见问题)

---

## 系统概述

### 1.1 设计背景

在原有的DPed_pro视觉-语言-导航(VLN)框架中，机器人的导航决策完全依赖于策略网络。面对动态行人的复杂场景，策略网络可能无法及时做出最优的避让决策。

### 1.2 解决方案

本模块引入"外接大脑"概念：
1. **行人检测模块**：实时检测场景中的行人，获取位置、距离、密度等信息
2. **Gemma Brain**：基于轻量化大模型(Gemma-4 E2B)分析行人状态，生成优化指令
3. **决策融合**：在行人近距离时，Gemma Brain的决策可覆盖策略网络的输出

### 1.3 核心特性

- **多检测器支持**：YOLOv8、RT-DETR等
- **轻量级大模型**：Gemma-4 E2B (2.3B有效参数)
- **训练友好**：冻结Brain参数，只训练主策略网络
- **即插即用**：通过配置开关控制，不影响原有系统

---

## 模块架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      Brain增强VLN系统                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   RGB图像    │───▶│ 行人检测模块    │───▶│ 行人状态信息    │  │
│  └─────────────┘    │ (YOLO/RT-DETR) │    │ (位置/距离等)   │  │
│                      └─────────────────┘    └────────┬────────┘  │
│                                                      │           │
│                                                      ▼           │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ 导航指令     │───▶│   Gemma Brain   │◀───│ 行人状态信息    │  │
│  │ + 环境上下文 │    │   (外接大脑)    │    │                 │  │
│  └─────────────┘    └────────┬────────┘    └─────────────────┘  │
│                               │                                   │
│                               ▼                                   │
│                      ┌─────────────────┐                         │
│                      │   优化指令       │                         │
│                      │ (STOP/PAUSE等)  │                         │
│                      └────────┬────────┘                         │
│                               │                                   │
│                               ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    决策融合层                             │    │
│  │  高置信度行人警告 → Brain决策覆盖策略网络输出               │    │
│  │  正常情况 → 使用策略网络原始决策                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                   │
│                               ▼                                   │
│                      ┌─────────────────┐                         │
│                      │   动作执行      │                         │
│                      │   (STOP等6种)   │                         │
│                      └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

1. **输入**：RGB图像 + 导航指令 + 环境状态
2. **行人检测**：使用YOLO/RT-DETR检测行人，输出位置、置信度、相对面积
3. **Gemma Brain推理**：
   - 输入：行人信息 + 导航上下文
   - 输出：建议动作 + 置信度 + 推理过程
4. **决策融合**：
   - 行人近距离 + 高置信度 → 覆盖策略网络
   - 其他情况 → 使用策略网络输出

---

## 快速开始

### 3.1 环境准备

```bash
# 1. 安装必要的依赖
pip install ultralytics transformers torch
pip install habitat-lab habitat-baselines

# 2. 下载数据集 (参考原有文档)
bash pedestrian_benchmark/download_assets.sh

# 3. 下载Gemma模型 (首次运行时自动下载)
# 模型将缓存在 HF_HOME 目录
```

### 3.2 快速训练

```bash
# 进入项目目录
cd /share/home/u19666033/dhj/DPed_pro

# 启用Brain的训练
bash main_slurm_brain_ppo_train_v2_ddppo.bash

# 或禁用Brain (对比实验)
BRAIN_ENABLED=false bash main_slurm_brain_ppo_train_v2_ddppo.bash
```

### 3.3 快速评估

```bash
# 评估带Brain的模型
bash main_slurm_brain_ppo_rl_val.bash \
    --checkpoint /path/to/checkpoint.pth \
    --num_episodes 100
```

---

## 配置说明

### 4.1 Brain模块配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `brain.enabled` | bool | true | 是否启用Brain |
| `brain.pedestrian_enabled` | bool | true | 是否启用行人检测 |
| `brain.pedestrian_detector` | str | "yolov8n" | 检测器类型 |
| `brain.model_type` | str | "gemma4_e2b" | Gemma模型类型 |
| `brain.override_threshold` | float | 0.8 | 覆盖阈值 |
| `brain.freeze_brain` | bool | true | 冻结Brain参数 |

### 4.2 行人检测器选项

| 检测器 | 参数量 | 速度 | 精度 | 适用场景 |
|--------|--------|------|------|----------|
| yolov8n | 3.2M | 最快 | 中等 | 实时推理 |
| yolov8s | 11.2M | 快 | 较高 | 平衡场景 |
| yolov8m | 25.9M | 中等 | 高 | 高精度需求 |
| rtdetr_r18 | ~20M | 中等 | 高 | DETR系首选 |
| rtdetr_r50 | ~40M | 较慢 | 最高 | 最高精度 |

### 4.3 Gemma模型选项

| 模型 | 有效参数 | 量化后大小 | 说明 |
|------|----------|------------|------|
| gemma4_e2b | 2.3B | ~3GB | 轻量首选 |
| gemma4_e4b | 4.5B | ~6GB | 更高精度 |

### 4.4 配置示例

```yaml
# 训练配置
habitat_baselines:
  brain:
    enabled: true
    pedestrian_enabled: true
    pedestrian_detector: "yolov8n"     # 轻量检测器
    pedestrian_confidence: 0.25
    model_type: "gemma4_e2b"          # 轻量大模型
    brain_device: "cuda"
    override_threshold: 0.8
    freeze_brain: true                 # 冻结Brain
    freeze_pedestrian: true           # 冻结检测器
```

---

## 代码结构

### 5.1 文件列表

```
habitat-baselines/habitat_baselines/rl/ppo/
├── brain/                           # Brain模块目录
│   ├── __init__.py                  # 模块导出
│   ├── pedestrian_detection.py       # 行人检测器
│   ├── gemma_brain.py               # Gemma Brain
│   ├── prompts.py                   # Prompt模板
│   └── utils.py                    # 工具函数
├── brain_ppo_trainer.py             # Brain训练器
└── brain_ppo_evaluator.py           # Brain评估器

habitat-baselines/habitat_baselines/config/DPed_brain/
├── brain_ppo_train_v2_ddppo.yaml    # 训练配置
└── brain_ppo_rl_val.yaml            # 评估配置

DPed_pro/
├── main_slurm_brain_ppo_train_v2_ddppo.bash  # 训练脚本
└── main_slurm_brain_ppo_rl_val.bash          # 评估脚本
```

### 5.2 核心类

#### PedestrianDetector
- 统一行人检测接口
- 支持YOLO和RT-DETR
- 返回标准化检测结果

#### GemmaBrain
- Gemma模型推理封装
- Prompt构建与解析
- 决策生成与格式化

#### BrainManager
- 统一管理行人检测和Brain
- 感知-决策流程编排
- 配置化开关控制

#### BrainPPOTrainer
- 继承DynamicVLNTrainer
- 集成Brain感知
- 训练统计收集

#### BrainPPOEvaluator
- 继承FALCONEvaluator
- 详细评估报告
- 行人轨迹分析

---

## 训练指南

### 6.1 标准训练流程

```bash
# 1. 准备数据集
# (参考DPed_pro原有文档)

# 2. 配置参数
# 编辑 brain_ppo_train_v2_ddppo.yaml 或使用命令行覆盖

# 3. 提交训练任务
sbatch main_slurm_brain_ppo_train_v2_ddppo.bash

# 4. 监控训练
tensorboard --logdir tb_logs/brain_ppo_train
```

### 6.2 训练策略

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `freeze_brain` | true | 冻结Brain参数 |
| `freeze_pedestrian` | true | 冻结行人检测器 |
| `freeze_clip` | true | 冻结视觉编码器 |
| `brain.override_threshold` | 0.8 | 合理的覆盖阈值 |

### 6.3 常用命令

```bash
# 单GPU训练
python -m habitat_baselines.run \
    --config-name=DPed_brain/brain_ppo_train_v2_ddppo \
    habitat_baselines.num_environments=2

# 多GPU训练
torchrun --nproc_per_node=8 ...

# 调整学习率
LEARNING_RATE=5e-5 bash main_slurm_brain_ppo_train_v2_ddppo.bash

# 使用更强的检测器
PEDESTRIAN_DETECTOR=yolov8m bash main_slurm_brain_ppo_train_v2_ddppo.bash
```

### 6.4 训练输出

```
outputs/brain_ppo_train/
├── checkpoints/                      # 模型检查点
│   └── brain_ppo_train_epoch_100.pth
├── tb_logs/                          # Tensorboard日志
└── brain_stats/                      # Brain统计
    └── brain_ppo_train_brain_stats.json
```

---

## 评估指南

### 7.1 标准评估流程

```bash
# 1. 准备检查点
# CHECKPOINT_PATH=/path/to/checkpoint.pth

# 2. 运行评估
bash main_slurm_brain_ppo_rl_val.bash \
    --checkpoint $CHECKPOINT_PATH \
    --num_episodes 100
```

### 7.2 评估指标

| 指标 | 说明 |
|------|------|
| Success Rate (SR) | 任务成功率 |
| SPL | 路径效率加权成功率 |
| Pedestrian Detection Rate | 行人检测率 |
| Brain Override Rate | Brain覆盖策略网络的比率 |
| Average Detection Latency | 平均检测延迟(ms) |

### 7.3 评估报告

评估完成后，生成以下文件：

```
outputs/brain_ppo_eval/
├── brain_evaluation_report.json       # 详细JSON报告
├── brain_evaluation_summary.csv       # CSV摘要
└── videos/                            # 评估视频(可选)
```

### 7.4 对比评估

```bash
# 有Brain vs 无Brain对比
# 1. 有Brain评估
bash main_slurm_brain_ppo_rl_val.bash \
    --checkpoint $CHECKPOINT \
    --brain_enabled \
    --output brain_with_brain

# 2. 无Brain评估
bash main_slurm_brain_ppo_rl_val.bash \
    --checkpoint $CHECKPOINT \
    --brain_disabled \
    --output brain_without_brain

# 3. 对比分析
python -c "
import json
with_brain = json.load(open('brain_with_brain/report.json'))
without_brain = json.load(open('brain_without_brain/report.json'))
print('SR差异:', with_brain['success_rate'] - without_brain['success_rate'])
"
```

---

## 扩展开发

### 8.1 添加新的检测器

```python
# 在 pedestrian_detection.py 中添加新检测器类型

class DetectorType(Enum):
    # ... 现有类型 ...
    NEW_DETECTOR = "new_detector"

class PedestrianDetector:
    def _initialize_model(self):
        if self.detector_type == DetectorType.NEW_DETECTOR:
            self._init_new_detector()

    def _init_new_detector(self):
        # 实现新检测器初始化
        pass

    def _detect_new(self, image, frame_id):
        # 实现新检测器推理
        pass
```

### 8.2 添加新的Brain模型

```python
# 在 gemma_brain.py 中添加新模型类型

class BrainModelType(Enum):
    # ... 现有类型 ...
    NEW_MODEL = "new_model"

class GemmaBrain:
    def _initialize_model(self, model_id=None):
        if self.model_type == BrainModelType.NEW_MODEL:
            self._init_new_model()
        # ... 其他模型初始化 ...

    def generate_instruction(self, context):
        if self.model_type == BrainModelType.NEW_MODEL:
            return self._generate_new_model(context)
        # ... 其他模型推理 ...
```

### 8.3 自定义Prompt模板

```python
# 在 prompts.py 中添加新模板

CUSTOM_SYSTEM_PROMPT = """
你是自定义导航助手...
"""

def build_custom_prompt(context):
    # 自定义prompt构建逻辑
    pass
```

---

## 常见问题

### Q1: 模型加载失败

**问题**: Gemma模型下载或加载失败

**解决方案**:
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载
huggingface-cli download google/gemma-4-E2B
```

### Q2: GPU内存不足

**问题**: 训练时OOM

**解决方案**:
```bash
# 使用更小的Gemma模型
GEMMA_MODEL=gemma4_e2b bash main_slurm_brain_ppo_train_v2_ddppo.bash

# 或减少batch size
habitat_baselines.rl.ppo.num_mini_batch=1
```

### Q3: 检测速度慢

**问题**: 行人检测成为瓶颈

**解决方案**:
```bash
# 使用更快的检测器
PEDESTRIAN_DETECTOR=yolov8n bash main_slurm_brain_ppo_train_v2_ddppo.bash

# 或减少检测频率
# (需要修改代码，每N帧检测一次)
```

### Q4: Brain覆盖过多

**问题**: Brain决策过度覆盖策略

**解决方案**:
```bash
# 提高覆盖阈值
BRAIN_OVERRIDE_THRESHOLD=0.95 bash main_slurm_brain_ppo_train_v2_ddppo.bash

# 或完全禁用Brain
BRAIN_ENABLED=false bash main_slurm_brain_ppo_train_v2_ddppo.bash
```

### Q5: 如何恢复到原始系统

**方案**: 禁用Brain模块即可

```bash
# 方式1: 环境变量
BRAIN_ENABLED=false bash main_slurm_brain_ppo_train_v2_ddppo.bash

# 方式2: 使用原有的非Brain配置
# 使用 dynamic_vlnce_ddppo_hm3d_train_v2_ddppo.yaml
```

---

## 参考文献

1. Gemma 4 Technical Report - Google DeepMind
2. YOLOv8: Ultralytics
3. RT-DETR: Real-Time Detection Transformer
4. Habitat: A Platform for Embodied AI Research
5. DynamicVLNCE: Dynamic Vision-Language Navigation with Crowd Agents

---

## 更新日志

### v1.0.0 (2026-04)
- 初始版本发布
- 支持Gemma-4 E2B/E4B
- 支持YOLOv8和RT-DETR检测器
- 完整的训练和评估流程
