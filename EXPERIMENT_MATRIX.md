# DPed-VLN 实验矩阵: StreamVLN / NaVILA / NaVid 完整实验清单

## 1. 模型与环境概述

| 模型 | 训练方法 | 评估方法 | Conda环境 | GPU需求 |
|------|---------|---------|-----------|---------|
| NaVILA | LoRA SFT (llava) / LoRA Habitat | Habitat模拟器 (generate) | sk_streamvln(SFT) / falcon(Habitat) | 1×L40/A800 |
| StreamVLN | LoRA SFT (streamvln_train) / LoRA Habitat | Habitat模拟器 (generate) | sk_streamvln(SFT) / falcon(Habitat) | 1×L40/A800 |
| NaVid | LoRA Habitat (direct_il) | Habitat模拟器 (generate) | falcon | 1×L40 |

### 训练策略详解

**A) 旧版 Habitat 方法** (navilla_lora_v1_train.bash, streamvln_lora_v1_train.bash, navid_lora_v1_train.bash):
- 使用 `python -m habitat_baselines.run --config-name=DPed_vlm/xxx/v1_train.yaml`
- Action sequence: window_size=0 (完整 trajectory 作为一个序列)
- **NaVILA/StreamVLN**: VLM generate() 在 torch.no_grad() 中 → **仅训练 ~2K 参数的动作头, LoRA 不更新**
- **NaVid**: 类似但有自己的 VLM 结构
- batch_size=1, epochs=1, max_episodes=500

**B) 新版 SFT 方法** (navilla_lora_sft_*.bash, streamvln_lora_v*_llava_train.bash):
- 使用 LLaVA train_mem.py (教师强制前向传播)
- **LoRA 适配器真正收到梯度并训练**
- NaVILA: 27,522 条 conversation 样本 (v2)
- StreamVLN: 使用 streamvln_train.py (Qwen2-7B + SigLIP)
- **比旧方法快 5-10 倍, 且真正训练了 VLM**

### 评估类型

| 评估类型 | 含义 | 配置目录 |
|---------|------|---------|
| zero_shot | 预训练模型, 动态行人 | DPed_vlm/xxx/zero_shot/v*_{split}.yaml |
| zero_shot_static | 预训练模型, 静态行人 | DPed_vlm/xxx/zero_shot_static/v*_{split}.yaml |
| lora | LoRA 微调后, 动态行人 | DPed_vlm/xxx/lora/v*_{split}.yaml |
| human_static | 微调后, 静态行人 | DPed_vlm/xxx/human_static/v*_{split}.yaml |

---

## 2. 数据集

| Split | v1 文件数 | v2 文件数 | 每文件约 episodes | 预估总 episodes |
|-------|----------|----------|------------------|----------------|
| train | 548 | 548 | 1-3 | ~1000 |
| val_seen | 333 | 333 | 1 | ~333 |
| val_unseen | 64 | 64 | 1 | ~64 |
| test_unseen | 61 | 61 | 1 | ~61 |

### StreamVLN SFT 数据
- v1: annotations.json (via convert_dped_to_streamvln.py)
- v2: 27,522 samples (navilla_conversations.json)
- 格式: `<video>\n{instruction}` → `TURN LEFT, MOVE FORWARD, ..., STOP`

### 采样数据 (用于快速测试)
- 位置: `/share/home/u19666033/dhj/dped-vln/DPed_VLN/sampled_data/{v1,v2}/{split}/`
- 每个 split 10 个文件

---

## 3. 时间预估

### 训练时间

| 实验 | 模型 | 数据量 | GPU | 预估时间 |
|------|------|--------|-----|---------|
| navilla_lora_sft (SFT) 1GPU | NaVILA | 27,522 samples | 1×L40 | ~6h/epoch |
| navilla_lora_sft (SFT) 6GPU | NaVILA | 27,522 samples | 6×A800 | ~45min/epoch |
| streamvln_lora_llava 1GPU | StreamVLN | 27,522 samples | 1×A800 | ~8-10h/epoch |
| navilla_lora_v1_train (Habitat) | NaVILA | 548 files | 1×L40 | ~6-15h/epoch |
| streamvln_lora_v1_train (Habitat) | StreamVLN | 548 files | 1×L40 | ~6-15h/epoch |
| navid_lora_v1_train (Habitat) | NaVid | 548 files | 1×L40 | ~6-15h/epoch |
| human_static train | 各模型 | 548 files | 1×L40 | ~4-8h/epoch |

### 评估时间 (1×L40, per split, 预估)

| 模型 | Zero-Shot | Zero-Shot Static | LoRA Eval | Human Static Eval |
|------|-----------|-----------------|-----------|-------------------|
| NaVILA | ~2-4h | ~1-2h | ~2-4h | ~1-2h |
| StreamVLN | ~3-6h | ~2-3h | ~3-6h | ~2-3h |
| NaVid | ~2-4h | ~1-2h | ~2-4h | ~1-2h |

> 每 split: val_seen (~333 eps), val_unseen (~64 eps), test_unseen (~61 eps)
> eval_fast: ~80 episodes (~30-60min)
> 动态行人版本更慢 (需要模拟行人运动), static 版本快 50%+

### 实测数据 (来自 slurm_logs)

| 实验 | 时间 | 备注 |
|------|------|------|
| NaVILA ZS v1 val_seen | 未完成 (OOM/超时) | 需要修复 |
| StreamVLN ZS v1 val_unseen | 未稳定完成 | LLM输出解析问题 |
| NaVILA SFT Train 1GPU | 6:22 (20 steps) | 0.64 steps/sec |

---

## 4. 所有 sbatch 命令

### NaVILA - Zero-Shot

```bash
# V1 动态行人 (val_seen, val_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_zero_shot_v1.bash

# V2 动态行人 (val_seen, val_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_zero_shot_v2.bash

# V1 静态行人 (val_seen, val_unseen, test_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_zero_shot_static_v1.bash

# V2 静态行人 (val_seen, val_unseen, test_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_zero_shot_static_v2.bash
```

### NaVILA - Fine-tune

```bash
# SFT 训练 v2 (1GPU 测试 - 20步)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_lora_v2_llava_train_1gpu.bash

# SFT 训练 v2 (6GPU 生产 - 完整1epoch)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_lora_sft_6gpu_train.bash

# Habitat LoRA 训练 v1 (旧方法, 单卡L40)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_lora_v1_train.bash

# Habitat LoRA 训练 v2 (旧方法, 单卡L40)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_lora_v2_train.bash
```

### NaVILA - Fine-tune Eval

```bash
# LoRA eval v1 (val_seen, val_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_lora_v1_eval.bash

# LoRA eval v2 (val_seen, val_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_lora_v2_eval.bash

# Human static eval v1 (val_seen, val_unseen, test_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_human_static_v1_eval.bash

# Human static eval v2 (val_seen, val_unseen, test_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navilla_human_static_v2_eval.bash
```

### StreamVLN - Zero-Shot

```bash
# V1 动态 (val_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_zero_shot_v1.bash
# V2 动态 (val_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_zero_shot_v2.bash
# V1 静态 (val_seen, val_unseen, test_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_zero_shot_static_v1.bash
# V2 静态 (val_seen, val_unseen, test_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_zero_shot_static_v2.bash
```

### StreamVLN - Fine-tune

```bash
# SFT 训练 v2 (1×A800)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v2_llava_train.bash
# SFT 续训 v2 (+2 epochs)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v2_llava_resume_2more.bash
# Habitat 训练 v1
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v1_train.bash
# Habitat 训练 v2
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v2_train.bash
```

### StreamVLN - Fine-tune Eval

```bash
# LoRA eval v1/v2, val_unseen
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v1_eval.bash
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v2_eval.bash
# val_unseen specific
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v1_val_unseen.bash
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v2_val_unseen.bash
# Static eval v1/v2
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_human_static_v1_eval.bash
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_human_static_v2_eval.bash
# val_unseen static
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v1_val_unseen_static.bash
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/streamvln_lora_v2_val_unseen_static.bash
```

### NaVid - Zero-Shot

```bash
# V1 动态
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navid_zero_shot_v1.bash
# V1 静态
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navid_zero_shot_static_v1.bash
```

### NaVid - Fine-tune + Eval

```bash
# 训练 v1
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navid_lora_v1_train.bash
# Eval v1 (val_seen, val_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navid_lora_v1_eval.bash
# Static eval v1 (val_seen, val_unseen, test_unseen)
cd /share/home/u19666033/dhj/dped-vln && sbatch sbatch/DPed_vlm/navid_human_static_v1_eval.bash
```

---

## 5. 所有 Python 命令 (直接运行, 非 sbatch)

### NaVILA

```bash
cd /share/home/u19666033/dhj/dped-vln
conda activate falcon  # 或 sk_streamvln for SFT
export PYTHONPATH=/share/home/u19666033/dhj/dped-vln:/share/home/u19666033/dhj/dped-vln/habitat-lab:/share/home/u19666033/dhj/dped-vln/habitat-baselines

# Zero-Shot v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True

# Zero-Shot v1 val_unseen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True

# Zero-Shot Static v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True

# Zero-Shot Static v1 val_unseen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_val_unseen.yaml habitat_baselines.evaluate=True

# Zero-Shot Static v1 test_unseen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/zero_shot_static/v1_test_unseen.yaml habitat_baselines.evaluate=True

# LoRA Training v1 (Habitat, 旧方法)
python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/lora/v1_train.yaml habitat_baselines.evaluate=False

# LoRA Eval v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navilla/lora/v1_val_seen.yaml habitat_baselines.evaluate=True

# SFT Training v2 (新方法, 教师强制)
/share/home/u19666033/.conda/envs/sk_streamvln/bin/python -u \
  habitat-baselines/habitat_baselines/rl/ddppo/policy/navilla_launcher.py \
  --model_name_or_path pretrained_model/navila_checkpoint \
  --data_mixture dped_v2_train --lora_enable True --lora_llm True \
  --num_train_epochs 1 --per_device_train_batch_size 1 --max_steps 20 \
  --output_dir /tmp/test

# SFT Training v2 6GPU
/share/home/u19666033/.conda/envs/sk_streamvln/bin/torchrun --nproc_per_node=6 \
  habitat-baselines/habitat_baselines/rl/ddppo/policy/navilla_launcher.py \
  --data_mixture dped_v2_train --lora_enable True --lora_llm True \
  --deepspeed ./scripts/zero3.json \
  --output_dir pretrained_model/navilla_lora_sft_dped_v2/NaVILA_SFT_DPed_v2
```

### StreamVLN

```bash
cd /share/home/u19666033/dhj/dped-vln
conda activate falcon  # 或 sk_streamvln for SFT
export PYTHONPATH=/share/home/u19666033/dhj/dped-vln:/share/home/u19666033/dhj/dped-vln/habitat-lab:/share/home/u19666033/dhj/dped-vln/habitat-baselines

# Zero-Shot v1 val_unseen
python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True

# Zero-Shot Static v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True

# LoRA Eval v1 val_unseen
python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/lora/v1_val_unseen.yaml habitat_baselines.evaluate=True

# LoRA Eval Static v1 val_unseen
python -u -m habitat_baselines.run --config-name=DPed_vlm/streamvln/lora/v1_val_unseen_static.yaml habitat_baselines.evaluate=True

# SFT Training v2 (streamvln_train.py)
cd habitat-baselines/habitat_baselines/rl/ddppo/policy/streamvln
/share/home/u19666033/.conda/envs/sk_streamvln/bin/python -u streamvln/streamvln_train.py \
  --model_name_or_path pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln \
  --data_path DPed_VLN/streamvln_training_data_v2/annotations.json \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 1 --lora_enable True --output_dir /tmp/streamvln_test
```

### NaVid

```bash
cd /share/home/u19666033/dhj/dped-vln
conda activate falcon
export PYTHONPATH=/share/home/u19666033/dhj/dped-vln:/share/home/u19666033/dhj/dped-vln/habitat-lab:/share/home/u19666033/dhj/dped-vln/habitat-baselines

# Zero-Shot v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_val_seen.yaml habitat_baselines.evaluate=True

# Zero-Shot v1 val_unseen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot/v1_val_unseen.yaml habitat_baselines.evaluate=True

# Zero-Shot Static v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/zero_shot_static/v1_val_seen.yaml habitat_baselines.evaluate=True

# LoRA Training v1
python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/lora/v1_train.yaml habitat_baselines.evaluate=False

# LoRA Eval v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/lora/v1_val_seen.yaml habitat_baselines.evaluate=True

# Human Static Eval v1 val_seen
python -u -m habitat_baselines.run --config-name=DPed_vlm/navid/human_static/v1_val_seen.yaml habitat_baselines.evaluate=True
```

---

## 6. 两周实验计划 (每日3-6 GPU)

### 优先级: 高(R1) > 中(R2) > 低(R3)

**原则:**
- 先用 1GPU 采样数据测试，确认 pipeline 通后再用完整数据
- 零样本实验优先 (不需要训练)
- 流式并行: 训练和评估可同时提交
- 每天可用 3-6 张卡，优先用 L40 (A800 留给 6GPU SFT 训练)

### Day 1-2: 数据准备 + Zero-Shot (部分已完成)

| 任务 | GPU | 预估 | 状态 |
|------|-----|------|------|
| NaviLa prep v1+v2 | 1×L40 | <1h | ⬜ |
| StreamVLN prep v1+v2 | 1×L40 | <1h | ⬜ |
| NaVid prep v1+v2 | 1×L40 | <1h | ⬜ |
| StreamVLN ZS v1 val_unseen | 1×L40 | ~3-6h | 🔄 (部分完成) |
| StreamVLN ZS Static v1 val_seen+val_unseen | 1×L40 | ~2-3h | 🔄 |
| StreamVLN ZS v2 val_unseen | 1×L40 | ~3-6h | 🔄 |
| StreamVLN ZSS v2 val_seen+val_unseen | 1×L40 | ~2-3h | 🔄 |

### Day 3-4: NaVILA Zero-Shot + NaVid Zero-Shot

| 任务 | GPU | 预估 | 状态 |
|------|-----|------|------|
| NaVILA ZS v1 val_seen+val_unseen | 1×L40 | ~4-8h | 🔄 (部分完成) |
| NaVILA ZSS v1 val_seen+val_unseen+test | 1×L40 | ~2-4h | 🔄 |
| NaVILA ZS v2 | 1×L40 | ~4-8h | ⬜ |
| NaVILA ZSS v2 | 1×L40 | ~2-4h | ⬜ |
| NaVid ZS v1 val_seen+val_unseen | 1×L40 | ~4-8h | ⬜ |
| NaVid ZSS v1 | 1×L40 | ~2-4h | ⬜ |

### Day 5-7: SFT 训练 (并行提交)

| 任务 | GPU | 预估 | 状态 |
|------|-----|------|------|
| NaVILA LoRA SFT v2 6GPU | 6×A800 | ~45min | ⬜ |
| NaVILA LoRA SFT v1 1GPU | 1×A800 | ~6h | ⬜ |
| StreamVLN LoRA SFT v2 1GPU | 1×A800 | ~8-10h | 🔄 (有续训) |
| StreamVLN LoRA SFT v1 1GPU | 1×A800 | ~8-10h | 🔄 (有续训) |
| NaVid LoRA Train v1 Habitat | 1×L40 | ~6-15h | ⬜ |

### Day 8-10: Fine-tune Eval (动态+静态)

| 任务 | GPU | 预估 | 状态 |
|------|-----|------|------|
| NaVILA LoRA Eval v1 val_seen+val_unseen | 1×L40 | ~4-8h | ⬜ |
| NaVILA LoRA Eval v2 val_seen+val_unseen | 1×L40 | ~4-8h | ⬜ |
| NaVILA Human Static Eval v1 | 1×L40 | ~2-4h | ⬜ |
| NaVILA Human Static Eval v2 | 1×L40 | ~2-4h | ⬜ |
| StreamVLN LoRA Eval all | 2×L40 | ~6-12h | ⬜ |
| StreamVLN Static Eval all | 1×L40 | ~2-4h | ⬜ |
| NaVid LoRA Eval v1 | 1×L40 | ~4-8h | ⬜ |
| NaVid Static Eval v1 | 1×L40 | ~2-4h | ⬜ |

### Day 11-14: 补充 + Test Eval + 收尾

| 任务 | GPU | 预估 | 状态 |
|------|-----|------|------|
| 所有模型 test_unseen 评估 | 3×L40 | ~1天 | ⬜ |
| 失败的实验重跑 | 按需 | - | ⬜ |
| 结果汇总 + 分析 | 0 | - | ⬜ |

---

## 7. 采样数据测试命令 (快速验证 pipeline)

```bash
# 使用 sampled_data 覆盖原数据路径的环境变量
# 或者直接修改 config 中的 data_path
# 测试命令 (以 NaVILA ZS v1 val_unseen 为例):
cd /share/home/u19666033/dhj/dped-vln
conda activate falcon
export PYTHONPATH=/share/home/u19666033/dhj/dped-vln:/share/home/u19666033/dhj/dped-vln/habitat-lab:/share/home/u19666033/dhj/dped-vln/habitat-baselines
python -u -m habitat_baselines.run \
  --config-name=DPed_vlm/navilla/zero_shot/v1_val_unseen.yaml \
  habitat_baselines.evaluate=True \
  habitat.dataset.data_path="/share/home/u19666033/dhj/dped-vln/DPed_VLN/sampled_data/v1/val_unseen/{scene}.json.gz"
```

---

## 8. 当前进度 & 已知问题

### 已完成的评估
- ⬜ (尚未有完全完成的评估，之前运行存在问题)

### 已知问题
1. **NaVILA ZS v1**: 在 val_seen 上 OOM, 可能需要减少 num_video_frames
2. **StreamVLN ZS**: LLM 输出解析失败 (LLM 输出 <|im_end|><|im_start|> 循环)
3. **旧版 Habitat 训练**: LoRA 不更新 VLM (torch.no_grad() 问题)
4. **SFT 训练已通过测试**: NaVILA 1GPU 20步成功 (loss 从 0.81 → 0.46)

---

## 9. 实测结果 (已完成的 val_unseen 评估)

| 模型 | 实验 | Episodes | SPL | Success |
|------|------|----------|-----|---------|
| NaVILA | ZS v1 | 675 | 0.369 | 0.376 |
| NaVILA | ZS v2 | 630 | 0.296 | 0.307 |
| NaVILA | ZSS v1 | 787 | 0.342 | 0.347 |
| NaVILA | ZSS v2 | 727 | 0.302 | 0.310 |
| StreamVLN | ZS v1 | 1000 | 0.257 | 0.291 |
| StreamVLN | ZS v2 | 1000 | 0.269 | 0.301 |
| StreamVLN | ZSS v1 | 1000 | 0.265 | 0.294 |
| StreamVLN | ZSS v2 | 1000 | 0.263 | 0.298 |

> 注: 
> - ZS = Zero-Shot (动态行人), ZSS = Zero-Shot Static (静态行人)
> - 观察: (1) v1 动态行人比 v2 效果更好 (不同数据集分布), (2) NaVILA 显著优于 StreamVLN
> - 所有 val_seen 和 test_unseen 评估尚未完成

---

## 10. 未完成的实验清单 (按优先级)

### Immediate (今天): 完成 Zero-Shot val_seen (补充现有结果)
1. NaVILA ZS v1 val_seen
2. NaVILA ZS v2 val_seen
3. StreamVLN ZS v1 val_seen
4. StreamVLN ZS v2 val_seen
5. NaVid ZS v1 val_seen+val_unseen (全新)
6. NaVid ZSS v1 val_seen+val_unseen (全新)

### Short-term (D3-7): 完成所有模型训练
7. NaVILA SFT v2 6GPU train
8. StreamVLN SFT v2 1GPU train (续训或从头)
9. NaVid LoRA v1 train (Habitat)

### Medium-term (D8-12): Fine-tune Evaluations
10-18. 各模型 LoRA eval / Human Static eval

### Final (D13-14): test_unseen + 收尾
19-27. 各模型 test_unseen eval

## 实验命令速查

每个实验都有对应的 sbatch 脚本在 `sbatch/DPed_vlm/` 下。
对于没有独立脚本的实验组合，运行方式如下：

### 缺失脚本的补充命令:

#### 1. NaVILA Zero-Shot v1 val_seen (补充)
```bash
python -u -m habitat_baselines.run \
  --config-name=DPed_vlm/navilla/zero_shot/v1_val_seen.yaml \
  habitat_baselines.evaluate=True
```

#### 2. NaVid Zero-Shot v1 val_seen
```bash
python -u -m habitat_baselines.run \
  --config-name=DPed_vlm/navid/zero_shot/v1_val_seen.yaml \
  habitat_baselines.evaluate=True
```

#### 3. NaVid Zero-Shot v1 val_unseen
```bash
python -u -m habitat_baselines.run \
  --config-name=DPed_vlm/navid/zero_shot/v1_val_unseen.yaml \
  habitat_baselines.evaluate=True
```

### test_unseen 评估 (各模型, 最终实验):
```bash
# NaVILA - ZS + ZSS + LoRA + Human Static
--config-name=DPed_vlm/navilla/zero_shot/v1_test_unseen.yaml
--config-name=DPed_vlm/navilla/zero_shot/v2_test_unseen.yaml
--config-name=DPed_vlm/navilla/zero_shot_static/v1_test_unseen.yaml
--config-name=DPed_vlm/navilla/zero_shot_static/v2_test_unseen.yaml
--config-name=DPed_vlm/navilla/lora/v1_test_unseen.yaml
--config-name=DPed_vlm/navilla/lora/v2_test_unseen.yaml
--config-name=DPed_vlm/navilla/human_static/v1_test_unseen.yaml
--config-name=DPed_vlm/navilla/human_static/v2_test_unseen.yaml

# StreamVLN - 同上模式
--config-name=DPed_vlm/streamvln/zero_shot/v1_test_unseen.yaml
# ... 等

# NaVid - v1 only
--config-name=DPed_vlm/navid/zero_shot/v1_test_unseen.yaml
--config-name=DPed_vlm/navid/zero_shot_static/v1_test_unseen.yaml
--config-name=DPed_vlm/navid/lora/v1_test_unseen.yaml
--config-name=DPed_vlm/navid/human_static/v1_test_unseen.yaml
```
