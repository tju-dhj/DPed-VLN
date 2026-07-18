---
name: DPed-VLM DPed-v1-v2 finetune+eval
overview: 完成 DPed-VLN v1/v2 的 StreamVLN/NaVILA/NaVid 三模型微调 + 4 种 eval (zero_shot / zero_shot_static / finetune / finetune_static), 14 天内 (L40 池每天 4-5 卡)。
todos:
  - id: day1_prep_streamvln
    content: "Day 1 上午: 提交 streamvln_prep_data_v1/v2 (L40)"
    status: in_progress
  - id: day1_prep_navid
    content: "Day 1 下午: 新建 navid_prep_data_v1/v2_sampled.bash, 提交"
    status: pending
  - id: day2_smoke
    content: "Day 2 上午: StreamVLN LoRA v1 smoke test (max_steps=200)"
    status: pending
  - id: day2_eval_baseline
    content: "Day 2 下午: StreamVLN eval v1 val_unseen baseline"
    status: pending
  - id: day3_train
    content: "Day 3: StreamVLN LoRA v1 全量 + NaVILA SFT v1 smoke"
    status: pending
  - id: day4_train
    content: "Day 4: StreamVLN LoRA v2 + NaVILA SFT v1 全量 (并行)"
    status: pending
  - id: day5_train
    content: "Day 5: NaVILA SFT v2 + NaVid SFT v1 (并行)"
    status: pending
  - id: day6_eval
    content: "Day 6: NaVid SFT v2 + StreamVLN zero-shot eval (3 splits × 2 模型)"
    status: pending
  - id: day7_eval
    content: "Day 7: NaVILA zero-shot eval + StreamVLN finetune eval 启动"
    status: pending
  - id: day8_10_eval
    content: "Day 8-10: 24 个 finetune/finetune_static eval 任务 (4-5/天)"
    status: pending
  - id: day11_13_buffer
    content: "Day 11-13: buffer + 重跑失败任务 + 修复"
    status: pending
  - id: day14_summary
    content: "Day 14: 汇总所有 metric, 对比 finetune vs zero-shot + static vs action"
    status: pending
isProject: false
---

# DPed-VLN V1/V2 完整微调+评测计划

## 数据集完整性检查

抽样与未抽样都正常, scene 数 548, 但每个 scene 是同一 .json.gz 内多 episode — **抽样后每文件内 episode 数被截短**:
- `v1/` 28,345 episodes / 548 scenes → `v1_sampled_3000/` 3,000 / 548
- v1 与 v2 json.gz 文件各 episode 数大致一致 (StreamVLN 28,345 经 min_actions>=4 过滤后 27,522)

**采用方案: 训练用 `v_sampled_3000` (27522 中截到 3000, L40 上 1 卡 ≈30 分钟训完), eval 仍用全量 `v1/v2`** — 因为 LoRA 推理一致, 训练子样本仍能提升 performance。如果效果不达标可后续用全量重训。

---

## 三种 Finetune 方法对比 (按速度/稳定性排序)

| # | 方法 | 时间 (27522) | 时间 (3000 抽样) | 稳定性 | 备注 |
|---|------|------------|---------------|------|------|
| 1 | **navilla SFT (teacher forcing, llava train.py)** | 9-10h 1卡 / 2h ×8 卡 | **25-30 min ×1 卡** | 稳定 | 已验证 work (job 1761850) |
| 2 | streamvln LoRA (`streamvln_train.py`, llava-based) | ~4h ×1 A800 | ~30 min | 中 | L40 VRAM 够, 但 `dataloader_num_workers=2` l40 1卡足够 |
| 3 | navid LoRA (`convert_dped_vln_to_navid_sft.py` -> train) | ~4-8h 1L40 | ~30-50 min | 中 | 需要 gather `navid_sft_data_v{1,2}` (目前仅 v1) |
| 4 | ~~direct_il / habitat rollout (带 simulator)~~ | 24h+ | - | 不稳定 | 历史跑均失败 (malloc, OOM, LoRA probe无grad). **不推荐** |

**最优方法 (基于 w61/w61_navila 参考实现):**
- **NaVILA 用 SFT (teacher forcing llava)** — 已在 `slurm_logs/dped_vlm/navilla_sft_v1/1761850` 验证 1卡 ✓
- **StreamVLN 用 train.py streamvln_train** — L40 1卡可跑 (从 w61 StreamVLN 默认实现 + 当前 LoRA settings)
- **NaVid 同 NaVILA** — generate navid_sft_data 后 llava train.py

**核心改进 (相比历史失败的 habitat 方法):**
1. `scripts/convert_*.py` 预先把 episode 转 SFT 格式 (video/对话)
2. 使用 `torchrun` + `deepspeed zero2/zero3`, 非 habitat simulator rollout
3. LoRA on LLM only, vision_tower/mm_projector frozen
4. `gradient_checkpointing=True`, `bf16=True` 省 VRAM

---

## 总实验清单 (14 天 × L40 4-5/天)

| 模型 | 微调任务 | Eval 任务数 | 微调用时 |
|------|---------|--------|---------|
| StreamVLN v1 + v2 | 2 | 4 split × 2 mode × 2 ckpt = 16 eval/run | ~40 min ×2 |
| NaVILA v1 + v2 | 2 | 4 split × 2 mode × 2 ckpt = 16 eval/run | ~30 min ×2 |
| NaVid v1 + v2 | 2 | 4 split × 2 mode × 2 ckpt = 16 eval/run | ~50 min ×2 |

**总 GPU·时: 训练 ~6h + eval 96split × 12-20min ≈ 20h, 共 ≈ 26 GPU·时**

---

## 预备数据生成 (3 个脚本, 各约 10-15 分钟)

### 复用 + 新增 (混合策略)
抽样数据已存在, SFT conversations 已存在, **navid_sft_data 需重生成 v1/v2**

```bash
# StreamVLN/NaVILA 通用 (复用现有, 无需重做)
ls /share/home/u19666033/dhj/dped-vln/DPed_VLN/streamvln_training_data_v1/
# annotations.json 27,522 entries + navilla_conversations.json ✓
ls /share/home/u19666033/dhj/dped-vln/DPed_VLN/streamvln_training_data_v2/  ✓

# NaVid - 需补 v1 + v2 (--sampled 用 3000 抽样集做加速版)
# 新建 sbatch/DPed_vlm/<model>_prep_data_v<N>.bash
```

**新增脚本 (若缺):** `sbatch/DPed_vlm/navid_prep_data_v2_sampled.bash` 用 sampled_3000 数据集:
```bash
python scripts/convert_dped_vln_to_navid_sft.py \
  --data_root /share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/v2_sampled_3000/train \
  --rgb_roots /share/home/u19666033/dhj/dped-vln/data/collect_data/train \
              /share/home/u19666033/dhj/DPed_pro/data/collect_data/train \
  --output_dir /share/home/u19666033/dhj/dped-vln/DPed_VLN/navid_sft_data_v2_sampled \
  --action_sequence_length 4 --action_stride 4 --num_video_frames 4 \
  --video_format --dataset_tag v2_sampled --max_episodes 3000
```

---

## 14 天 (Day 1-14) 详细日计划

### Day 1-2: 数据准备 + 第一个端到端 smoke test
- **Day 1 上午**: 提交 `streamvln_prep_data_v1.bash`, `streamvln_prep_data_v2.bash` (L40, ~1h 各)
- **Day 1 下午**: 提交 `navid_prep_data_v1_sampled.bash` + v2 (新建)
- **Day 2 上午**: streamvln LoRA v1 (smoke test, max_steps=200) on 1张 L40 (~30min)
- **Day 2 下午**: streamvln eval v1 val_unseen (1张 L40, ~30min)

### Day 3-5: StreamVLN v1+v2 全量微调
- **Day 3**: StreamVLN LoRA v1 全量 (1 epoch, ~40min, 1 L40) + navilla SFT v1 端到端 (~30min 验证)  
- **Day 4**: StreamVLN LoRA v2 全量 + NaVILA SFT v1 全量 (2 jobs, 2 L40并行)
- **Day 5**: NaVILA SFT v2 + NaVid SFT v1 (2 jobs, 2 L40并行)

### Day 6-7: NaVid v2 + StreamVLN/NaVILA 8 个 splits eval
- **Day 6**: NaVid SFT v2 + StreamVLN zero_shot eval (3 splits × 2 模型, 6 jobs 并行)
- **Day 7**: NaVILA zero_shot eval + StreamVLN finetune eval 启动

### Day 8-10: 8 个 finetune eval × 3 模型 × 3 splits × 2 mode (Static/Action)
- 每天 4 job × ~15-20 min/eval ≈ 80min/天 → 18 个 eval × 4 (test_unseen/val_unseen/val_seen × static/action) = 24 eval tasks
- 每天 4-5 task, 5 天完成

### Day 11-13: Buffer + 复检 + 修坏重跑
- 排队 buffer
- Failed eval 重跑
- 对比 zero-shot vs finetune, frozen vs trainable LoRA

### Day 14: Final + writeup
- 汇总所有 metric JSON
- 对比 finetune vs zero-shot + static vs dynamic

---

## 训练指令模板 (sbatch 版本与 python 版本)

### A. StreamVLN LoRA (推荐: train.py, 来自 w61)
```bash
# Python 指令 (单卡 L40)
cd /share/home/u19666033/dhj/dped-vln/habitat-baselines/habitat_baselines/rl/ddppo/policy/streamvln
conda activate sk_streamvln
PYTHONPATH=/share/home/u19666033/dhj/dped-vln:/share/home/u19666033/dhj/dped-vln/habitat-lab:/share/home/u19666033/dhj/dped-vln/habitat-baselines:/share/home/u19666033/dhj/dped-vln/habitat-baselines/habitat_baselines/rl/ddppo/policy/streamvln \
python -u streamvln/streamvln_train.py \
  --model_name_or_path /share/home/u19666033/dhj/dped-vln/pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln \
  --version qwen_1_5 --vision_tower google/siglip-so400m-patch14-384 \
  --video_folder /share/home/u19666033/dhj/dped-vln/DPed_VLN/streamvln_training_data_v1 \
  --data_path   /share/home/u19666033/dhj/dped-vln/DPed_VLN/streamvln_training_data_v1/annotations.json \
  --num_history 8 --num_future_steps 4 --num_frames 32 \
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
  --bf16 True --lora_enable True --lora_r 16 --lora_alpha 32 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 1 --learning_rate 2e-5 \
  --gradient_checkpointing True --model_max_length 32768 \
  --output_dir /share/home/u19666033/dhj/dped-vln/pretrained_model/streamvln_lora_sft_dped_v1_sampled \
  --run_name StreamVLN_LoRA_DPed_v1_sampled
```

> **Sbatch 模板** 复用 `streamvln_lora_v1_llava_train.bash`, 改 L40 partition + 改 data_root + output_dir。

### B. NaVILA SFT (推荐: llava train_mem.py 教学强制)
参考脚本 `navilla_lora_sft_1gpu_test.bash`, 但去掉 `--max_steps 20`, 加 `--data_mixture dped_vN_train`:
```python
python -u habitat-baselines/habitat_baselines/rl/ddppo/policy/navilla_launcher.py \
    --model_name_or_path /share/home/u19666033/dhj/dped-vln/pretrained_model/navila_checkpoint \
    --version llama_3 --seed 42 \
    --data_path /share/home/u19666033/dhj/dped-vln/DPed_VLN/streamvln_training_data_v1/navilla_conversations.json \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch --mm_projector mlp_downsample \
    --num_video_frames 4 --mm_vision_select_layer -2 \
    --image_aspect_ratio resize --bf16 True \
    --output_dir /share/home/u19666033/dhj/dped-vln/pretrained_model/navilla_sft_dped_v1 \
    --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
    --do_eval False --save_strategy steps --save_steps 500 \
    --learning_rate 2e-5 --tf32 True --model_max_length 2048 \
    --gradient_checkpointing True --lazy_preprocess True \
    --report_to tensorboard --lora_enable True --lora_llm True \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

### C. NaVid SFT (类似 NaVILA + 自己的 SFT 数据)
```python
python -u habitat-baselines/habitat_baselines/rl/ddppo/policy/navila_launcher.py \
    --model_name_or_path /share/home/u19666033/dhj/dped-vln/pretrained_model/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split \
    --version navid --seed 42 \
    --data_path /share/home/u19666033/dhj/dped-vln/DPed_VLN/navid_sft_data_v1_sampled/navid_conversations.json \
    --vision_tower /share/home/u19666033/w61/NaVid-VLN-CE/model_zoo/eva_vit_g.pth \
    ...
```
> ⚠️ NaVid model_path 现在是符号链接 `navid-7b-full-224` → `navid_checkpoint`, 需检查 next_path 是否预期

### D. 评估 (4 种)
```bash
# Zero-Shot Static (最重要 baseline, 无需任何 LoRA)
python -u -m habitat_baselines.run --config-name=DPed_vlm/{model}/zero_shot_static/v{N}_{split}.yaml habitat_baselines.evaluate=True

# Zero-Shot Action
python -u -m habitat_baselines.run --config-name=DPed_vlm/{model}/zero_shot/v{N}_{split}.yaml habitat_baselines.evaluate=True

# Finetune Static (eval 后 LoRA 重新合并)
python -u -m habitat_baselines.run --config-name=DPed_vlm/{model}/human_static/v{N}_{split}.yaml habitat_baselines.evaluate=True

# Finetune Action (LoRA-merged)
python -u -m habitat_baselines.run --config-name=DPed_vlm/{model}/lora/v{N}_{split}.yaml habitat_baselines.evaluate=True
```

---

## 已存在可复用脚本 (直接 cp + 改名 + 改路径)

| 用途 | 已有脚本 | 操作 |
|------|--------|------|
| StreamVLN LoRA v1 训 | `streamvln_lora_v1_llava_train.bash` | 改 L40 partition + data_path → v1_sampled_3000 |
| StreamVLN LoRA v2 训 | `streamvln_lora_v2_llava_train.bash` | 同上 |
| NaVILA SFT v1/v2 训 | `navilla_lora_sft_1gpu_test.bash` | 改 output_dir, data_path = sampled |
| StreamVLN zero-shot static v1 | `streamvln_zero_shot_static_v1.bash` | 跑 val_seen/val_unseen/test_unseen |
| NaVILA LoRA eval v1 | `navilla_lora_v1_eval.bash` | 跑完 3 splits |
| StreamVLN finetune eval v1 | `streamvln_lora_v1_val_unseen.bash` | 跑完 3 splits |
| StreamVLN finetune static v1 | `streamvln_lora_v1_val_unseen_static.bash` | 跑完 3 splits |

---

## Slurm 配置参考 (L40 1 卡 baseline)

```bash
#!/bin/bash
#SBATCH --job-name=...
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --time=04:00:00        # 训练 30-50 分钟
#SBATCH --output=slurm_logs/dped_vlm/<job>/%j_%x.out
#SBATCH --error=slurm_logs/dped_vlm/<job>/%j_%x.err
#SBATCH --wckey=p19666033 -A p_p19666033
```

---

## 关键风险与缓解

| 风险 | 缓解 |
|------|------|
| L40 队列排队拥堵 | 每天 4-5 卡预留 buffer; 失败任务延后到 buffer day |
| 数据样本过少影响 quality | 抽样 3000→全量 28345, 1卡训练仅3h, 1天可补训 |
| NaVid 训练首次需验证 | Day 5 单卡 smoke test 500 steps 验证 |
| LoRA 合并后 eval 加载失败 | merge_lora_to_*.py 需在 merge 后 cp adapter 文件 |
| Out of memory | gradient_checkpointing=True + bf16 (历史已验证, 1 L40 足够 8B) |

---

## 验证/成功标准

- 每个模型 × level (3×2=6) 训练任务, 至少产出 1 个 LoRA adapter checkpoint
- 每个模型 × level × {zero_shot, zero_shot_static, finetune, finetune_static} = 24 个 eval 任务, 产出 metrics JSON
- 全部 6 个 LoRA 训练 + 24 个 eval 在 14 天内完成 = **每周约 3 个训练 + 12 个 eval**

