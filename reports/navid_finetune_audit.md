# NaVid 微调方式审计 + 接入 DPed-VLN 方案设计报告

> **生成日期**: 2026-07-04
> **目标**: 审计 NaVid 微调方式，对比 StreamVLN / NaVILA / w61 / sk 方法，给出 NaVid 接入 DPed-VLN 的推荐方案

---

## A. 本地是否找到 NaVid 代码

**已找到。** NaVid 代码存在于以下位置：

| 位置 | 内容 |
|------|------|
| `/share/home/u19666033/w61/NaVid-VLN-CE` | NaVid + Uni-NaVid 评估代码 (run.py, agent_navid.py, navid/ 模块) |
| `/share/home/u19666033/w61/Uni-NaVid` | Uni-NaVid 训练代码 (train.py, uninavid_stage_1.sh, uninavid_stage_2.sh) |
| `/share/home/u19666033/sk/NaVid-VLN-CE` | NaVid + Uni-NaVid 评估代码 (含 run_lvln.py 终身 VLN 变体) |
| `/share/home/u19666033/sk/Uni-NaVid` | Uni-NaVid 训练代码 |
| `/share/home/u19666033/qxy/NaVid` | NaVid 模型 checkpoint (navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split) |
| `/share/home/u19666033/wxy/Uni-NaVid` | Uni-NaVid 代码 |

**已有预训练权重**：
- EVA-ViT-G: `/share/home/u19666033/w61/NaVid-VLN-CE/model_zoo/eva_vit_g.pth` (~2GB)
- NaVid fine-tuned: `model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split/`
- Uni-NaVid fine-tuned: `model_zoo/uninavid-7b-full-224-video-fps-1-grid-2/`

---

## B. NaVid 代码路径

### 核心模块 (w61/NaVid-VLN-CE)

```
w61/NaVid-VLN-CE/
├── navid/
│   ├── constants.py              # 特殊 token 定义
│   ├── conversation.py           # 对话模板 (imgsp_v1)
│   ├── mm_utils.py               # tokenizer_image_token, KeywordsStoppingCriteria
│   ├── model/
│   │   ├── navid_arch.py         # NaVidMetaModel, NaVidMetaForCausalLM (核心 VLM)
│   │   ├── builder.py            # load_pretrained_model()
│   │   ├── language_model/
│   │   │   └── llava_navid.py    # LlavaLlamaAttForCausalLM (forward + generate)
│   │   ├── multimodal_encoder/
│   │   │   ├── builder.py        # build_vision_tower (仅 EVA-ViT)
│   │   │   └── eva_vit.py        # EVAVisionTowerLavis
│   │   └── multimodal_projector/
│   │       └── builder.py        # mlp2x_gelu projector
│   └── train/
│       ├── train.py              # 训练主脚本 (LazySupervisedDataset, train())
│       ├── train_mem.py           # FlashAttn monkey-patch 入口
│       ├── llava_trainer.py      # LLaVATrainer (继承 HF Trainer)
│       └── llama_flash_attn_monkey_patch.py
├── agent_navid.py                # NaVid 评估 agent
├── agent_uninavid.py             # Uni-NaVid 评估 agent
├── run.py                        # 评估主入口 (Habitat env loop)
└── eval_navid_vlnce.sh           # 评估脚本
```

### Uni-NaVid 训练代码 (w61/Uni-NaVid)

```
w61/Uni-NaVid/
├── uninavid/
│   ├── model/uninavid_arch.py    # 带 online token merging
│   ├── train/train.py            # 训练主脚本 (与 NaVid 结构相同)
│   └── train/train_mem.py
├── scripts/
│   ├── uninavid_stage_1.sh       # Stage 1: 从 Vicuna-7B 训练
│   ├── uninavid_stage_2.sh       # Stage 2: 从 Uni-NaVid 继续训练
│   ├── zero2.json                # DeepSpeed ZeRO-2
│   └── zero2_offload.json        # DeepSpeed ZeRO-2 + CPU offload
└── offline_eval_uninavid.py
```

---

## C. NaVid 训练入口

### 训练脚本

**训练入口**: `uninavid/train/train_mem.py` → `uninavid/train/train.py:train()`

**Stage 1 shell** (`uninavid_stage_1.sh`):
```bash
deepspeed --no_local_rank uninavid/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path vicuna-7b-v1.5 \
    --version imgsp_v1 \
    --data_path open_uninavid_sampled_500.json \
    --video_folder nav_videos/ \
    --vision_tower eva_vit_g.pth \
    --mm_projector_type mlp2x_gelu \
    --compress_type grid:2 \
    --tune_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lora_enable False \
    --group_by_modality_length True \
    --video_fps 1 \
    --logging_steps 1 \
    --save_steps 8000 \
    --output_dir uninavid-7b-full-224-video-fps-1-grid-2-from-vicuna
```

**关键参数**：
- `--lora_enable`: 支持 LoRA (r=64, alpha=16, dropout=0.05)
- `--freeze_backbone`: 冻结 LLM backbone
- `--tune_vision_encoder`: 是否训练 vision encoder
- `--tune_mm_mlp_adapter`: 仅训练 projector
- `--bits 4|8`: BitsAndBytes 量化
- `--gradient_checkpointing`: 支持
- `--deepspeed`: ZeRO-2 支持

### 评估入口

`run.py` + `eval_navid_vlnce.sh`:
- 通过 Habitat VLN-CE 环境循环
- 每步调用 `agent.act()` → `model.generate()` → `extract_result()` 解析动作
- Early stop: >25 连续旋转 或 >400 步
- 输出指标: distance_to_goal, success, spl, path_length

---

## D. NaVid 数据格式

### 训练 JSON 格式

```json
{
    "id": "NAV_ID_xxx",
    "video": "nav_videos/ep001.mp4",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nImagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: 'Walk to the couch...'. Analyze this series of images to decide your next move, which could involve turning left or right by a specific degree or moving forward a certain distance."
        },
        {
            "from": "gpt",
            "value": "forward forward left forward forward stop"
        }
    ]
}
```

### 数据特性

- `id` 包含 `NAV_ID` 触发导航数据特殊处理
- `video` 字段指向 mp4 视频文件，训练时用 `decord.VideoReader` 以 1 FPS 采样
- conversations 是标准 LLaVA 格式（human/gpt 对话）
- 仅 human tokens 被 mask (IGNORE_INDEX)，gpt tokens 用于 loss 计算
- 导航视频额外应用数据增强：随机帧丢弃（最多10%中间帧）、帧复制（3%概率）、color jitter

### 特殊 Token 组织

导航数据的 `<image>` placeholder 在 tokenization 时被替换为：
```
<video_special><image_sep><image></video_special><image_special></image_special>[Navigation]
```

---

## E. NaVid 动作格式

### 输出空间

| 动作 | 自然语言输出 | 映射 ID | Habitat 动作 |
|------|-------------|---------|-------------|
| STOP | `stop` | 0 | STOP |
| MOVE_FORWARD | `forward X` (X=距离cm) | 1 | MOVE_FORWARD (25cm/步) |
| TURN_LEFT | `left X` (X=角度) | 2 | TURN_LEFT (30°/步) |
| TURN_RIGHT | `right X` (X=角度) | 3 | TURN_RIGHT (30°/步) |

### Action Parser

位置: `agent_navid.py:extract_result()` 和 `agent_uninavid.py:extract_result()`

**NaVid** (单步模式): regex 匹配 `"stop"|r"forward\s*(-?\d+)"|r"left\s*(-?\d+)"|r"right\s*(-?\d+)"`，距离/25 转步数，角度/30 转步数，最多 3 步

**Uni-NaVid** (多步模式): 空格分割 `"forward left right stop"`，每个词映射到 action ID，最多 queue 2 步

### 动作队列

- NaVid: `pending_action_list`，单次推理最多 queue 3 步
- Uni-NaVid: `pending_action_list`，单次推理最多 queue 2 步（"to accelerate the inference"）

---

## F. NaVid 微调方式

### 训练类型：纯离线 SFT (Teacher-Forcing)

| 特性 | 详情 |
|------|------|
| **训练范式** | 离线 SFT，因果语言模型 teacher-forcing |
| **Loss** | CrossEntropyLoss，shifted logits vs shifted labels |
| **model.generate() 在训练中** | **不调用** — 仅使用 forward() 计算 loss |
| **env.step() 在训练中** | **不调用** — 完全离线 |
| **labels mask** | human 部分 + visual tokens + special tokens → IGNORE_INDEX=-100 |
| **LoRA** | 支持 (r=64, alpha=16, dropout=0.05, target=所有 Linear 除 vision/projector) |
| **Full Fine-tune** | 支持 (默认) |
| **DeepSpeed** | ZeRO-2，配置在 `scripts/zero2.json` |
| **Gradient Checkpointing** | 支持 (`--gradient_checkpointing True`) |
| **Flash Attention** | 支持 (train_mem.py monkey-patch LLaMA attention) |
| **BitsAndBytes** | 支持 4-bit/8-bit 量化训练 |
| **多阶段训练** | 两阶段: (1) 从 Vicuna-7B 训练 → (2) 从 Uni-NaVid base 继续训练 |
| **Vision encoder** | 默认冻结 (`--tune_vision_encoder False`) |
| **LLM backbone** | 默认全部训练，可选冻结 (`--freeze_backbone True`) |

### 视觉 Token 压缩策略

- 历史帧: `grid:2` 压缩 → **4 tokens/frame**
- 当前帧: 8×8 grid 压缩 → **64 tokens**
- K 帧历史 + 1 帧当前 = 64 + 4K visual tokens

### 数据增强（仅 Uni-NaVid 训练脚本）

- 随机帧丢弃（最多 10% 中间帧）
- 随机帧复制（3% 概率/帧）
- 随机 color jitter

---

## G. w61 目录微调方式总结

w61 是一个大型研究代码库，包含 50+ 项目。核心 VLN 微调方法：

### G.1 NaVid-VLN-CE: 仅评估，不做训练

- 提供 Habitat VLN-CE 评估循环
- Agent 实现: NaVid_Agent (逐步新编码所有历史帧)，UniNaVid_Agent (online token merging)
- eval 中每步调用 `model.generate()` 生成文本动作 → regex 解析 → env.step()
- **评估慢** (per-step inference on 7B model)

### G.2 Uni-NaVid: 训练代码

- 两阶段 SFT: Vicuna-7B → Uni-NaVid base → 继续训练
- 支持 LoRA / Full FT / DeepSpeed ZeRO-2
- Online token merging 用于 5Hz 推理

### G.3 w61_navila: NaVILA 评估框架

- 基于 LLaVA 的 NaVILA 评估
- `evaluation/run.py`, evaluation 环境配置
- dagger_trainer, ddppo_waypoint_trainer 等

### G.4 vln-llama-factory: LLaMA-Factory 微调

- `/share/home/u19666033/w61/vln-llama-factory/`
- 使用 LLaMA-Factory 框架进行 VLN SFT
- 支持 LoRA/QLoRA/Full FT/DPO/PPO
- VLN 数据转换脚本 (`merge_train_json.py`)

### G.5 SoM / Visual-RFT / NavComposer

- Set-of-Mark prompting、Visual RFT (强化微调)、NavComposer (动作组合)
- 使用 VLM (Qwen-VL 系列) 的导航方法

---

## H. sk 目录微调方式总结

sk 目录包含 50+ 项目，核心 VLN 微调框架：

### H.1 sk_streamvln: 最成熟的 VLN 微调框架

| 特性 | 详情 |
|------|------|
| **模型基座** | Qwen2-7B-Instruct |
| **Vision Encoder** | SigLIP So400M (google/siglip-so400m-patch14-384) |
| **Vision-Language Projector** | mlp2x_gelu |
| **Token 压缩** | 2D spatial pooling (stride=2) |
| **历史机制** | `<memory>` token + num_history frames |
| **多视图输入** | RGB + Depth + Pose + Intrinsics + time_ids |
| **Tunable Parts 系统** | `mm_tunable_parts` (mm_mlp_adapter, mm_vision_tower, mm_language_model, mm_lora_layer 等，逗号分隔) |
| **LoRA** | r=8~64，target: Qwen2 的 q/k/v/o/gate/up/down proj |
| **DeepSpeed** | ZeRO-2/3 |
| **多任务训练** | 支持 nav + QA + ScanQA + captioning 联合训练 (CombineDataset) |
| **数据增强** | ColorJitter, Posterize, Sharpness, AutoContrast |
| **torch.compile** | 支持 inductor backend |

**sk_streamvln 的关键创新 - `mm_tunable_parts` 系统**：
- 先 `model.requires_grad_(False)` 全部参数
- 再按 `mm_tunable_parts` 选择性启用 `requires_grad=True`
- 支持组合: `mm_vision_tower,mm_lora_layer` (vision 全量 + LLM LoRA)
- 这是最灵活的生产级 fine-tuning 控制方式

### H.2 InternNav: Qwen-VL 微调

- 模型: Qwen2_5_VL / Qwen2VL / Qwen3-VL
- 知识蒸馏: Qwen3-VL 32B (teacher) → Qwen3-VL 2B (student)
- LLaMA-Factory 集成

### H.3 JanusVLN / OpenVLA / LH-VLN

- 各种 VLN/VLA 微调方案

---

## I. 当前 StreamVLN 微调方式总结

### 模型架构

| 组件 | 详情 |
|------|------|
| **LLM** | Qwen2-7B-Instruct |
| **Vision Encoder** | SigLIP So400M (384×384) |
| **Projector** | mlp2x_gelu |
| **历史机制** | `<memory>` token, num_history=8 frames, KV cache 保持 past_key_values |
| **多模态输入** | RGB + Depth + Pose + Intrinsics + time_ids |

### 两条训练路径

#### Path A: LLaVA-style SFT (`streamvln_train.py`)

- 使用 `LoRASafeLLaVATrainer` → `LLaVATrainer` → HF `Trainer`
- **纯 teacher-forcing**，不调用 `generate()`
- LoRA: PEFT `get_peft_model()` 仅注入 Qwen2 LM 层
- `LoRASafeLLaVATrainer.compute_loss()` 有梯度安全检查 (验证 `loss.requires_grad`)，fallback recomputation
- 数据集: `VLNActionDataset`，interleaved conversation，num_future_steps=4 动作/轮
- 输出动作: `{'0': 'STOP', '1': '↑', '2': '←', '3': '→'}` (4 动作空间)

#### Path B: DirectIL Trainer (`direct_il_trainer.py`)

- 在 Habitat 仿真器内运行，每步调用 `policy.net()` forward
- **也是 teacher-forcing**，loss 为 CE(action_logits, gt_action)
- LoRA 单独注入

### 评估方式

- `StreamVLNEvaluator`: 调用 `model.generate()` 生成动作序列
- Action queue: 生成的后续动作入队，下次 step 直接出队无需推理
- KV cache (`past_key_values`) 在帧间复用
- 每 `num_frames=32` 步 reset memory

### 当前 SBatch 脚本

```
sbatch/DPed_vlm/streamvln_lora_v1_train.bash (1×L40)
sbatch/DPed_vlm/streamvln_lora_v2_train.bash (1×L40)
sbatch/DPed_vlm/streamvln_lora_v1_llava_train.bash
main_slurm_streamvln_lora_dped_pro.bash (2×L40, DeepSpeed ZeRO-2)
```

---

## J. 当前 NaVILA 微调方式总结

### 模型架构

| 组件 | 详情 |
|------|------|
| **LLM** | LLaMA-based (NaVILA checkpoint, LLaVA-1.6 风格) |
| **Vision Encoder** | SigLIP 或 CLIP (来自 NaVILA base) |
| **Projector** | Linear/MLP |
| **对话模板** | `llama_3` |

### 两条训练路径

#### Path A: Habitat IL Training (`direct_il_trainer.py`)

- **行为克隆 (Behavior Cloning)**，NOT teacher-forcing
- 训练中调用 `model.generate()` 生成文本动作
- Loss: CE(action_dist.logits, gt_action)
- LoRA: r=16, alpha=32, target: q/k/v/o/gate/up/down proj

#### Path B: 离线 SFT (`train.py` + `LLaVATrainer`)

- **纯 teacher-forcing**，不调用 `generate()`
- 预转换 JSON 对话数据
- LoRA: r=16, alpha=32, 仅注入 LLM 部分
- Action sequence mode: `action_sequence_length=4`

### Action Sequence Mode (离线 SFT 特有)

- 数据转换: `scripts/convert_dped_vln_to_navila_sft.py`
- 输出格式: `"move forward 25 cm; turn left 15 degrees; stop"` (分号分隔)
- Parser: `action_parser.parse_action_sequence()` (处理编号列表、分号分隔)
- Action queue: 第一个动作立即执行，其余入队

### 评估方式

- `NaVILAEvaluator`: 每环境维护独立 action queue
- 默认单步 generate + num_repeats queue (不用 action_sequence_mode)
- `use_cache=True` 用于加速

### 当前 SBatch 脚本

```
sbatch/DPed_vlm/navilla_lora_v1_train.bash (1×L40, 72h)
sbatch/DPed_vlm/navilla_lora_v2_train.bash
sbatch/DPed_vlm/navilla_human_static_v1_train.bash
sbatch/DPed_vlm/navilla_human_static_v2_train.bash
```

---

## K. 五者对比表

| 维度 | StreamVLN | NaVILA | NaVid | w61方法 | sk方法 |
|------|-----------|--------|-------|---------|--------|
| **模型基座** | Qwen2-7B-Instruct | LLaMA-based (NaVILA ckpt) | Vicuna-7B-v1.5 (Llama-7B) | Vicuna-7B / Qwen2-7B | Qwen2-7B / Qwen-VL |
| **Vision Encoder** | SigLIP So400M | SigLIP/CLIP | EVA-CLIP ViT-G | CLIP / EVA-ViT-G | SigLIP So400M |
| **输入模态** | RGB+D+Pose+Intrinsics | RGB video | RGB video (monocular) | RGB video | RGB+D+Pose+Intrinsics |
| **视觉 Token 压缩** | 2D pooling stride=2 | 标准 LLaVA | grid:2 (4t/历史帧) + 8×8 (64t/当前帧) | grid:2 | 2D pooling stride=2 |
| **历史视频** | `<memory>` token, KV cache | 标准视频 token | 所有历史帧 → `history_rgb_tensor` | 所有历史帧 / online merging | `<memory>` token |
| **Depth/Pose/Map** | ✅ 使用 | ❌ RGB only | ❌ RGB only (论文强调) | ❌ | ✅ |
| **输出动作类型** | 符号 + NL: STOP/↑/←/→ | NL: "move forward 25cm" | NL: "forward"/"left"/"stop" | NL 动作词 | 符号 + NL |
| **动作空间尺寸** | 4 (STOP, FWD, L, R) | 4 (同) | 4 (同) | 4 | 4+ |
| **One-step 还是 Sequence** | Sequence (4步) | 可选: 单步或4步 seq | 单步 (NaVid) / 多步 (Uni-NaVid) | 多步 | Sequence (4步) |
| **训练 Teacher-Forcing** | ✅ (两条路径都是) | Path A: ❌ (BC)  Path B: ✅ (SFT) | ✅ (纯 SFT) | ✅ (SFT) | ✅ |
| **训练中 generate()** | ❌ 不调用 | Path A: ✅ 调用  Path B: ❌ 不调用 | ❌ 不调用 | ❌ 不调用 | ❌ 不调用 |
| **LoRA** | ✅ r=16, LLM only | ✅ r=16, LLM only | ✅ r=64, all Linear except vision/proj | ✅ (LLaMA-Factory) | ✅ r=8~64, mm_lora_layer |
| **DeepSpeed** | ✅ ZeRO-2 | ✅ (HF Trainer) | ✅ ZeRO-2 | ✅ ZeRO-2/3 | ✅ ZeRO-2/3 |
| **单卡 A800 支持** | ✅ (1×L40 可跑，LoRA) | ✅ (1×L40) | ✅ (full FT 需多卡) | ✅ | ✅ |
| **tunable_parts 控制** | ❌ (LoRA 时 PEFT 管理) | ❌ | ❌ (freeze_backbone 单独开关) | ❌ | ✅ mm_tunable_parts |
| **Gradient Checkpointing** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Flash Attention** | ✅ (内置) | ✅ | ✅ (monkey-patch) | ✅ | ✅ |
| **数据格式** | annotations.json + 帧目录 | JSON conversations + video/PNG | JSON conversations + mp4 video | JSON conversations | annotations.json + 帧目录 |
| **Parser 复杂度** | 中 (regex + symbol matching) | 低 (分号分隔) | 低 (regex 关键词匹配) | 低 (空格分隔) | 中 (regex + symbol) |
| **Eval 速度** | 快 (action queue + KV cache) | 中 (action queue + repeats) | 慢 (每步重编码所有历史帧) | 慢 (Uni-NaVid online merging 加速到 5Hz) | 快 |
| **接入 DPed-VLN 难度** | 已接入 ✅ | 已接入 ✅ | 中-高 (需改写数据+parser+policy) | 高 (需从头搭建) | 已接入 ✅ |
| **预计训练速度** | 快 (teacher-forcing) | Path A 慢 / Path B 快 | 中 (视频编码开销) | 中 | 快 |
| **与 DPed-VLN 动作兼容性** | ✅ 相同 4 动作 | ✅ 相同 4 动作 | ✅ 相同 4 动作 | ✅ | ✅ |

---

## L. NaVid 接入 DPed-VLN 的推荐方案

### L.1 核心判断：NaVid 应该改成当前统一的离线 LoRA SFT 框架

**理由**：

1. **NaVid 原生就是 SFT teacher-forcing**（训练不调 generate），这与你的 NaVILA Path B（离线 SFT）和 StreamVLN LLaVA-style 路径一致
2. **NaVid 不支持在线 BC 训练**（没有 DAgger 或 RL），天然适合你的离线 SFT 模式
3. NaVid 的 LLaVA 架构与你现有的 NaVILA 代码几乎同源（都来自 haotian-liu/LLaVA），复用成本低
4. **不推荐直接用 NaVid 原始训练代码**：因为缺少 `mm_tunable_parts` 系统和你的 LoRA-Safe Trainer

### L.2 推荐接入路线：仿 NaVILA 离线 SFT 路径

```
NaVid 模型加载 → LoRA 注入 (LLM only) → DPed-VLN 数据转换 → 离线 SFT → Action Queue Eval
```

### L.3 数据格式差距

| 差距点 | NaVid 格式 | 你的 DPed-VLN 格式 | 解决方案 |
|--------|-----------|-------------------|---------|
| 视频格式 | mp4 文件 | PNG/GIF 帧序列 | 录制 mp4 或直接用帧列表 |
| 对话模板 | `imgsp_v1` (USER/ASSISTANT) | `llama_3` (NaVILA) / `qwen_1_5` (StreamVLN) | 切换到 NaVid 原生 `imgsp_v1` 或适配 |
| Action 文本 | `"forward forward left stop"` | `"move forward 25 cm; turn left 15 degrees; stop"` | 需对齐到 NaVid 的自然语言格式 |
| 特殊 Token | `<video_special>`, `[Navigation]` 等 6 种 | 标准 `<image>` | 数据转换脚本自动处理 |
| ID 前缀 | `NAV_ID_xxx` | `dped_train_ep...` | 加前缀即可 |

**差距评级**: 中等 — 数据格式接近但细节不同，需要专门的转换脚本

### L.4 多动作序列转换脚本能否复用？

**部分可复用**。`scripts/convert_dped_vln_to_navila_sft.py` 的核心逻辑（轨迹切分、多步窗口、动作文本化）可复用，但需修改：

1. 对话模板: `llama_3` → `imgsp_v1`
2. Action 文本映射: `"move forward 25 cm"` → `"forward"` (NaVid format)
3. 视频输出格式: GIF → mp4 (或直接用帧)
4. 特殊 token 注入: 需添加 `NAVIGATION_IDENTIFIER` 检测逻辑

**建议**: 写一个新的 `scripts/convert_dped_vln_to_navid_sft.py`，基于 NaVILA 版本修改

### L.5 NaVid 是否适合输出多步动作序列？

**推荐使用 4 步序列**，理由：

1. Uni-NaVid 原生支持多步 action 输出（`"forward forward left stop"`）
2. Eval 速度大幅提升（每 4 步推理一次 vs 每步推理）
3. 与 StreamVLN/NaVILA 的 `action_sequence_length=4` 对齐，便于公平对比
4. 不推荐 6 步：NaVid 未经此长度训练，可能不稳定

### L.6 NaVid 是否需要 action_queue？

**强烈建议实现 action_queue**，理由：

1. 原始 NaVid 评估每步都需 `model.generate()` (极其慢，重编码所有历史帧)
2. 如果输出 4 步序列 + action_queue，eval 速度可提升 ~4×
3. 应与 NaVILA evaluator 中的 action_queue 逻辑一致

### L.7 Eval 应支持 parse_action_sequence

**应该**，与 NaVILA evaluator 保持一致：
- `action_sequence_mode=True` 时使用 `parse_action_sequence()`
- 第一个动作立即执行，其余入 `action_queue`
- 动作格式适配 NaVid 的自然语言风格

---

## M. 需要新增/修改的文件列表

### 必须新增

| 文件 | 用途 | 基于 |
|------|------|------|
| `scripts/convert_dped_vln_to_navid_sft.py` | DPed-VLN → NaVid SFT 数据转换 | 修改 `convert_dped_vln_to_navila_sft.py` |
| `navid/action_parser.py` | NaVid 动作解析器 (单步+序列) | 参考 `agent_navid.py:extract_result()` |
| `navid_policy.py` | NaVid 的 Falcon Policy 封装 | 参考 `navila_policy.py` |
| `sbatch/DPed_vlm/navid_lora_seq_k4_train.bash` | NaVid LoRA SFT 训练脚本 | 参考 `navilla_lora_v1_train.bash` |
| `config/DPed_vlm/navid/lora/v1_sft_dped_seq_k4_train.yaml` | SFT 训练 config | 参考 NaVILA 对应 config |
| `config/DPed_vlm/navid/lora/v1_sft_dped_seq_k4_debug.yaml` | Debug 2k 样本 config |

### 可能需要新增

| 文件 | 用途 |
|------|------|
| `navid/navid_model.py` | NaVid 模型加载封装 (如需特殊处理) |
| `sbatch/DPed_vlm/navid_lora_v1_eval.bash` | NaVid LoRA 评估脚本 |
| `config/DPed_vlm/navid/lora/v1_val_seen.yaml` | 评估 config |
| `config/DPed_vlm/navid/lora/v1_val_unseen.yaml` | 评估 config |
| `config/DPed_vlm/navid/lora/v1_test_unseen.yaml` | 评估 config |
| `sbatch/DPed_vlm/navid_zero_shot_v1.bash` | 零样本评估脚本 |

### 可能需要修改

| 文件 | 修改内容 |
|------|---------|
| `habitat-baselines/habitat_baselines/rl/ppo/navila_evaluator.py` | 添加 NaVid 分支 (如 action parser 不同) |
| `habitat-baselines/habitat_baselines/rl/ddppo/policy/navila/llava/train/train.py` | 可能需要添加 NaVid 模型加载支持 |
| 相关 config YAML 文件 | 添加 NaVid 特定超参数 |

---

## N. 不建议做的事情

1. ❌ **不要直接用 NaVid 原始训练代码微调** — 缺少 LoRA-Safe Trainer、mm_tunable_parts 系统，且与你的实验框架不统一
2. ❌ **不要在训练中调用 model.generate()** — NaVid 原始代码已经避免，你也应该避免（除非是做 BC 对比实验）
3. ❌ **不要做 per-step generate 的 eval** — 极慢（每步需编码所有历史帧），必须用 action_sequence + action_queue
4. ❌ **不要用 NaVid 的 full fine-tune** — 7B 全量训练需要多张 A800，且与 StreamVLN/NaVILA 的 LoRA 设置不 fair
5. ❌ **不要尝试 6 步动作序列** — NaVid 未在此设置下训练，先稳定 4 步
6. ❌ **不要忽略 Uni-NaVid 的 online token merging** — 如果 eval 太慢，这是最直接的加速手段
7. ❌ **不要覆盖已有 StreamVLN/NaVILA 配置** — 所有 NaVid 相关文件放独立目录

---

## O. 下一步建议 Prompt

### 第一阶段：确认环境 + 数据准备

```
请帮我：
1. 检查 NaVid 预训练权重是否可正常加载：
   - EVA-ViT-G: /share/home/u19666033/w61/NaVid-VLN-CE/model_zoo/eva_vit_g.pth
   - NaVid 7B: /share/home/u19666033/w61/NaVid-VLN-CE/model_zoo/navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split/
2. 写 scripts/convert_dped_vln_to_navid_sft.py（基于 convert_dped_vln_to_navila_sft.py 修改）
3. 先转 2k debug 数据验证格式
4. 确认视频/帧格式能被 NaVid 的 LazySupervisedDataset 正确读取
```

### 第二阶段：训练 Debug

```
请帮我：
1. 写 navid/action_parser.py（支持 parse_action + parse_action_sequence）
2. 写 navid_policy.py（基于 navila_policy.py）
3. 写 sbatch/DPed_vlm/navid_lora_seq_k4_train.bash（debug 模式：2k 样本, 1 epoch, 1×L40）
4. 写相应 config YAML
5. 跑 2k debug 训练确认 loss 下降
6. 解决可能的 tokenizer/embedding 不匹配问题
```

### 第三阶段：Fast 5k → Full Train

```
请帮我：
1. 准备 5k 样本 fast 训练（验证收敛）
2. 准备 full train（全部 DPed-VLN 训练数据）
3. 配置 sbatch (1×L40, LoRA r=16, action_seq=4, ~1 epoch)
4. 启动训练并监控
```

### 第四阶段：Eval

```
请帮我：
1. 写 NaVid evaluator（在 navila_evaluator.py 中添加 NaVid 分支，复用 action_queue 逻辑）
2. 写 eval sbatch 脚本和 config
3. 跑 R2R val_seen / val_unseen / test_unseen
4. 跑 DPed-VLM val 评估
5. 收集指标 SR/SPL/NE/nDTW/SDTW
6. 对比 StreamVLN / NaVILA / NaVid 结果
```

### 如果 Eval 太慢的降级方案

```
如果 NaVid eval 每步 inference 超过 2 秒：
1. 优先用 Uni-NaVid 的 online token merging (5Hz on A100)
2. 增大 action_sequence_length 到 6（减少推理频次）
3. 减少视频历史帧数（num_video_frames=4 而非全量）
4. 量化到 4-bit (BitsAndBytes) 加速
5. 使用 vLLM 替代 HF generate
```

---

## 附录：训练慢风险分析

### NaVid 如果直接接入会不会出现以下问题？

| 风险 | 风险等级 | 说明 | 避免方式 |
|------|---------|------|---------|
| 每步 generate | 🔴 高 | 原始 NaVid eval 每步 generate，且重编码所有历史帧 | action_sequence_mode + action_queue eval |
| 训练时 online rollout | 🟢 低 | NaVid 原生不支持（无 env.step / DAgger） | 直接用离线 SFT |
| 一个样本只监督一个动作 | 🟡 中 | 原始 NaVid 单样本单步 | action_sequence_length=4, 每个样本监督 4 步 |
| 20k+ 样本训练太慢 | 🟡 中 | 视频编码 (EVA-ViT-G) 比单图编码慢 | 预提取 vision features, 减少视频帧数 |
| Eval 每步调用 7B generate | 🔴 高 | 这是最大瓶颈 | action_queue (evaluate 4 步才 generate 1 次) + Uni-NaVid online merging |

### 推荐的训练配置

```yaml
# 推荐 NaVid LoRA SFT 配置
lora_enable: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2e-5
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 1
model_max_length: 2048
video_fps: 1
compress_type: "grid:2"
bf16: true
gradient_checkpointing: true
deepspeed: "scripts/zero2.json"  # 如需
action_sequence_length: 4
action_stride: 4
num_video_frames: 8  # 限制历史帧数
freeze_vision_tower: true
```

### 分阶段数据量建议

```
Debug:  2,000 samples  → 验证数据流、loss 下降
Fast:   5,000 samples  → 验证收敛、各指标合理
Full:   all DPed-VLN   → 最终对比实验
```

---

> **报告结束**  
> 下一步: 请确认是否进入第一阶段 — 创建 `scripts/convert_dped_vln_to_navid_sft.py` + 2k debug 数据转换
