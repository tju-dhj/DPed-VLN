# DAgger到RL训练过渡指南

## 概述

这个文档说明如何从DAgger（IL）训练切换到RL训练，同时保持两种训练方法可以独立运行。

## 文件说明

### 1. 配置文件

- **`dynamic_vlnce_hm3d_dagger_train.yaml`**: DAgger（IL）训练配置
- **`dynamic_vlnce_hm3d_train.yaml`**: 纯RL训练配置
- **`dynamic_vlnce_hm3d_dagger_to_rl_train.yaml`**: 从DAgger checkpoint加载并继续RL训练的配置（新增）

### 2. 代码修改

- **`habitat_baselines/rl/ppo/dynamic_vln_trainer.py`**: 添加了从IL checkpoint加载的支持

## 使用方法

### 步骤1: 使用DAgger训练

首先使用DAgger训练到一定阶段：

```bash
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train.yaml
```

训练完成后，checkpoint会保存在：
```
evaluation-vln/dynamic_vlnce_clip_dagger/hm3d/checkpoints/
```

### 步骤2: 配置IL checkpoint路径

编辑 `dynamic_vlnce_hm3d_dagger_to_rl_train.yaml`，设置IL checkpoint路径：

```yaml
habitat_baselines:
  load_from_il_checkpoint: True  # 启用从IL checkpoint加载
  il_checkpoint_path: "evaluation-vln/dynamic_vlnce_clip_dagger/hm3d/checkpoints/latest.pth"  # 或指定具体的checkpoint文件
```

如果不设置 `il_checkpoint_path`，系统会尝试从默认路径查找：
```
evaluation-vln/dynamic_vlnce_clip_dagger/hm3d/checkpoints/latest.pth
```

### 步骤3: 使用RL训练继续训练

使用新的配置文件启动RL训练：

```bash
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_to_rl_train.yaml
```

## 配置说明

### 关键配置项

1. **`load_from_il_checkpoint`**: 是否从IL checkpoint加载（默认：False）
2. **`il_checkpoint_path`**: IL checkpoint的完整路径（可选，如果不设置会尝试默认路径）
3. **`rl.ddppo.reset_critic`**: 是否重置critic层（默认：True，因为IL没有critic）
4. **`rl.ddppo.train_encoder`**: 是否继续训练编码器（默认：True）

### 模型架构兼容性

确保以下配置在DAgger和RL训练中保持一致：

- `backbone`: 必须相同（例如：`resnet50_clip_attnpool`）
- `rnn_type`: 必须相同（例如：`LSTM`）
- `num_recurrent_layers`: 必须相同（例如：`2`）
- `hidden_size`: 必须相同（例如：`512`）

## 工作原理

### Checkpoint格式

- **IL checkpoint**: 直接保存policy的`state_dict`（只包含actor权重）
- **RL checkpoint**: 保存agent的完整状态（包含actor和critic）

### 加载过程

1. RL训练器初始化时，检查`load_from_il_checkpoint`配置
2. 如果启用，加载IL checkpoint
3. 将IL的policy权重映射到RL的actor部分
4. Critic部分使用随机初始化（因为IL没有critic）
5. 继续使用RL方法训练

### 键名映射

IL checkpoint的键名格式可能与RL训练器期望的格式不同，代码会自动处理以下映射：

- `net.xxx` → `agents.0.actor_critic.policy.net.xxx`
- `policy.net.xxx` → `agents.0.actor_critic.policy.net.xxx`
- 等等

## 独立训练

### 纯DAgger训练

使用原始配置文件，不受影响：

```bash
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train.yaml
```

### 纯RL训练

使用原始配置文件，不受影响：

```bash
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train.yaml
```

## 注意事项

1. **学习率**: 从预训练模型开始，建议使用较小的学习率（已在配置中设置）
2. **Critic初始化**: Critic层会随机初始化，可能需要一些时间才能收敛
3. **键名匹配**: 如果遇到键名不匹配的警告，检查模型架构是否一致
4. **Checkpoint路径**: 确保IL checkpoint路径正确，否则会使用随机初始化的权重

## 故障排除

### 问题1: 找不到IL checkpoint

**症状**: 日志显示"IL checkpoint path specified but file does not exist"

**解决**: 
- 检查`il_checkpoint_path`配置是否正确
- 确认DAgger训练已完成并保存了checkpoint
- 检查文件路径权限

### 问题2: 键名不匹配

**症状**: 大量"Missing keys"或"Unexpected keys"警告

**解决**:
- 检查模型架构配置是否一致（backbone, rnn_type等）
- 查看日志中的具体键名，确认格式是否正确
- 如果只是critic相关的键缺失，这是正常的（IL没有critic）

### 问题3: 训练不稳定

**症状**: 训练初期loss很大或性能下降

**解决**:
- 这是正常的，因为critic需要从头学习
- 可以降低学习率
- 可以先用较小的步数进行warm-up

## 示例工作流

```bash
# 1. DAgger训练（例如训练5个迭代）
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train.yaml

# 2. 检查checkpoint
ls evaluation-vln/dynamic_vlnce_clip_dagger/hm3d/checkpoints/

# 3. 编辑配置文件，设置checkpoint路径
# 编辑 dynamic_vlnce_hm3d_dagger_to_rl_train.yaml

# 4. 继续RL训练
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_to_rl_train.yaml
```

## 总结

这个实现允许你：
- ✅ 从DAgger训练无缝切换到RL训练
- ✅ 保持DAgger和RL训练器独立工作
- ✅ 灵活配置checkpoint路径
- ✅ 自动处理键名映射和格式转换

如有问题，请检查日志中的详细错误信息。



