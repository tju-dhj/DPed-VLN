# DAgger分布式训练使用指南

## 概述

本指南介绍如何启动分布式DAgger训练，利用多个GPU加速训练过程。

## 前置条件

1. 已申请多个GPU资源（例如：2个A800 GPU）
2. 已配置好训练环境

## 方法一：使用torchrun启动（推荐）

### 步骤1: 申请GPU资源

```bash
srun -p A800 -n 1 --cpus-per-task=14 --gres=gpu:a800:2 --job-name=dhj --pty /bin/bash
```

### 步骤2: 启用分布式训练并运行

有两种方式启用分布式训练：

#### 方式A: 在命令行中启用（推荐）

```bash
cd /share/home/u14004/dhj/Falcon-main

# 使用torchrun启动分布式训练（自动检测2个GPU）
torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train \
    habitat_baselines.il.distributed.enabled=True
```

#### 方式B: 使用启动脚本

```bash
cd /share/home/u14004/dhj/Falcon-main

# 给脚本添加执行权限（首次使用）
chmod +x scripts/run_dagger_distributed.sh

# 运行脚本（脚本会自动检测GPU数量）
bash scripts/run_dagger_distributed.sh
```

### 步骤3: 验证分布式训练是否启动

训练开始后，你应该看到类似以下的日志：

```
Initialized distributed training: rank=0, world_size=2, local_rank=0
Initialized distributed training: rank=1, world_size=2, local_rank=1
Wrapped policy model with DistributedDataParallel
```

## 方法二：手动设置环境变量（适用于复杂场景）

如果需要更精细的控制，可以手动设置环境变量：

```bash
# 设置主节点地址和端口
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 设置每个进程的环境变量并启动
# 进程0
export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0
CUDA_VISIBLE_DEVICES=0 python -u -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train \
    habitat_baselines.il.distributed.enabled=True &

# 进程1
export RANK=1
export WORLD_SIZE=2
export LOCAL_RANK=1
CUDA_VISIBLE_DEVICES=1 python -u -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train \
    habitat_baselines.il.distributed.enabled=True &

wait  # 等待所有后台进程完成
```

## 配置说明

分布式训练的相关配置位于配置文件的 `habitat_baselines.il.distributed` 部分：

```yaml
il:
  distributed:
    enabled: True  # 启用分布式训练
    backend: "nccl"  # 使用NCCL后端（GPU）
    init_method: "env://"  # 从环境变量读取初始化信息
    world_size: -1  # -1表示从环境变量读取
    rank: -1  # -1表示从环境变量读取
    local_rank: -1  # -1表示从环境变量读取
```

## 重要注意事项

1. **数据收集阶段**：当前实现中，只有rank 0进程收集数据。如果需要并行收集数据，需要修改代码。

2. **Checkpoint保存**：只有rank 0进程保存checkpoint，其他进程会自动同步等待。

3. **日志输出**：只有rank 0进程输出训练日志和TensorBoard记录。

4. **内存使用**：每个GPU进程都需要加载完整的模型，确保显存足够。

5. **Batch Size**：分布式训练时，每个进程使用相同的batch_size，实际的总batch_size = batch_size × num_gpus。

6. **LMDB数据加载**：由于LMDB在多进程下可能有兼容性问题，`dataloader_num_workers` 建议保持为0。

## 故障排除

### 问题1: 端口被占用

如果遇到 `Address already in use` 错误，更改端口号：

```bash
torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \  # 更改端口号
    ...
```

### 问题2: NCCL初始化失败

检查NCCL环境：

```bash
# 检查NCCL版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 如果NCCL有问题，可以尝试使用gloo后端（仅用于调试，性能较差）
# 在配置文件中设置: backend: "gloo"
```

### 问题3: GPU显存不足

减少batch_size或使用梯度累积：

```bash
# 在配置文件中减小batch_size
habitat_baselines.il.batch_size=64  # 原来是128，改为64
```

## 性能优化建议

1. **调整num_environments**：分布式训练时，每个进程可以运行更少的环境并行收集数据
2. **使用梯度累积**：如果单个GPU显存不足，可以使用梯度累积来模拟更大的batch size
3. **调整学习率**：分布式训练时，总的有效batch size增大，可能需要相应调整学习率

## 示例：完整的SLURM作业脚本

如果你使用SLURM作业提交系统，可以创建以下作业脚本：

```bash
#!/bin/bash
#SBATCH --job-name=dagger_distributed
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=14
#SBATCH --ntasks=1

cd /share/home/u14004/dhj/Falcon-main

torchrun \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_dagger_train \
    habitat_baselines.il.distributed.enabled=True
```

然后使用 `sbatch script.sh` 提交作业。







