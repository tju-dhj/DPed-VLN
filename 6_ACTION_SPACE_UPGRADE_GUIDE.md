# 6动作空间升级说明文档

## 📋 概述

本次升级将原有的4动作空间扩展为6动作空间，新增了**暂停（PAUSE）**和**后退（MOVE_BACKWARD）**两个动作。

## 🎯 动作空间映射

### 原有4动作空间
- **0**: STOP - 停止并终止任务
- **1**: MOVE_FORWARD - 前进
- **2**: TURN_LEFT - 左转
- **3**: TURN_RIGHT - 右转

### 新6动作空间
- **0**: STOP - 停止并终止任务（保持不变）
- **1**: MOVE_FORWARD - 前进（保持不变）
- **2**: TURN_LEFT - 左转（保持不变）
- **3**: TURN_RIGHT - 右转（保持不变）
- **4**: PAUSE - 暂停一步（新增，不移动但不终止任务）
- **5**: MOVE_BACKWARD - 后退（新增，与前进相反）

## 🔧 修改内容

### 1. 代码修改

#### falcon/additional_action.py
✅ **已包含所需动作类**（无需修改）
- `DiscreteStopAction` (line 38-56)
- `DiscretePauseAction` (line 59-76) - 暂停动作
- `DiscreteMoveForwardAction` (line 79-133)
- `DiscreteMoveBackwardAction` (line 135-190) - 后退动作
- `DiscreteTurnLeftAction` (line 193-208)
- `DiscreteTurnRightAction` (line 211-226)

**关键区别：**
- `DiscreteStopAction`: 设置 `is_stop_called=True` 和 `should_end=True`，终止任务
- `DiscretePauseAction`: 设置 `is_stop_called=False` 和 `should_end=False`，仅暂停不终止
- `DiscreteMoveBackwardAction`: 使用负线速度 `lin_vel = -self.lin_vel` 实现后退

#### habitat-lab/habitat/gym/gym_wrapper.py
✅ **已修改 `continuous_vector_action_to_hab_dict_v3` 函数** (line 218-249)

**修改内容：**
```python
# 动作列表从4个扩展到6个
agent_0_action_list = [
    'agent_0_discrete_stop',           # 0
    'agent_0_discrete_move_forward',   # 1
    'agent_0_discrete_turn_left',      # 2
    'agent_0_discrete_turn_right',     # 3
    'agent_0_discrete_pause',          # 4 (新增)
    'agent_0_discrete_move_backward'   # 5 (新增)
]
```

### 2. 配置文件

#### 新建配置文件

##### RL训练配置
**文件路径**: `habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train_v2_6actions.yaml`

**关键配置：**
```yaml
defaults:
  # 新增6个动作的配置
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_stop: discrete_stop
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_move_forward: discrete_move_forward
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_turn_left: discrete_turn_left
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_turn_right: discrete_turn_right
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_pause: discrete_pause
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_move_backward: discrete_move_backward

habitat:
  task:
    actions:
      agent_0_discrete_pause:
        lin_speed: 0.0
        ang_speed: 0.0
        allow_dyn_slide: False
      agent_0_discrete_move_backward:
        lin_speed: -25.0  # 负值表示后退
        ang_speed: 0.0
        allow_dyn_slide: True
```

##### IL验证配置
**文件路径**: `habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_il_val_v11_6actions.yaml`

（配置内容与训练配置类似，但用于评估）

## 🚀 使用方法

### 1. RL训练（6动作空间）

```bash
python -u -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train_v2_6actions
```

### 2. IL验证（6动作空间）

```bash
python -u -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_il_val_v11_6actions
```

### 3. 自定义配置

如需修改动作参数，在配置文件中调整：

```yaml
habitat:
  task:
    actions:
      agent_0_discrete_move_forward:
        lin_speed: 30.0  # 修改前进速度
      agent_0_discrete_move_backward:
        lin_speed: -20.0  # 修改后退速度
      agent_0_discrete_turn_left:
        ang_speed: 15.0  # 修改转向速度
```

## ⚙️ 技术细节

### 动作实现机制

1. **gym_wrapper.py 处理流程**:
   ```
   网络输出 (0-5) → continuous_vector_action_to_hab_dict_v3()
   → 转换为动作名称 → Habitat动作字典
   ```

2. **动作执行流程**:
   ```
   动作字典 → BaseVelAction.step() → 设置速度控制器
   → 更新agent位置
   ```

3. **暂停 vs 停止**:
   - **暂停（PAUSE）**: `is_stop_called=False`, `should_end=False`
     - 仅将速度设为0，agent保持在原位一步
     - 任务继续，不会触发episode结束

   - **停止（STOP）**: `is_stop_called=True`, `should_end=True`
     - 设置速度为0并标记任务结束
     - 触发episode终止逻辑

4. **后退实现**:
   - 使用负线速度: `lin_vel = -self.lin_vel`
   - 与前进相同的腿部动画（backward action也支持leg animation）
   - 通过 `allow_dyn_slide=True` 允许动态滑动碰撞处理

## 📊 兼容性说明

### 与4动作模型的兼容性

- **前4个动作（0-3）保持完全兼容**
- 使用4动作训练的模型**无法**直接用于6动作评估（输出维度不匹配）
- 需要重新训练模型以支持6动作空间

### 数据集兼容性

如果使用IL训练，需要确保：
1. 专家数据包含新动作（4和5）的标注
2. 或者仅在现有4动作数据上训练，新动作通过RL探索学习

## 🔍 验证测试

### 检查动作空间大小

在训练开始时，日志应显示：
```
Action space: Discrete(6)  # 而不是 Discrete(4)
```

### 测试各动作功能

可以通过修改配置临时禁用某些动作来测试：
```yaml
# 仅测试前进和后退
defaults:
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_move_forward: discrete_move_forward
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_move_backward: discrete_move_backward
```

## ⚠️ 注意事项

1. **DiscreteMoveBackwardAction 需要腿部动画文件**:
   ```
   data/robots/spot_data/spot_walking_trajectory.csv
   ```
   如果文件不存在，会报错：`AssertionError: Checkpoint not found`

2. **动作顺序很重要**:
   - gym_wrapper.py 中的 `agent_0_action_list` 顺序必须与网络输出对应
   - 不要随意调整列表顺序

3. **配置文件中的动作定义必须完整**:
   - 必须在 `defaults` 中声明所有6个动作
   - 必须在 `habitat.task.actions` 中配置所有6个动作的参数

4. **checkpoint 兼容性**:
   - 4动作checkpoint无法加载到6动作网络（输出层维度不匹配）
   - 建议为6动作训练单独创建checkpoint目录

## 📝 常见问题

### Q1: 如何确认6动作空间生效？
**A**: 查看训练日志中的 `Action space: Discrete(6)` 输出。

### Q2: 能否在4动作和6动作之间切换？
**A**: 可以，但需要：
- 使用不同的配置文件
- 使用不同的checkpoint
- 重新训练模型

### Q3: 暂停动作的应用场景？
**A**:
- 等待动态障碍物（行人）通过
- 在不确定情况下观察环境
- 避免错误动作导致的任务失败

### Q4: 后退动作的应用场景？
**A**:
- 从死胡同退出
- 避让突然出现的行人
- 纠正过度前进的错误

## 🎓 进阶使用

### 自定义动作权重（用于IL训练）

如果某些动作的expert示例较少，可以调整损失权重：
```python
# 在 direct_il_trainer.py 中
action_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])  # 增加pause和backward的权重
```

### 添加动作约束

可以通过修改 `additional_action.py` 添加动作约束：
```python
def step(self, *args, **kwargs):
    # 例：禁止连续后退超过5步
    if self.consecutive_backward_count > 5:
        return  # 跳过此动作
    self.consecutive_backward_count += 1
    # ... 原有逻辑
```

## 📚 相关文件索引

- 动作实现: `falcon/additional_action.py` (line 38-227, 984-1141)
- 动作映射: `habitat-lab/habitat/gym/gym_wrapper.py` (line 218-271)
- RL训练配置: `habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train_v2_6actions.yaml`
- IL验证配置: `habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_il_val_v11_6actions.yaml`

---

**最后更新**: 2026-03-31
**版本**: v2.0 (6-action space)
