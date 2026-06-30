# 6动作空间快速参考

## 动作映射表

| ID | 动作名称 | 英文名称 | 线速度 | 角速度 | 终止任务 | 说明 |
|----|---------|---------|-------|-------|---------|------|
| 0  | 停止    | STOP | 0.0 | 0.0 | ✓ | 停止并结束episode |
| 1  | 前进    | MOVE_FORWARD | 25.0 | 0.0 | ✗ | 向前移动 |
| 2  | 左转    | TURN_LEFT | 0.0 | 10.0 | ✗ | 原地左转 |
| 3  | 右转    | TURN_RIGHT | 0.0 | -10.0 | ✗ | 原地右转 |
| 4  | 暂停    | PAUSE | 0.0 | 0.0 | ✗ | 原地不动一步 |
| 5  | 后退    | MOVE_BACKWARD | -25.0 | 0.0 | ✗ | 向后移动 |

## 快速启动命令

### RL训练（6动作）
```bash
python -u -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train_v2_6actions
```

### IL验证（6动作）
```bash
python -u -m habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_il_val_v11_6actions
```

### 验证配置
```bash
python test_6action_space.py
```

## 核心修改文件

1. **habitat-lab/habitat/gym/gym_wrapper.py** (line 218-249)
   - 修改了 `continuous_vector_action_to_hab_dict_v3` 函数
   - 动作列表从4扩展到6

2. **falcon/additional_action.py** (已有代码)
   - `DiscretePauseAction` (line 59-76)
   - `DiscreteMoveBackwardAction` (line 135-190)

3. **配置文件** (新建)
   - `dynamic_vlnce_hm3d_train_v2_6actions.yaml` - RL训练
   - `dynamic_vlnce_hm3d_il_val_v11_6actions.yaml` - IL验证

## 关键代码片段

### gym_wrapper.py 动作映射
```python
agent_0_action_list = [
    'agent_0_discrete_stop',           # 0
    'agent_0_discrete_move_forward',   # 1
    'agent_0_discrete_turn_left',      # 2
    'agent_0_discrete_turn_right',     # 3
    'agent_0_discrete_pause',          # 4
    'agent_0_discrete_move_backward'   # 5
]
```

### 配置文件动作定义
```yaml
defaults:
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
      agent_0_discrete_move_backward:
        lin_speed: -25.0  # 负值 = 后退
        ang_speed: 0.0
```

## 常见问题

**Q: 4动作模型能否用于6动作评估？**
A: 不能，输出维度不匹配（4 vs 6），需要重新训练。

**Q: 暂停和停止的区别？**
A:
- 暂停(4): 仅停留一步，episode继续
- 停止(0): 终止episode

**Q: 如何验证6动作生效？**
A: 检查日志中 `Action space: Discrete(6)` 输出。

**Q: 需要修改数据集吗？**
A: IL训练需要包含动作4和5的expert数据；RL训练会自动探索。

## 性能建议

- **num_environments**: 建议4（避免OOM）
- **num_steps**: 建议64（平衡效率和稳定性）
- **checkpoint_interval**: 建议100（定期保存）

## 文件路径索引

- 使用文档: `/share/home/u19666033/dhj/DPed_pro/6_ACTION_SPACE_UPGRADE_GUIDE.md`
- 测试脚本: `/share/home/u19666033/dhj/DPed_pro/test_6action_space.py`
- RL配置: `habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train_v2_6actions.yaml`
- IL配置: `habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_il_val_v11_6actions.yaml`
