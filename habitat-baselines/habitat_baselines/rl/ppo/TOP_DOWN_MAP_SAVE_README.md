# Top Down Map 单独保存功能说明

## 功能概述
在 `expert_data_collector_v3.py` 中新增了单独保存 top_down_map 图像的功能，每一帧的 top_down_map 都会被保存为独立的 PNG 文件。

## 新增函数

### `save_top_down_map_frame()`
```python
def save_top_down_map_frame(info: Dict, save_path: str, frame_idx: int, target_height: int = 480) -> bool
```

**功能**: 单独保存 top_down_map 图像到指定路径

**参数**:
- `info`: 信息字典（包含 top_down_map）
- `save_path`: top_down_map 保存的目录路径
- `frame_idx`: 当前帧的索引号（用于文件命名）
- `target_height`: 目标高度（用于调整地图大小，默认 480）

**返回值**: 
- `bool`: 是否成功保存（True/False）

**位置**: 第 167-200 行

## 保存路径结构

top_down_map 图像会保存在以下路径结构中：
```
{data_folder}/{split}/{scene_name}/{episode_id}/top_down_maps/
    ├── top_down_map_000000.png  # 第 0 帧
    ├── top_down_map_000001.png  # 第 1 帧
    ├── top_down_map_000002.png  # 第 2 帧
    └── ...
```

例如：
```
expert_data/train/apartment_0/episode_123/top_down_maps/
    ├── top_down_map_000000.png
    ├── top_down_map_000001.png
    └── ...
```

## 实现细节

### 1. 初始化阶段（第 2014-2030 行）
在每个 episode 开始时：
- 创建 `frame_indices` 字典：追踪每个环境的帧索引
- 创建 `top_down_map_paths` 字典：存储每个环境的 top_down_map 保存路径

### 2. 保存初始帧（第 2048-2051 行）
在视频初始帧生成后，调用 `save_top_down_map_frame()` 保存 top_down_map

### 3. 保存后续帧（第 2331-2334 行）
在每一步 step 后，如果生成了视频帧，同时调用 `save_top_down_map_frame()` 保存对应的 top_down_map

## 特性

1. **非侵入式设计**: 不修改原有的 `create_agent0_video_frame()` 函数
2. **独立保存**: top_down_map 单独保存为 PNG 文件，不影响视频生成
3. **自动创建目录**: 如果保存目录不存在，会自动创建
4. **异常处理**: 包含完善的异常处理，保存失败不会影响主流程
5. **帧索引命名**: 使用 6 位数字命名（000000-999999），便于排序和查找
6. **条件保存**: 只有当 info 中包含 top_down_map 时才会保存

## 使用说明

该功能在以下条件下自动启用：
1. `save_video_enabled = True`（即配置了 video_option）
2. 环境的 info 中包含 `top_down_map` 数据

不需要额外的配置，当视频功能启用时，top_down_map 会自动单独保存。

## 注意事项

1. top_down_map 保存与视频帧同步，每生成一帧视频就会尝试保存对应的 top_down_map
2. 如果 info 中没有 top_down_map 数据，会跳过保存（返回 False）
3. 保存失败会记录警告日志，但不会中断程序执行
4. 每个 episode 的 top_down_map 保存在独立的子目录中，便于管理

