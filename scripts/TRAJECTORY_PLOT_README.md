# 轨迹长度分布可视化脚本

## 📊 功能说明

独立脚本，用于生成高质量的轨迹长度（步数）分布直方图，专门针对学术论文使用。

## ✨ 核心特性

### 1. 数据过滤
- ✅ 自动过滤超过指定步数的轨迹（默认 500 步）
- ✅ 支持自定义最大步数阈值
- ✅ 显示过滤前后的数据统计

### 2. 布局优化（解决重叠问题）
- ✅ 统计文本框：右上角，有充足边距
- ✅ 图例：左上角，与文本框分离
- ✅ Y轴范围：自动增加 15% 上边距，确保文本框不被遮挡
- ✅ 使用 `zorder` 确保文本层级正确

### 3. 视觉效果
- ✅ Times New Roman 字体
- ✅ 300 DPI 高分辨率 PDF
- ✅ 均值/中位数线清晰标注
- ✅ 专业配色和网格

## 🚀 快速使用

### 方法1: 默认参数
```bash
cd /share/home/u14004/dhj/Falcon-main
python scripts/plot_trajectory_distribution.py
```

**默认设置**:
- 输入: `dataset_analysis_academic/all_episodes.csv`
- 输出: `trajectory_length_distribution.pdf`
- 最大步数: 500

### 方法2: 自定义参数
```bash
python scripts/plot_trajectory_distribution.py \
    --csv dataset_analysis_academic/all_episodes.csv \
    --output my_trajectory_plot.pdf \
    --max_steps 400
```

### 方法3: 使用快捷脚本
```bash
bash scripts/run_trajectory_plot.sh
```

## 📋 输入数据格式

CSV 文件需要包含以下列：

| 列名 | 类型 | 说明 | 必需 |
|------|------|------|------|
| `num_steps` | int | 每个 episode 的步数 | ✅ 必需 |
| `split` | str | 数据集划分 ('train' 或 'val') | ⚠️ 可选 |

**示例数据**:
```csv
scene,episode_id,num_steps,split
scene1.basis,ep001,245,train
scene1.basis,ep002,389,train
scene2.basis,ep003,156,val
```

## 📊 输出说明

### 图表布局

```
┌─────────────────────────────────────────────────────────────────┐
│  Trajectory Length Distribution Across Train and Validation Sets │
├──────────────────────────┬──────────────────────────────────────┤
│ Training Set             │ Validation Set                       │
│                          │                                      │
│ Legend (upper left)      │ Legend (upper left)                  │
│ - Mean                   │ - Mean                               │
│ - Median                 │ - Median                             │
│                          │                          Stats Box   │
│    [Histogram bars]      │    [Histogram bars]      (upper right)│
│                          │                          Episodes: X  │
│                          │                          Std: X.X     │
│                          │                          Min: X       │
│ X-axis: Steps (0-500)    │ X-axis: Steps (0-500)    Max: X       │
└──────────────────────────┴──────────────────────────────────────┘
```

### 终端输出示例

```
================================================================================
Generating Trajectory Length Distribution
================================================================================

Loading data from: dataset_analysis_academic/all_episodes.csv
  Total episodes loaded: 33073
  Training episodes: 32172
  Validation episodes: 901

After filtering (≤500 steps):
  Training episodes: 31856 (98.9%)
  Validation episodes: 893 (99.1%)

Training Set Statistics (≤500 steps):
  Episodes: 31,856
  Mean: 187.45 steps
  Median: 175.00 steps
  Std: 89.32 steps
  Range: [10, 500] steps

Validation Set Statistics (≤500 steps):
  Episodes: 893
  Mean: 189.23 steps
  Median: 177.00 steps
  Std: 91.15 steps
  Range: [12, 500] steps

✓ Saved: trajectory_length_distribution.pdf
  File size: 45.3 KB

================================================================================
Trajectory Length Distribution Generation Complete!
================================================================================
```

## 🎨 布局优化细节

### 问题：图注重叠
**原因**: 统计文本框和图例可能与直方图柱状重叠

**解决方案**:
1. **统计框位置**: `(0.98, 0.98)` 右上角，远离数据区
2. **图例位置**: `loc='upper left'` 左上角，与统计框对角
3. **Y轴扩展**: `ylim = max_height * 1.15`，顶部留出 15% 空间
4. **层级控制**: 文本框 `zorder=100`，确保在最上层
5. **透明度**: 统计框 `alpha=0.85`，图例 `framealpha=0.92`

### 代码关键部分
```python
# 统计框：右上角
ax.text(0.98, 0.98, stats_text, 
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        zorder=100)  # 确保在最上层

# 图例：左上角
ax.legend(loc='upper left', framealpha=0.92)

# Y轴留出空间
ax.set_ylim(0, max_hist_height * 1.15)  # 15% 上边距
```

## 📝 参数详解

### `--csv` (输入文件)
- **类型**: 字符串
- **默认**: `dataset_analysis_academic/all_episodes.csv`
- **说明**: 包含 episode 数据的 CSV 文件路径

### `--output` (输出文件)
- **类型**: 字符串
- **默认**: `trajectory_length_distribution.pdf`
- **说明**: 输出 PDF 文件路径

### `--max_steps` (最大步数)
- **类型**: 整数
- **默认**: `500`
- **说明**: 只显示步数 ≤ 此值的轨迹
- **示例**: `--max_steps 400` 只显示 ≤400 步的轨迹

## 🔍 常见问题

### Q1: 为什么要过滤 >500 步的轨迹？
**A**: 
- 超长轨迹是异常值（outliers）
- 压缩主要数据分布的可视化空间
- 500 步以内覆盖 98%+ 的数据

### Q2: 如何只显示训练集？
**A**: 修改 CSV，只保留 `split == 'train'` 的行，或使用 pandas 预处理：
```python
import pandas as pd
df = pd.read_csv('all_episodes.csv')
df_train = df[df['split'] == 'train']
df_train.to_csv('train_only.csv', index=False)
```

### Q3: 图注仍然重叠怎么办？
**A**: 调整位置参数：
```python
# 在代码中修改
ax.text(0.99, 0.99, ...)  # 更靠右上
ax.set_ylim(0, max_hist_height * 1.2)  # 增加到20%边距
```

### Q4: 如何修改直方图的柱数？
**A**: 在代码第 140 行修改 `bins` 参数：
```python
n, bins, patches = ax.hist(steps, bins=40, ...)  # 改为40个柱
```

## 📦 依赖包

```bash
pip install pandas numpy matplotlib seaborn
```

**版本要求**:
- Python >= 3.7
- pandas >= 1.2.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## ✅ 质量检查清单

生成 PDF 后检查：

- [ ] 训练集图表完整显示
- [ ] 验证集图表完整显示
- [ ] 统计框在右上角，**不与柱状图重叠**
- [ ] 图例在左上角，**不与统计框重叠**
- [ ] 均值线和中位数线清晰可见
- [ ] X 轴范围为 0-500
- [ ] 字体为 Times New Roman
- [ ] PDF 分辨率清晰（300 DPI）

## 🎯 与原脚本对比

| 特性 | 原脚本 (`analyze_dataset_academic.py`) | 新脚本 (`plot_trajectory_distribution.py`) |
|------|----------------------------------------|-------------------------------------------|
| **功能** | 数据分析 + 多种可视化 | **仅轨迹分布图** |
| **过滤** | 无过滤 | ✅ **过滤 >500 步** |
| **布局** | 基础布局 | ✅ **优化避免重叠** |
| **独立性** | 需要完整分析流程 | ✅ **单独运行** |
| **定制性** | 固定参数 | ✅ **可配置参数** |
| **代码行数** | 529 行 | **298 行** |

---

**创建时间**: 2026-01-09  
**版本**: v1.0  
**状态**: ✅ 生产就绪

