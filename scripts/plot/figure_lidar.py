import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 准备数据 - 严格参考图片标注的数据
labels = ['SR (%)', 'SPL (%)', 'STL (%)', 'PSC (%)', 'Safety (%)']
num_vars = len(labels)

# 原始数据 (基于 Table III 的 Ins-L2 评估结果)
# Safety 计算方式为 100 - HCR
data_raw = {
    'Ours (RL)': [38.55, 36.83, 37.97, 92.88, 100 - 36.70], # Safety = 63.3
    'Ours (IL)': [22.37, 18.45, 20.42, 94.08, 100 - 30.62], # Safety = 69.38 (Peak)
    'NaVILA':    [23.45, 22.86, 23.35, 90.37, 100 - 43.32], # Safety = 56.68
    'StreamVLN': [15.96, 14.67, 15.07, 93.13, 100 - 41.59]  # Safety = 58.41
}

df = pd.DataFrame(data_raw, index=labels).T

# 2. 独立指标归一化 (Min-Max Normalization)
# 每一列的最小值映射为 0，最大值映射为 1
df_norm = (df - df.min()) / (df.max() - df.min())

# 3. 绘图设置
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # 闭合曲线

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='white')

# 学术配色方案与线条样式
colors = ['#d62728', '#ff8000', '#2ca02c', '#004c99']
line_styles = ['-', '-', '--', '--']

for i, (model_name, row) in enumerate(df_norm.iterrows()):
    values = row.values.tolist()
    values += values[:1] # 闭合数据
    ax.plot(angles, values, color=colors[i], linewidth=2.5, label=model_name, 
            linestyle=line_styles[i], marker='o', markersize=5, clip_on=False)
    # 仅为 Ours (RL) 添加极淡的填充
    if "RL" in model_name:
        ax.fill(angles, values, color=colors[i], alpha=0.03)

# 4. 坐标轴与视觉偏移 (添加中心偏移量，避免 0 值缩到圆心)
ax.set_theta_offset(np.pi / 2) # 顶部开始
ax.set_theta_direction(-1)     # 顺时针
ax.set_ylim(-0.2, 1.1)         # -0.2 产生中心圆孔偏移，1.1 留出边缘空间
ax.yaxis.set_ticks_position('none') 
ax.xaxis.set_ticks_position('none')
ax.tick_params(axis='both', which='both', length=0) # 彻底禁止刻度线显示
# 手动设置维度标签及 Peak 标注，并统一为黑色
custom_labels = [
    'SR (%)\nPeak: 38.55', 
    'SPL (%)\nPeak: 36.83', 
    'STL (%)\nPeak: 37.97', 
    'PSC (%)\nPeak: 94.08', 
    'Safety (100-HCR)\nPeak: 69.38'
]
# 确保所有文字（包括 Peak 数值）为黑色加粗
ax.set_thetagrids(np.degrees(angles[:-1]), custom_labels, fontsize=11, fontweight='bold', color='black')

# 5. 【核心修改】：加深雷达图的圈（网格线）
# 设置 rgrids 刻度
ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.2', '0.4', '0.6', '0.8', '1.0'], 
             color='#666666', size=10) # 加深刻度文字颜色

# 去掉背景边框
ax.spines['polar'].set_visible(False)

# 加深网格线：颜色改为中等灰色 (#999999)，样式改为实线 ('-')
ax.grid(True, color='#999999', linestyle='-', linewidth=1.0, alpha=0.8)

# 6. 图例与保存
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=10)
plt.title('Relative Performance Advantage: Normalized Multi-dimensional Comparison', 
          fontsize=14, fontweight='bold', pad=40, color='black')

plt.tight_layout()
# 输出为高质量 PDF (学术矢量图) 和 PNG
plt.savefig('figure12_radar_deep_circles.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure12_radar_deep_circles.png', format='png', dpi=300, bbox_inches='tight')
plt.show()