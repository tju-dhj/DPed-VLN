import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 准备数据 (根据表 II) [cite: 518, 519]
data = np.array([
    [16.72, 16.18],   # M_L1 (IL)
    [22.69, 22.37], # M_L2 (IL)
    [37.13, 36.26], # M_L1 (RL)
    [36.16, 38.55]  # M_L2 (RL)
])

rows = ['$M_{L1}$ (IL)', '$M_{L2}$ (IL)', '$M_{L1}$ (RL)', '$M_{L2}$ (RL)']
cols = ['Ins-L1 (Static)', 'Ins-L2 (Dynamic)']

# 2. 绘图设置 [cite: 502]
fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='white')
# 使用 Blues 调色盘，颜色从纯白开始 [cite: 465]
im = ax.imshow(data, cmap="Blues", aspect='auto', vmin=0, vmax=45)

# 3. 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.set_ylabel("Success Rate (%)", rotation=-90, va="bottom", fontsize=11)
cbar.outline.set_visible(False) 

# 4. 修改标注文字颜色：全部统一为黑色
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # 强制所有标注为黑色，确保在浅蓝底色上清晰可见 [cite: 411]
        color = "black"
        ax.text(j, i, f"{data[i, j]:.2f}",
                ha="center", va="center", color=color, 
                fontsize=13, fontweight='bold')

# 5. 设置坐标轴标签 [cite: 478]
ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols, fontsize=11, fontweight='bold')
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows, fontsize=11, fontweight='bold')

ax.set_xlabel('Evaluation Instruction Complexity', fontsize=12, labelpad=10)
ax.set_ylabel('Training Policy Strategy', fontsize=12, labelpad=10)
ax.set_title('Cross-Validation Matrix: Navigation Success Rate (%)', fontsize=14, pad=20, fontweight='bold')

# 6. 细化网格线与样式优化 [cite: 138]
ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=4)
ax.tick_params(which="minor", bottom=False, left=False)
for edge, spine in ax.spines.items():
    spine.set_visible(False)

plt.tight_layout()
# 同时保存为 PNG 和学术专用的 PDF 格式 [cite: 11]
plt.savefig('figure11_black_text.png', format='png', dpi=300)
plt.savefig('figure11_black_text.pdf', format='pdf', bbox_inches='tight')
plt.show()