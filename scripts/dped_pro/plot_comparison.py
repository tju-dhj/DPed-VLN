import matplotlib
matplotlib.use('Agg')  # 无头后端，适配服务器环境
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 1. 数据准备 =================
checkpoints = list(range(20, 61))  # ckpt-20 到 ckpt-60 (共41个)

dped_6actions = [
    0.3153, 0.3054, 0.3498, 0.2217, 0.2857, 0.3399, 0.3153, 0.3300, 0.3399, 0.3645,
    0.3153, 0.3005, 0.2956, 0.334975369, 0.2709, 0.335, 0.2857, 0.2956, 0.3793, 0.3399,
    0.3645, 0.3498, 0.3645, 0.3350, 0.3153, 0.3153, 0.3596, 0.3251, 0.3892, 0.3547,
    0.4089, 0.3941, 0.399, 0.3498, 0.3645, 0.3645, 0.4089, 0.3744, 0.3054, 0.3695, 0.4039
]

dped_eq_reward = [
    0.3596, 0.2906, 0.2562, 0.0739, 0.33, 0.3399, 0.3005, 0.3202, 0.3103, 0.2857,
    0.2217, 0.3645, 0.3547, 0.3153, 0.3892, 0.3153, 0.3645, 0.357, 0.3695, 0.3547,
    0.3695, 0.3941, 0.3645, 0.3005, 0.3941, 0.3842, 0.4039, 0.3498, 0.335, 0.3842,
    0.3645, 0.3892, 0.3793, 0.3695, 0.3300, 0.4286, 0.4089, 0.4236, 0.3941, 0.4039, 0.3941
]

# ================= 2. 统计参数计算 =================
def compute_stats(data):
    arr = np.array(data)
    return {
        'std': np.std(arr),
        'max': np.max(arr),
        'best_ckpt': checkpoints[np.argmax(arr)]
    }

stats_6a = compute_stats(dped_6actions)
stats_eq = compute_stats(dped_eq_reward)

# ================= 3. 绘图设置 =================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
ax.set_ylim(0.05, 0.55) 

# 背景阶段划分
ax.axvspan(20, 30, color='orange', alpha=0.15, zorder=0)
ax.text(25, 0.53, 'EARLY UNSTABLE PHASE', ha='center', va='bottom',
        fontsize=12, color='darkorange', weight='bold', zorder=1)

ax.axvspan(40, 60, color='green', alpha=0.15, zorder=0)
ax.text(50, 0.53, 'CONVERGENCE PHASE', ha='center', va='bottom',
        fontsize=12, color='darkgreen', weight='bold', zorder=1)

# ================= 4. 绘制折线 =================
ax.plot(checkpoints, dped_6actions, marker='o', markersize=4, linestyle='-', 
        linewidth=2, color='#1f77b4', label='dped_6actions', zorder=3)
ax.plot(checkpoints, dped_eq_reward, marker='s', markersize=4, linestyle='--', 
        linewidth=2, color='#9467bd', label='dped_eq_reward', zorder=3)

# ================= 5. ✅ 高亮最大峰值点（红色显著标记） =================
best_6a_idx = np.argmax(dped_6actions)
best_eq_idx = np.argmax(dped_eq_reward)

# 使用 scatter 单独绘制峰值点，确保覆盖在折线上方
ax.scatter(checkpoints[best_6a_idx], dped_6actions[best_6a_idx],
           color='red', s=130, zorder=4, edgecolors='white', linewidth=2, label='_nolegend_')
ax.scatter(checkpoints[best_eq_idx], dped_eq_reward[best_eq_idx],
           color='red', s=130, zorder=4, edgecolors='white', linewidth=2, label='_nolegend_')

# ================= 6. 标注峰值文字 =================
ax.annotate(f'Peak: {stats_6a["max"]:.4f}',
            xy=(checkpoints[best_6a_idx], dped_6actions[best_6a_idx]),
            xytext=(0, 14), textcoords='offset points', fontsize=11, 
            color='#1f77b4', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.0))

ax.annotate(f'Peak: {stats_eq["max"]:.4f}',
            xy=(checkpoints[best_eq_idx], dped_eq_reward[best_eq_idx]),
            xytext=(0, -17), textcoords='offset points', fontsize=11, 
            color='#9467bd', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.0))

# ================= 7. 添加精简统计面板 (顶部居中) =================
label_6a = f"ckpt-{stats_6a['best_ckpt']}"
label_eq = f"ckpt-{stats_eq['best_ckpt']}"

stats_text = (
    f"{'─'*30}\n"
    f"{'dped_6actions':>14} | {'dped_eq_reward':>14}\n"
    f"{'─'*30}\n"
    f"Std:  {stats_6a['std']:.4f}   |   {stats_eq['std']:.4f}\n"
    f"Max:  {stats_6a['max']:.4f}   |   {stats_eq['max']:.4f}"
)

ax.text(0.46, 0.96, stats_text,
        transform=ax.transAxes,
        fontsize=14,
        family='monospace',
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.95),
        zorder=5)

# ================= 8. 坐标轴与图例 =================
ax.set_xlabel('Checkpoint', fontsize=3, fontweight='bold')
ax.set_ylabel('Success Rate', fontsize=18, fontweight='bold')
ax.set_title('RL Validation: SR dped-6 (6actions vs eq_reward)', fontsize=18, fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
ax.set_xticks(range(20, 61, 5))
ax.set_xticklabels([f'ckpt-{c}' for c in range(20, 61, 5)])

plt.tight_layout()

# ================= 9. 双格式保存 =================
base_name = 'rl_validation_dped6_comparison'

png_file = f'{base_name}.png'
plt.savefig(png_file, dpi=300, bbox_inches='tight', format='png')
print(f"✅ PNG saved: {os.path.abspath(png_file)}")

pdf_file = f'{base_name}.pdf'
plt.savefig(pdf_file, bbox_inches='tight', format='pdf', metadata={'Creator': 'RL-Plot-Script'})
print(f"✅ PDF saved: {os.path.abspath(pdf_file)}")

plt.close()

# ================= 10. 终端打印 =================
print("\n📈 KEY METRICS (Terminal View):")
print(f"{'Metric':<10} {'dped_6actions':>14} {'dped_eq_reward':>14}")
print(f"{'─'*40}")
print(f"{'Std':<10} {stats_6a['std']:>14.4f} {stats_eq['std']:>14.4f}")
print(f"{'Max':<10} {stats_6a['max']:>14.4f} {stats_eq['max']:>14.4f}")
print(f"{'Best Ckpt':<10} {label_6a:>14} {label_eq:>14}")