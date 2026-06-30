import matplotlib.pyplot as plt
import numpy as np

# 1. 提取数据 (严格参考论文表 IV )
models = ['NaVILA', 'StreamVLN']

# 成功率数据 (Success Rate)
static_sr = [23.02, 19.11] 
dynamic_sr = [21.61, 15.74]

# 安全率数据 (Safety Rate = 100 - HCR) 
# NaVILA: 100 - 32.46 = 67.54 (SPed); 100 - 40.17 = 59.83 (DPed)
# StreamVLN: 100 - 23.89 = 76.11 (SPed); 100 - 37.24 = 62.76 (DPed)
static_safe = [67.54, 76.11]
dynamic_safe = [59.83, 62.76]

# 2. 布局设置
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor='white')
x = np.arange(len(models))
width = 0.35

# --- 子图 1: Success Rate (SR) ---
# 使用 r'' 原始字符串防止 \u 转义错误
rects1 = ax1.bar(x - width/2, static_sr, width, label='Static (SPed)', 
                 color='#EEEEEE', edgecolor='#333333', linewidth=1)
rects2 = ax1.bar(x + width/2, dynamic_sr, width, label='Dynamic (DPed)', 
                 color='#004C99', edgecolor='#333333', linewidth=1)

ax1.set_ylabel(r'Success Rate (%) $\uparrow$', fontsize=11, fontweight='bold')
ax1.set_title('Success Rate (SR) Comparison', fontsize=12, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
ax1.set_ylim(0, 30)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
# 添加左图图例
ax1.legend(loc='upper right', frameon=False, fontsize=10)

# --- 子图 2: Safety Rate (100-HCR) ---
rects3 = ax2.bar(x - width/2, static_safe, width, label='Static (SPed)', 
                 color='#EEEEEE', edgecolor='#333333', linewidth=1)
rects4 = ax2.bar(x + width/2, dynamic_safe, width, label='Dynamic (DPed)', 
                 color='#D95319', edgecolor='#333333', linewidth=1)

ax2.set_ylabel(r'Safety Rate (%) $\uparrow$', fontsize=11, fontweight='bold')
ax2.set_title('Safety Rate (100-HCR) Comparison', fontsize=12, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
# 【关键修复】：添加右图图例
ax2.legend(loc='upper right', frameon=False, fontsize=10)

# 3. 数值标注函数
def add_labels(ax):
    for container in ax.containers:
        # 使用更稳健的标注方法
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold', fontsize=10)

add_labels(ax1)
add_labels(ax2)

# 4. 全局标题
plt.suptitle(r'Performance Degradation: Transition from Static (SPed) to Dynamic (DPed)', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
# 输出文件
plt.savefig('figure14_performance_degradation.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure14_performance_degradation.png', format='png', dpi=300, bbox_inches='tight')
print("Figure 14 saved successfully with legends on both subplots.")