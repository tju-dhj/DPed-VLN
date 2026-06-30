import os
import subprocess

# 视频路径
video_path = "video_dir/scene=yr17PDCnDDW-episode=328_1-ckpt=58-distance_to_goal=2.95-distance_to_goal_reward=0.16-multi_agent_nav_reward=0.23-success=1.00-did_multi_agents_collide=0.00-num_steps=68.00-spl=1.00-psc=0.84-stl=1.00-human_collision=0.00.mp4"

# 输出目录
output_dir = "pic_dir/scene=yr17PDCnDDW-episode=328_1/"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 输出图片命名格式
output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

# ffmpeg 命令
# -r 2 表示每秒提取2帧
# -i 指定输入视频
# -q:v 2 设置输出图片质量（1-31，数值越小质量越高）
cmd = [
    "ffmpeg",
    "-i", video_path,
    "-r", "2",
    "-q:v", "2",
    output_pattern
]

# 执行命令
print("开始提取视频帧...")
try:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"视频帧提取完成！图片已保存至：{output_dir}")
except subprocess.CalledProcessError as e:
    print(f"ffmpeg 执行失败：{e.stderr}")
except FileNotFoundError:
    print("错误：未找到 ffmpeg，请确保已安装 ffmpeg")