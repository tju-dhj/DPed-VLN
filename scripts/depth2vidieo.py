import cv2
import numpy as np
import os
import subprocess
import glob

def analyze_depth_ranges(folder_path):
    """
    分析所有深度图像的全局深度范围
    """
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[0]))
    
    all_min_vals = []
    all_max_vals = []
    
    print("分析深度图像范围...")
    for filename in png_files:
        img_path = os.path.join(folder_path, filename)
        depth_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if depth_img is not None:
            min_val = np.min(depth_img)
            max_val = np.max(depth_img)
            all_min_vals.append(min_val)
            all_max_vals.append(max_val)
            print(f"  {filename}: min={min_val}, max={max_val}")
    
    if all_min_vals and all_max_vals:
        global_min = min(all_min_vals)
        global_max = max(all_max_vals)
        print(f"\n全局深度范围: min={global_min}, max={global_max}")
        return global_min, global_max
    else:
        return None, None

def process_images_with_consistent_mapping(folder_path, output_folder, colormap=cv2.COLORMAP_JET):
    """
    处理图像，使用全局一致的深度范围
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 分析全局深度范围
    global_min, global_max = analyze_depth_ranges(folder_path)
    if global_min is None or global_max is None:
        print("无法分析深度范围")
        return []
    
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[0]))
    
    # 获取目标尺寸（使用第一个图像的尺寸）
    first_img_path = os.path.join(folder_path, png_files[0])
    first_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    target_size = first_img.shape[:2][::-1]  # (width, height)
    
    processed_images = []
    for i, filename in enumerate(png_files):
        img_path = os.path.join(folder_path, filename)
        depth_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            print(f"无法读取图像: {filename}")
            continue
        
        # 调整尺寸
        if depth_img.shape[:2][::-1] != target_size:
            depth_img = cv2.resize(depth_img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 使用全局范围进行归一化
        depth_normalized = np.clip((depth_img - global_min) / (global_max - global_min) * 255, 0, 255)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # 应用颜色映射
        colored_img = cv2.applyColorMap(depth_normalized, colormap)
        
        # 重命名图像为连续数字格式，便于FFmpeg处理
        output_filename = f"frame_{i:06d}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, colored_img)
        processed_images.append(output_path)
        
        if i % 10 == 0:  # 每10张打印一次进度
            print(f"已处理: {filename} -> {output_filename}")
    
    print(f"总共处理了 {len(processed_images)} 张图像")
    return processed_images

def create_video_simple(folder_path, output_video, fps=10):
    """
    使用最简单直接的方法创建视频
    """
    # 确保图像文件按顺序命名
    frame_pattern = os.path.join(folder_path, "frame_*.png")
    frames = glob.glob(frame_pattern)
    frames.sort()
    
    if not frames:
        print("未找到frame_*.png格式的图像文件")
        return False
    
    print(f"找到 {len(frames)} 张图像用于视频制作")
    
    # 使用最简单的FFmpeg命令
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-framerate', str(fps),  # 输入帧率
        '-i', os.path.join(folder_path, 'frame_%06d.png'),  # 输入图像序列
        '-vcodec', 'libx264',  # 视频编码器
        '-pix_fmt', 'yuv420p',  # 像素格式
        '-crf', '23',  # 质量控制
        '-preset', 'medium',  # 编码速度与质量平衡
        output_video
    ]
    
    try:
        print("正在创建视频...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"视频创建成功: {output_video}")
        print("FFmpeg输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg创建视频失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    # 设置路径
    input_folder = "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train/1EiJpeRNEs1.basis/220/depth"
    temp_folder = "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train/1EiJpeRNEs1.basis/220/depth/temp_processed"
    output_video = "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train/1EiJpeRNEs1.basis/220/depth/depth_video.mp4"
    
    print("开始处理深度图像...")
    
    # 处理图像
    processed_images = process_images_with_consistent_mapping(
        input_folder, 
        temp_folder, 
        colormap=cv2.COLORMAP_JET
    )
    
    if not processed_images:
        print("图像处理失败")
        return
    
    print(f"\n图像处理完成，共处理 {len(processed_images)} 张图像")
    
    # 创建视频
    print("\n开始创建视频...")
    success = create_video_simple(temp_folder, output_video, fps=10)
    
    if success:
        print(f"\n视频创建成功: {output_video}")
        
        # 验证视频
        cap = cv2.VideoCapture(output_video)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"\n视频信息:")
            print(f"  帧率: {fps}")
            print(f"  总帧数: {frame_count}")
            print(f"  时长: {duration:.2f} 秒")
            print(f"  分辨率: {width}x{height}")
            
            cap.release()
    else:
        print("\n视频创建失败")
        
        # 提供手动命令
        print("\n请尝试手动运行以下FFmpeg命令:")
        print(f"ffmpeg -y -framerate 10 -i {temp_folder}/frame_%06d.png -vcodec libx264 -pix_fmt yuv420p -crf 23 -preset medium {output_video}")

if __name__ == "__main__":
    # 检查FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg已找到")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未找到FFmpeg")
        print("请安装FFmpeg: https://ffmpeg.org/download.html")
        exit(1)
    
    main()