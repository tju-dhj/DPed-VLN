import os
import glob
import subprocess
import cv2
import numpy as np
from pathlib import Path

def create_video_from_jpgs(input_folder=".", output_video="rgb_video.mp4", fps=10):
    """
    从JPG图像创建视频
    """
    # 获取所有JPG文件并按数字顺序排序
    jpg_files = glob.glob(os.path.join(input_folder, "*.jpg"))
    jpg_files.extend(glob.glob(os.path.join(input_folder, "*.JPG")))
    jpg_files.extend(glob.glob(os.path.join(input_folder, "*.jpeg")))
    jpg_files.extend(glob.glob(os.path.join(input_folder, "*.JPEG")))
    
    if not jpg_files:
        print("未找到JPG图像文件")
        return False
    
    # 提取数字并排序（假设文件名格式为数字开头，如 0.jpg, 1.jpg, 2.jpg）
    def extract_number(filename):
        basename = os.path.basename(filename)
        number_part = basename.split('.')[0]
        if number_part.isdigit():
            return int(number_part)
        else:
            # 如果文件名包含下划线，如 0_0.jpg
            try:
                return int(number_part.split('_')[0])
            except:
                return 0
    
    jpg_files.sort(key=extract_number)
    
    # 注释掉详细输出以加快处理速度
    # print(f"找到 {len(jpg_files)} 张JPG图像")
    # for i, file in enumerate(jpg_files[:5]):  # 显示前5个文件
    #     print(f"  {os.path.basename(file)}")
    # if len(jpg_files) > 5:
    #     print(f"  ... 还有 {len(jpg_files)-5} 个文件")
    
    # 方法1: 使用ffmpeg直接从图像序列创建视频
    try:
        # 创建临时文件列表（如果文件名包含特殊字符）
        with open('temp_filelist.txt', 'w') as f:
            for jpg_file in jpg_files:
                f.write(f"file '{os.path.abspath(jpg_file)}'\n")
        
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-r', str(fps),  # 设置帧率
            '-f', 'concat',  # 使用文件列表格式
            '-safe', '0',  # 允许绝对路径
            '-i', 'temp_filelist.txt',  # 输入文件列表
            '-c:v', 'libx264',  # 视频编码器
            '-pix_fmt', 'yuv420p',  # 像素格式
            '-crf', '23',  # 质量控制 (18-28, 数值越小质量越高)
            '-preset', 'medium',  # 编码速度与压缩比的平衡
            output_video
        ]
        
        # print(f"\n正在创建视频: {output_video}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # print(f"视频创建成功: {output_video}")
            if os.path.exists('temp_filelist.txt'):
            os.remove('temp_filelist.txt')  # 删除临时文件
            return True
        else:
            # print(f"FFmpeg错误: {result.stderr}")
            if os.path.exists('temp_filelist.txt'):
            os.remove('temp_filelist.txt')
            return False
            
    except Exception as e:
        # print(f"方法1失败: {e}")
        if os.path.exists('temp_filelist.txt'):
        os.remove('temp_filelist.txt')
        return False

def create_video_from_jpgs_pattern(input_folder=".", output_video="rgb_video.mp4", fps=10):
    """
    如果图像文件有特定命名模式（如 0_0.jpg, 1_0.jpg），使用模式匹配
    """
    # 尝试匹配 0_0.jpg, 1_0.jpg, 2_0.jpg 等格式
    pattern_files = glob.glob(os.path.join(input_folder, "*_0.jpg"))
    pattern_files.extend(glob.glob(os.path.join(input_folder, "*_0.JPG")))
    
    if pattern_files:
        # 按数字排序
        def extract_number(filename):
            basename = os.path.basename(filename).split('_')[0]
            try:
                return int(basename)
            except:
                return 0
        
        pattern_files.sort(key=extract_number)
        
        if len(pattern_files) > 0:
            # 临时重命名文件为连续数字
            temp_dir = "temp_jpg_sequence"
            os.makedirs(temp_dir, exist_ok=True)
            
            for i, file in enumerate(pattern_files):
                new_name = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                os.link(file, new_name)  # 创建硬链接以节省空间
            
            # 使用ffmpeg创建视频
            cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'medium',
                output_video
            ]
            
            try:
                # print(f"正在创建视频: {output_video}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # print(f"视频创建成功: {output_video}")
                    # 清理临时文件
                    import shutil
                    shutil.rmtree(temp_dir)
                    return True
                else:
                    # print(f"FFmpeg错误: {result.stderr}")
                    import shutil
                    shutil.rmtree(temp_dir)
                    return False
            except Exception as e:
                # print(f"创建视频失败: {e}")
                import shutil
                shutil.rmtree(temp_dir)
                return False
    
    return False

def check_images_info(input_folder="."):
    """
    检查图像信息
    """
    jpg_files = glob.glob(os.path.join(input_folder, "*.jpg"))
    jpg_files.extend(glob.glob(os.path.join(input_folder, "*.JPG")))
    
    if not jpg_files:
        print("当前文件夹中没有找到JPG文件")
        return
    
    print(f"找到 {len(jpg_files)} 个JPG文件")
    
    # 读取第一张图像获取尺寸信息
    if jpg_files:
        img = cv2.imread(jpg_files[0])
        if img is not None:
            height, width, channels = img.shape
            print(f"图像尺寸: {width}x{height}, 通道数: {channels}")
        else:
            print("无法读取第一张图像")

def collect_all_episodes(root_dir):
    """
    收集所有episode的路径和对应的rgb目录
    返回: [(episode_path, rgb_dir), ...]
    """
    episodes = []
    root_path = Path(root_dir)
    
    # 遍历所有场景文件夹
    for scene_dir in sorted(root_path.glob("*.basis")):
        # 遍历每个场景下的所有episode文件夹
        for episode_dir in sorted(scene_dir.iterdir()):
            if episode_dir.is_dir():
                rgb_dir = episode_dir / "rgb"
                if rgb_dir.exists() and rgb_dir.is_dir():
                    episodes.append((episode_dir, rgb_dir))
    
    return episodes

def main():
    # 设置参数
    root_dir = "data/collect_data/val"
    fps = 10  # 帧率
    
    print(f"扫描目录: {root_dir}")
    print(f"帧率: {fps} fps\n")
    
    # 收集所有episode
    episodes = collect_all_episodes(root_dir)
    
    if not episodes:
        print("未找到任何episode")
        return
    
    print(f"找到 {len(episodes)} 个episode\n")
    
    # 为每个episode生成视频
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for idx, (episode_path, rgb_dir) in enumerate(episodes, 1):
        scene_name = episode_path.parent.name
        episode_id = episode_path.name
        
        # 视频保存在episode目录下
        output_video = episode_path / "rgb_video.mp4"
        
        print(f"[{idx}/{len(episodes)}] 处理: {scene_name}/{episode_id}")
        
        # 检查是否已存在视频
        if output_video.exists():
            print(f"  视频已存在，跳过: {output_video.name}")
            skip_count += 1
            continue
    
    # 尝试方法1：使用文件列表
        success = create_video_from_jpgs(str(rgb_dir), str(output_video), fps)
    
    if not success:
            print(f"  方法1失败，尝试方法2...")
        # 尝试方法2：使用模式匹配
            success = create_video_from_jpgs_pattern(str(rgb_dir), str(output_video), fps)
    
    if success:
            print(f"  ✓ 视频创建成功: {output_video.name}")
            success_count += 1
    else:
            print(f"  ✗ 视频创建失败")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成:")
    print(f"  成功: {success_count}/{len(episodes)}")
    print(f"  跳过: {skip_count}/{len(episodes)}")
    print(f"  失败: {fail_count}/{len(episodes)}")
    print(f"{'='*60}")

def create_video_manual_command():
    """
    提供手动FFmpeg命令
    """
    print("\n如果自动创建失败，可以使用以下手动命令:")
    print("方法1 (使用文件列表):")
    print("ffmpeg -y -r 10 -f concat -safe 0 -i filelist.txt -c:v libx264 -pix_fmt yuv420p rgb_video.mp4")
    print("\n方法2 (如果文件名是连续数字):")
    print("ffmpeg -y -framerate 10 -i %d.jpg -c:v libx264 -pix_fmt yuv420p rgb_video.mp4")
    print("\n方法3 (如果文件名有模式):")
    print("ffmpeg -y -framerate 10 -i %d_0.jpg -c:v libx264 -pix_fmt yuv420p rgb_video.mp4")

if __name__ == "__main__":
    # 检查FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg已找到\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未找到FFmpeg")
        create_video_manual_command()
        exit(1)
    
    # 运行主程序
    main()