#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import os
import json
import math
import cv2
import threading
import time
from threading import Thread
import numpy as np
import requests
from requests import RequestException
import copy
import io
import shutil

# --- 硬件与ROS库 ---
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError

# --- ROS 消息与服务 ---
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, PoseStamped
from sensor_msgs.msg import Image
from vln_stamp.srv import MoveDistance, MoveDistanceRequest, RotateAngle, RotateAngleRequest


sys.path.append("/home/tjark/alkaid")  # vlnce_utils.py 所在目录
from vlnce_utils import microphone, asr, tts, speaker, apply_gain, qwen_translate


BASE_URL = os.environ.get("VLNCE_SERVER_URL", "http://47.116.197.118:4173")
HTTP_TIMEOUT = float(os.environ.get("VLNCE_HTTP_TIMEOUT", "120"))

# ==============================================================================
#  RealSense 相机控制类
# ==============================================================================
class RealSense():
    def __init__(self):
        self.frame_width = 1280
        self.frame_height = 720
        self.frame_rate = 30
        self.depth_min = 0.1
        self.depth_max = 10.0
        self.decimation_filter_args = {
            "magnitude": 1.0
        }
        self.spatial_filter_args = {
            "magnitude": 2.0,
            "alpha": 0.5, 
            "delta": 20
        }
        self.temporal_filter_args = {
            "alpha": 0.4,
            "delta": 20
        }
        self.exit = False
        self.__color_image = None
        self.__depth_image = None
        self.__camera_pitch = 0
        self.__camera_roll = 0
        self.use_accel = False
        self.__frame_lock = threading.Lock()
        if not self.check():
            raise Exception("RealSense 相机初始化失败")
        self.updater = Thread(target=self.update_frame)
        self.updater.start()
        time.sleep(0.1)
        rospy.loginfo("RealSense 相机初始化成功")
        rospy.loginfo("RealSense 相机roll: {}, pitch: {}".format(self.camera_roll, self.camera_pitch))

    @classmethod
    def get_config(cls):
        return config.get("realsense", {})

    def update_frame(self):
        while not self.exit:
            time.sleep(0.01)
            with self.__frame_lock:
                try:
                    frames = self.pipeline.wait_for_frames(5000) # 增加超时
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not depth_frame or not color_frame:
                        rospy.logwarn("无法获取 RealSense 关键帧")
                        continue
                    
                    depth_frame = self.post_process_depth_frame(depth_frame)

                    self.__color_image = np.asanyarray(color_frame.get_data())
                    self.__depth_image = np.asanyarray(depth_frame.get_data())       # unit: mm

                    if self.use_accel:
                        accel_frame = frames.first(rs.stream.accel)
                        if accel_frame:
                            accel_data = accel_frame.as_motion_frame().get_motion_data()
                            accel_x, accel_y, accel_z = accel_data.x, accel_data.y, accel_data.z
                            pitch_rad = np.arctan2(np.sqrt(accel_x ** 2 + accel_z ** 2), accel_y)
                            roll_rad = np.arctan2(accel_x, np.sqrt(accel_y ** 2 + accel_z ** 2))
                            self.__camera_pitch = 180 - pitch_rad * 180 / np.pi
                            self.__camera_roll = roll_rad * 180 / np.pi
                except RuntimeError as e:
                    rospy.logerr("RealSense 运行时错误: {}".format(e))
                    continue


    def get_frame(self):
        '''获取当前帧的彩色图像和深度图像（深度图像单位为m）
        '''
        with self.__frame_lock:
            return self.__color_image, self.__depth_image
        
    @property
    def camera_pitch(self):
        '''获取相机pitch角度（单位：度）
        '''
        with self.__frame_lock:
            return self.__camera_pitch
        
    @property
    def camera_roll(self):
        '''获取相机roll角度（单位：度）
        '''
        with self.__frame_lock:
            return self.__camera_roll

    def post_process_depth_frame(self, depth_frame):
        assert (depth_frame.is_depth_frame())

        filtered_frame = self.decimation_filter.process(depth_frame)
        filtered_frame = self.threshold_filter.process(filtered_frame)
        filtered_frame = self.depth_to_disparity.process(filtered_frame)
        filtered_frame = self.spatial_filter.process(filtered_frame)
        filtered_frame = self.temporal_filter.process(filtered_frame)
        # filtered_frame = self.hole_filling_filter.process(filtered_frame)
        filtered_frame = self.disparity_to_depth.process(filtered_frame)

        return filtered_frame

    def check(self):
        stream_candidates = [
            (self.frame_width, self.frame_height, self.frame_rate, True),
            (self.frame_width, self.frame_height, self.frame_rate, False),
            (640, 480, 30, False),
        ]
        last_error = None

        for width, height, frame_rate, use_accel in stream_candidates:
            try:
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frame_rate)
                self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, frame_rate)
                if use_accel:
                    self.config.enable_stream(rs.stream.accel)

                self.pipeline.start(self.config)
                self.frame_width = width
                self.frame_height = height
                self.frame_rate = frame_rate
                self.use_accel = use_accel
                rospy.loginfo(
                    "RealSense 使用配置: {}x{}@{} accel={}".format(
                        width, height, frame_rate, use_accel
                    )
                )
                break
            except Exception as e:
                last_error = e
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                rospy.logwarn(
                    "RealSense 配置 {}x{}@{} accel={} 不可用: {}".format(
                        width, height, frame_rate, use_accel, e
                    )
                )
        else:
            rospy.logerr("无法创建 RealSense 相机实例: {}".format(last_error))
            return False

        try:
            device_list = self.pipeline.get_active_profile().get_device()
            device_name = device_list.get_info(rs.camera_info.name)
            rospy.loginfo("RealSense 相机设备名称: {}".format(device_name))

            # 设置自动曝光
            color_sensor = device_list.query_sensors()[1]  # 通常 color 是第二个 sensor
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1) # <-- 修改: 取消注释并设为1来启用自动曝光
            
            # 获取深度相机内参
            depth_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth)
            intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            self.cx = intrinsics.ppx
            self.cy = intrinsics.ppy
            self.fx = intrinsics.fx
            self.fy = intrinsics.fy
            self.coeffs = intrinsics.coeffs
            self.hfov = 2 * np.arctan(intrinsics.width / (2 * self.fx)) * 180 / np.pi
            self.vfov = 2 * np.arctan(intrinsics.height / (2 * self.fy)) * 180 / np.pi

            # 获取深度单位
            depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()       # unit: m
            rospy.loginfo("深度尺度为"+str(self.depth_scale))

            # 获取对齐
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            # 获取filters
            self.decimation_filter = rs.decimation_filter()
            self.threshold_filter = rs.threshold_filter(self.depth_min, self.depth_max)
            self.depth_to_disparity = rs.disparity_transform(True)
            self.spatial_filter = rs.spatial_filter()
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling_filter = rs.hole_filling_filter()
            self.disparity_to_depth = rs.disparity_transform(False)
            
            filter_magnitude = rs.option.filter_magnitude
            filter_smooth_alpha = rs.option.filter_smooth_alpha
            filter_smooth_delta = rs.option.filter_smooth_delta

            self.decimation_filter.set_option(filter_magnitude, self.decimation_filter_args["magnitude"])
            self.spatial_filter.set_option(filter_magnitude, self.spatial_filter_args["magnitude"])
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, self.spatial_filter_args["alpha"])
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, self.spatial_filter_args["delta"])
            self.temporal_filter.set_option(filter_smooth_alpha, self.temporal_filter_args["alpha"])
            self.temporal_filter.set_option(filter_smooth_delta, self.temporal_filter_args["delta"])

        except Exception as e:
            rospy.logerr("无法创建 RealSense 相机实例: {}".format(e))
            return False
        return True
    
    def close(self):
        self.exit = True
        self.updater.join()
        time.sleep(0.05)
        self.pipeline.stop()
def crop_and_resize(img, size=224):
    h, w = img.shape[:2]
    # 取短边，做中心裁剪
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    # 缩放到 size × size
    resized = cv2.resize(cropped, (size, size))
    return resized


def post_json(endpoint, **kwargs):
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.post(url, timeout=HTTP_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except RequestException as e:
        rospy.logerr(
            "请求模型服务失败: {} ({})。请检查服务端 Flask、FRP 和端口是否正常。".format(
                url, e
            )
        )
        raise
    except ValueError as e:
        rospy.logerr("模型服务返回的不是合法 JSON: {} ({})".format(url, e))
        raise


# ==============================================================================
#  路径执行器类 (核心推理逻辑)
# ==============================================================================
class PathInference:
    def __init__(self, info):
        self.info = info
        self.save_id = info
        
        if '_' not in self.info:
            rospy.logerr("在 'val' 模式下, info 参数格式必须为 'trajectory_id_instruction_id' (例如 '0_1')")
            sys.exit(1)
        self.trajectory_id, self.instruction_id = self.info.split('_')
        
        self.base_data_dir = '' # 存放原始轨迹数据的目录
        self.base_save_dir = '' # 存放推理时新产生数据的目录

        self.current_pose = None
        self.start_pose = None
        self.experiment_metadata = None
        self.pose_lock = threading.Lock()
        
        # --- 新增: 定义动作常量 ---
        self.move_distance = 0.25
        self.turn_angle_deg = 15.0
        self.turn_angle_rad = math.radians(self.turn_angle_deg)
        self.action_map = {
            0: "Stop",
            1: f"Move Forward ({self.move_distance}m)",
            2: f"Turn Left ({self.turn_angle_deg}°)",
            3: f"Turn Right ({self.turn_angle_deg}°)",
        }
        self.distance_limitation = 0.1      # 判断是否到达目标点的距离阈值（米）
        self.angle_limitation = 3 * math.pi / 180  # 判断是否到达目标姿态的角度阈值（弧度）

        # ------------------- 初始化ROS节点、服务、话题 -------------------
        try:
            rospy.init_node('path_executor', anonymous=True)
        except rospy.exceptions.ROSException:
            rospy.loginfo("ros主进程已初始化")

        
        # 初始化 RealSense 相机
        try:
            self.realsense = RealSense()
        except Exception as e:
            rospy.logfatal(f"无法初始化 RealSense 相机: {e}")
            sys.exit(1)

        # 等待运动控制服务
        rospy.loginfo("等待运动控制服务 'move_distance' 和 'rotate_angle'...")
        rospy.wait_for_service('move_distance')
        rospy.wait_for_service('rotate_angle')
        self.move_client = rospy.ServiceProxy('move_distance', MoveDistance)
        self.rotate_client = rospy.ServiceProxy('rotate_angle', RotateAngle)
        rospy.loginfo("运动控制服务已连接。")

        # 连接到 MoveBase (simple goal topic)
        # 注意: 如果您的move_base节点有不同的名字(例如tj_move_base), 话题也需要相应修改
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        rospy.loginfo("导航目标发布者已创建, 目标话题: /move_base_simple/goal")
        
        # 订阅位姿话题
        self.pose_sub = rospy.Subscriber("/robot_pose", PoseWithCovarianceStamped, self.pose_callback)
        
        rospy.loginfo(f"路径执行器已初始化。信息: {self.info}")

        rospy.sleep(1.0) # 等待发布者建立连接
    def pose_callback(self, msg):
        with self.pose_lock:
            self.current_pose = msg

    def set_start_pose_from_current_pose(self):
        """Fix the real-world episode start pose for GPS+Compass observations."""
        with self.pose_lock:
            if self.current_pose is None:
                return False
            self.start_pose = copy.deepcopy(self.current_pose.pose.pose)
        rospy.loginfo("已记录真实起点位姿，用于 starting_point_gps_compass。")
        return True

    def compute_starting_point_gps_compass(self, current_pose):
        """Return robot pose relative to the fixed start pose: [forward, left, heading]."""
        if self.start_pose is None:
            self.start_pose = copy.deepcopy(current_pose)

        start_yaw = self.quaternion_to_yaw(self.start_pose.orientation)
        current_yaw = self.quaternion_to_yaw(current_pose.orientation)

        dx = current_pose.position.x - self.start_pose.position.x
        dy = current_pose.position.y - self.start_pose.position.y

        cos_yaw = math.cos(start_yaw)
        sin_yaw = math.sin(start_yaw)
        forward = cos_yaw * dx + sin_yaw * dy
        left = -sin_yaw * dx + cos_yaw * dy
        heading = self.normalize_angle(current_yaw - start_yaw)

        return [forward, left, heading]

    def compute_pointgoal_to_start(self, current_pose):
        """Return the start point in the current robot frame: [forward, left]."""
        if self.start_pose is None:
            self.start_pose = copy.deepcopy(current_pose)

        current_yaw = self.quaternion_to_yaw(current_pose.orientation)
        dx = self.start_pose.position.x - current_pose.position.x
        dy = self.start_pose.position.y - current_pose.position.y

        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)
        forward = cos_yaw * dx + sin_yaw * dy
        left = -sin_yaw * dx + cos_yaw * dy

        return [forward, left]

    def build_predict_action_data(self, instruction, current_pose):
        starting_point_gps_compass = self.compute_starting_point_gps_compass(current_pose)
        pointgoal_to_start = self.compute_pointgoal_to_start(current_pose)
        meta = {
            "format": "cartesian",
            "frame": "start_pose",
            "values": ["forward_m", "left_m", "heading_rad"],
            "pointgoal_to_start": pointgoal_to_start,
        }
        return {
            "ep_id": self.info,
            "inst": instruction,
            "starting_point_gps_compass": json.dumps(starting_point_gps_compass),
            "starting_point_gps_compass_meta": json.dumps(meta),
        }

    def write_experiment_metadata(self, traj_save_dir, instruction):
        if self.experiment_metadata is None:
            return

        metadata_dir = os.path.join(traj_save_dir, "experiment_meta")
        os.makedirs(metadata_dir, exist_ok=True)
        metadata = copy.deepcopy(self.experiment_metadata)
        metadata.update({
            "ep_id": self.info,
            "trajectory_id": self.trajectory_id,
            "instruction_id": self.instruction_id,
            "instruction": instruction,
            "base_data_dir": self.base_data_dir,
            "base_save_dir": self.base_save_dir,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(os.path.join(metadata_dir, "0.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def call_move_service(self, distance):
        """调用移动服务。"""
        try:
            req = MoveDistanceRequest(distance=distance)
            res = self.move_client(req)
            if not res.success:
                rospy.logerr(f"移动 {distance:.2f} 米失败: {res.message}")
            return res.success
        except rospy.ServiceException as e:
            rospy.logerr(f"调用移动服务时出错: {e}")
            return False

    def call_rotate_service(self, angle):
        """调用旋转服务。"""
        try:
            req = RotateAngleRequest(angle=angle)
            res = self.rotate_client(req)
            if not res.success:
                rospy.logerr(f"旋转 {math.degrees(angle):.2f} 度失败: {res.message}")
            return res.success
        except rospy.ServiceException as e:
            rospy.logerr(f"调用旋转服务时出错: {e}")
            return False

    def run(self):
        """执行 'val' 模式的完整流程"""
        rospy.loginfo(f"开始执行 'val' 模式, 轨迹: {self.trajectory_id}, 指令: {self.instruction_id}")

        instruction, poses, actions = self.load_data()
        if instruction is None:
            return

        start_pose = poses[0]
        success = self.navigate_to_start(start_pose)
        # audio_file = tts.synthesis("Start")
        # speaker.play(audio_file)
        # while speaker.is_playing:
        #     time.sleep(0.5)
        # speaker.close()
        if not success:
            rospy.logerr("未能导航到初始位姿，任务中止。")
            return
        rospy.loginfo("已成功抵达初始位姿。")
        if not self.set_start_pose_from_current_pose():
            rospy.logerr("无法记录真实起点位姿，任务中止。")
            return
        rospy.sleep(3.0)

        self.execute_action_sequence2(actions, instruction)
        rospy.loginfo(f"轨迹 {self.info} 的所有动作已执行完毕。")

    def load_data(self):
        """从文件中加载指令、位姿和动作序列"""
        try:
            inst_path = os.path.join(self.base_data_dir, self.trajectory_id, 'inst_navcomposer', f"{self.instruction_id}.txt")
            pose_path = os.path.join(self.base_data_dir, self.trajectory_id, 'pose', '0.json')
            action_path = os.path.join(self.base_data_dir, self.trajectory_id, 'action', '0.json')

            rospy.loginfo(f"正在加载数据:\n  指令: {inst_path}\n  位姿: {pose_path}\n  动作: {action_path}")

            with open(inst_path, 'r') as f: instruction = f.read().strip()
            with open(pose_path, 'r') as f: poses = json.load(f)
            with open(action_path, 'r') as f: actions = json.load(f)
            
            rospy.loginfo(f"数据加载成功。共 {len(poses)} 个位姿点, {len(actions)} 个动作。")
            return instruction, poses, actions
        except Exception as e:
            rospy.logerr(f"加载数据时发生错误: {e}")
            return None, None, None

    def navigate_to_start(self, start_pose_dict):
        # time.sleep(5)
        # return True
        """通过发布话题的方式发送导航目标，并等待机器人到达。"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        if not isinstance(start_pose_dict, dict):
            start_pose_dict = self.pose_to_dict(start_pose_dict)
        pos = start_pose_dict['position']
        ori = start_pose_dict['orientation']
        goal_msg.pose.position.x = pos['x']
        goal_msg.pose.position.y = pos['y']
        goal_msg.pose.position.z = pos['z']
        goal_msg.pose.orientation.x = ori['x']
        goal_msg.pose.orientation.y = ori['y']
        goal_msg.pose.orientation.z = ori['z']
        goal_msg.pose.orientation.w = ori['w']
        rospy.loginfo(f"正在发布初始导航目标: [x: {pos['x']:.2f}, y: {pos['y']:.2f}]")

        self.goal_pub.publish(goal_msg)

        # --- 循环检查机器人是否到达目标点 ---
        start_time = rospy.Time.now()
        timeout_duration = rospy.Duration(120.0)
        rate = rospy.Rate(5) # 每秒检查5次
        
        dist_tolerance = 0.3  # 距离容忍误差 (米)
        angle_tolerance = 0.15 # 角度容忍误差 (弧度, 约8.6度)

        while not rospy.is_shutdown():
            # 检查是否超时
            if rospy.Time.now() - start_time > timeout_duration:
                rospy.logwarn("导航到初始位姿超时！")
                return False

            with self.pose_lock:
                current_pose_msg = self.current_pose
            
            if current_pose_msg is None:
                rospy.logwarn_throttle(5, "正在等待机器人当前位姿...")
                rate.sleep()
                continue
            
            # 计算误差
            current_pos = current_pose_msg.pose.pose.position
            dist_error = math.sqrt((pos['x'] - current_pos.x)**2 + (pos['y'] - current_pos.y)**2)
            
            goal_yaw = self.quaternion_to_yaw(goal_msg.pose.orientation)
            current_yaw = self.quaternion_to_yaw(current_pose_msg.pose.pose.orientation)
            angle_error = abs(self.normalize_angle(goal_yaw - current_yaw))

            # 检查是否满足条件
            if dist_error < dist_tolerance and angle_error < angle_tolerance:
                rospy.loginfo(f"已到达目标位置。距离误差: {dist_error:.3f}m, 角度误差: {math.degrees(angle_error):.2f}°")
                return True
            
            rospy.loginfo_throttle(5, f"正在前往初始点... 距离误差: {dist_error:.2f}m, 角度误差: {math.degrees(angle_error):.1f}°")
            rate.sleep()
        
        return False # 如果ROS关闭则返回失败

    def execute_action_sequence(self, gt_actions, instruction):
        """简单动作执行器"""
        infer_action = []
        infer_pose = []
        infer_time_info = []
        traj_save_dir = os.path.join(self.base_save_dir, self.save_id)
        if os.path.exists(traj_save_dir) and os.path.isdir(traj_save_dir):
            answer = input(f"文件夹 '{traj_save_dir}' 已存在，是否删除？(y/n): ").strip().lower()
            if answer == 'y':
                shutil.rmtree(traj_save_dir)
        resp = post_json("/reset_hiddens")
        print("reset_hiddens response:", resp)
        extra_step = len(gt_actions)//10+0
        for i in range(len(gt_actions)+extra_step):
            rospy.loginfo(f"-------------------- 步骤 {i + 1}/{len(gt_actions)+extra_step} --------------------")
            
            # 创建目录
            sub_dirs = ['infer_rgb', 'infer_depth', 'infer_pose', 'infer_action', 'infer_inst', 'infer_time_info']
            for sub_dir in sub_dirs:
                os.makedirs(os.path.join(traj_save_dir, sub_dir), exist_ok=True)
            if i == 0:
                self.write_experiment_metadata(traj_save_dir, instruction)
            filename_base = str(i)

            time.sleep(0.2)

            # 记录位姿
            if self.current_pose is None:
                rospy.logerr("无法获取当前机器人位姿，中止任务。")
                break
            with self.pose_lock:
                current_pose = self.current_pose.pose.pose
            infer_pose.append(self.pose_to_dict(current_pose))
            with open(os.path.join(traj_save_dir, "infer_pose", '0.json'), 'w') as f:
                json.dump(infer_pose, f)


            # 获取并保存图像
            cv_color, cv_depth = self.realsense.get_frame()
            cv2.imwrite(os.path.join(traj_save_dir, 'infer_rgb', f"{filename_base}.jpg"), cv_color)
            cv2.imwrite(os.path.join(traj_save_dir, 'infer_depth', f"{filename_base}.png"), cv_depth)
            cv_color = crop_and_resize(cv_color)
            cv_depth = crop_and_resize(cv_depth)

            _, cv_color_encode = cv2.imencode('.jpg', cv_color)
            _, cv_depth_encode = cv2.imencode('.png', cv_depth)

            # 保存指令 (仅在第一步保存)
            if i == 0:
                with open(os.path.join(traj_save_dir, "infer_inst", '0.txt'), 'w') as f:
                    f.write(instruction)

            # 请求动作
            files = {
                "rgb": (f"{i}.jpg", io.BytesIO(cv_color_encode.tobytes()), "image/jpeg"),
                "depth": (f"{i}.png", io.BytesIO(cv_depth_encode.tobytes()), "image/png")
            }
            data = {
                "ep_id": self.info,
                "inst": instruction
            }
            data.update(self.build_predict_action_data(instruction, current_pose))
            tic = time.time()
            resp = post_json("/predict_action", files=files, data=data)
            time_total_true = time.time()-tic
            print("predict_action response:", resp)
            action_code = resp["action"]
            time_info = resp["time_info"]
            time_info["net_true"] = time_total_true - time_info["total"]
            time_info["total_true"] = time_total_true

            # 记录动作
            infer_action.append(action_code)
            with open(os.path.join(traj_save_dir, "infer_action", '0.json'), 'w') as f:
                json.dump(infer_action, f)
            infer_time_info.append(time_info)
            with open(os.path.join(traj_save_dir, "infer_time_info", '0.json'), 'w') as f:
                json.dump(infer_time_info, f)

            # 执行动作
            if action_code == 1: # 前进
                res = self.move_client(MoveDistanceRequest(distance=self.move_distance))
                if not res.success: rospy.logerr(f"前进失败: {res.message}")
            
            elif action_code == 2: # 左转
                res = self.rotate_client(RotateAngleRequest(angle=self.turn_angle_rad))
                if not res.success: rospy.logerr(f"左转失败: {res.message}")

            elif action_code == 3: # 右转
                res = self.rotate_client(RotateAngleRequest(angle=-self.turn_angle_rad))
                if not res.success: rospy.logerr(f"右转失败: {res.message}")
            
            elif action_code == 0: # 停止
                    rospy.loginfo("动作是 'stop', 任务结束。")
                    break
            else:
                rospy.logwarn(f"未知的动作编码: '{action_code}'")


    def execute_action_sequence2(self, gt_actions, instruction):
        """里程计闭环动作执行器"""
        infer_action = []
        infer_pose = []
        infer_time_info = []
        traj_save_dir = os.path.join(self.base_save_dir, self.save_id)
        # 注意，会删除旧记录
        if os.path.exists(traj_save_dir) and os.path.isdir(traj_save_dir):
            shutil.rmtree(traj_save_dir)
        resp = post_json("/reset_hiddens")
        print("reset_hiddens response:", resp)
        extra_step = len(gt_actions)//10 + 10
        for i in range(len(gt_actions)+extra_step):
            rospy.loginfo(f"-------------------- 步骤 {i + 1}/{len(gt_actions)+extra_step} --------------------")
            
            # 创建目录
            sub_dirs = ['infer_rgb', 'infer_depth', 'infer_pose', 'infer_action', 'infer_inst', 'infer_time_info']
            for sub_dir in sub_dirs:
                os.makedirs(os.path.join(traj_save_dir, sub_dir), exist_ok=True)
            if i == 0:
                self.write_experiment_metadata(traj_save_dir, instruction)
            filename_base = str(i)

            rospy.sleep(0.1)

            # 记录位姿
            if self.current_pose is None:
                rospy.logerr("无法获取当前机器人位姿，中止任务。")
                break
            with self.pose_lock:
                current_pose = self.current_pose.pose.pose
            infer_pose.append(self.pose_to_dict(current_pose))
            with open(os.path.join(traj_save_dir, "infer_pose", '0.json'), 'w') as f:
                json.dump(infer_pose, f)


            # 获取并保存图像
            cv_color, cv_depth = self.realsense.get_frame()
            cv2.imwrite(os.path.join(traj_save_dir, 'infer_rgb', f"{filename_base}.jpg"), cv_color)
            cv2.imwrite(os.path.join(traj_save_dir, 'infer_depth', f"{filename_base}.png"), cv_depth)
            cv_color = crop_and_resize(cv_color)
            cv_depth = crop_and_resize(cv_depth)

            _, cv_color_encode = cv2.imencode('.jpg', cv_color)
            _, cv_depth_encode = cv2.imencode('.png', cv_depth)

            # 保存指令 (仅在第一步保存)
            if i == 0:
                with open(os.path.join(traj_save_dir, "infer_inst", '0.txt'), 'w') as f:
                    f.write(instruction)

            # 请求动作
            files = {
                "rgb": (f"{i}.jpg", io.BytesIO(cv_color_encode.tobytes()), "image/jpeg"),
                "depth": (f"{i}.png", io.BytesIO(cv_depth_encode.tobytes()), "image/png")
            }
            data = {
                "ep_id": self.info,
                "inst": instruction
            }
            data.update(self.build_predict_action_data(instruction, current_pose))
            tic = time.time()
            resp = post_json("/predict_action", files=files, data=data)
            time_total_true = time.time()-tic
            print("predict_action response:", resp)
            action_code = resp["action"]
            time_info = resp["time_info"]
            time_info["net_true"] = time_total_true - time_info["total"]
            time_info["total_true"] = time_total_true

            # 记录动作
            infer_action.append(action_code)
            with open(os.path.join(traj_save_dir, "infer_action", '0.json'), 'w') as f:
                json.dump(infer_action, f)
            infer_time_info.append(time_info)
            with open(os.path.join(traj_save_dir, "infer_time_info", '0.json'), 'w') as f:
                json.dump(infer_time_info, f)

            # # 执行动作（闭环）
            # if action_code == 0:
            #     rospy.loginfo("动作是 'stop', 任务结束。")
            #     break
            
            # # 3. 获取目标位姿 (下一个理论位姿点)
            # target_pose = self.compute_target_pose(current_pose, action_code)

            # # 4. 执行闭环移动控制

            # # 4a. 计算位移和朝向目标的旋转
            # dx = target_pose.position.x - current_pose.position.x
            # dy = target_pose.position.y - current_pose.position.y
            # distance_to_target = math.sqrt(dx**2 + dy**2)

            # if distance_to_target > self.distance_limitation:
            #     target_yaw_for_move = math.atan2(dy, dx)
            #     current_yaw = self.quaternion_to_yaw(current_pose.orientation)
            #     angle_diff_to_move = self.normalize_angle(target_yaw_for_move - current_yaw)

            #     rospy.loginfo(f"  -> 步骤1: 旋转 {math.degrees(angle_diff_to_move):.2f} 度朝向目标点。")
            #     self.call_rotate_service(angle_diff_to_move)
                
            #     rospy.loginfo(f"  -> 步骤2: 前进 {distance_to_target:.2f} 米。")
            #     self.call_move_service(distance_to_target)

            # # 4b. 到达后，调整为目标的最终姿态
            # rospy.sleep(0.1) # 等待位姿更新
            # with self.pose_lock:
            #     final_current_pose = self.current_pose.pose.pose

            # final_current_yaw = self.quaternion_to_yaw(final_current_pose.orientation)
            # target_final_yaw = self.quaternion_to_yaw(target_pose.orientation)
            # angle_diff_final = self.normalize_angle(target_final_yaw - final_current_yaw)

            # if abs(angle_diff_final) > self.angle_limitation:
            #     rospy.loginfo(f"  -> 步骤3: 调整最终姿态，旋转 {math.degrees(angle_diff_final):.2f} 度。")
            #     self.call_rotate_service(angle_diff_final)
            # 执行动作（开环）
            if action_code == 1: # 前进
                res = self.move_client(MoveDistanceRequest(distance=self.move_distance))
                if not res.success: rospy.logerr(f"前进失败: {res.message}")
            
            elif action_code == 2: # 左转
                res = self.rotate_client(RotateAngleRequest(angle=self.turn_angle_rad))
                if not res.success: rospy.logerr(f"左转失败: {res.message}")

            elif action_code == 3: # 右转
                res = self.rotate_client(RotateAngleRequest(angle=-self.turn_angle_rad))
                if not res.success: rospy.logerr(f"右转失败: {res.message}")
            
            elif action_code == 0: # 停止
                    rospy.loginfo("动作是 'stop', 任务结束。")
                    break
            else:
                rospy.logwarn(f"未知的动作编码: '{action_code}'")


    def run_free(self, language="en"):
        """执行 'free' 模式的完整流程"""
        # audio_file = tts.synthesis("Please say your instruction.")
        if language=="en": # 英语模式
            speaker.play("/home/tjark/alkaid/temp/vlnce_please_say.mp3")
            while speaker.is_playing:
                time.sleep(0.5)
            speaker.close()
            filename = microphone.record()
            microphone.close()
            # filename = apply_gain(filename)
            instruction = asr.recognize_en(filename)
            audio_file = tts.synthesis("Your instruction is: "+instruction)
            speaker.play(audio_file)
            # speaker.play("/home/tjark/alkaid/sdk/third_party/snowboy/resources/ding.wav")
            while speaker.is_playing:
                time.sleep(0.5)
            speaker.close()
        elif language=="cn": # 中文模式
            speaker.play("/home/tjark/alkaid/temp/vlnce_please_say_cn.mp3")
            while speaker.is_playing:
                time.sleep(0.5)
            speaker.close()
            filename = microphone.record()
            microphone.close()
            filename = apply_gain(filename, gain=1.5)
            instruction_cn = asr.recognize(filename)
            audio_file = tts.synthesis("您的指令是: "+instruction_cn)
            speaker.play(audio_file)
            # speaker.play("/home/tjark/alkaid/sdk/third_party/snowboy/resources/ding.wav")
            while speaker.is_playing:
                time.sleep(0.5)
            speaker.close()
            instruction = qwen_translate(instruction_cn)
        if not instruction.strip():
            return
        rospy.loginfo(f"开始执行 'free' 模式, 指令: {instruction}")

        if instruction is None:
            return

        with self.pose_lock:
            start_pose = self.current_pose.pose.pose
        success = self.navigate_to_start(start_pose)
        if not success:
            rospy.logerr("未能导航到初始位姿，任务中止。")
            return
        rospy.loginfo("已成功抵达初始位姿。")
        if not self.set_start_pose_from_current_pose():
            rospy.logerr("无法记录真实起点位姿，任务中止。")
            return

        self.execute_action_sequence3(instruction)
        rospy.loginfo(f"轨迹 {self.info} 的所有动作已执行完毕。")

    def execute_action_sequence3(self, instruction):
        """里程计闭环动作执行器, 自由样本（非val）"""
        free_action = []
        free_pose = []
        free_time_info = []
        traj_save_dir = os.path.join(self.base_save_dir, self.save_id)
        # 注意，会删除旧记录
        if os.path.exists(traj_save_dir) and os.path.isdir(traj_save_dir):
            shutil.rmtree(traj_save_dir)
        resp = post_json("/reset_hiddens")
        print("reset_hiddens response:", resp)
        extra_step = 150
        for i in range(extra_step):
            rospy.loginfo(f"-------------------- 步骤 {i + 1}/{extra_step} --------------------")
            
            # 创建目录
            sub_dirs = ['free_rgb', 'free_depth', 'free_pose', 'free_action', 'free_inst', 'free_time_info']
            for sub_dir in sub_dirs:
                os.makedirs(os.path.join(traj_save_dir, sub_dir), exist_ok=True)
            if i == 0:
                self.write_experiment_metadata(traj_save_dir, instruction)
            filename_base = str(i)

            rospy.sleep(0.1)

            # 记录位姿
            if self.current_pose is None:
                rospy.logerr("无法获取当前机器人位姿，中止任务。")
                break
            with self.pose_lock:
                current_pose = self.current_pose.pose.pose
            free_pose.append(self.pose_to_dict(current_pose))
            with open(os.path.join(traj_save_dir, "free_pose", '0.json'), 'w') as f:
                json.dump(free_pose, f)


            # 获取并保存图像
            cv_color, cv_depth = self.realsense.get_frame()
            cv2.imwrite(os.path.join(traj_save_dir, 'free_rgb', f"{filename_base}.jpg"), cv_color)
            cv2.imwrite(os.path.join(traj_save_dir, 'free_depth', f"{filename_base}.png"), cv_depth)
            cv_color = crop_and_resize(cv_color)
            cv_depth = crop_and_resize(cv_depth)

            _, cv_color_encode = cv2.imencode('.jpg', cv_color)
            _, cv_depth_encode = cv2.imencode('.png', cv_depth)

            # 保存指令 (仅在第一步保存)
            if i == 0:
                with open(os.path.join(traj_save_dir, "free_inst", '0.txt'), 'w') as f:
                    f.write(instruction)

            # 请求动作
            files = {
                "rgb": (f"{i}.jpg", io.BytesIO(cv_color_encode.tobytes()), "image/jpeg"),
                "depth": (f"{i}.png", io.BytesIO(cv_depth_encode.tobytes()), "image/png")
            }
            data = {
                "ep_id": self.info,
                "inst": instruction
            }
            data.update(self.build_predict_action_data(instruction, current_pose))
            tic = time.time()
            resp = post_json("/predict_action", files=files, data=data)
            time_total_true = time.time()-tic
            print("predict_action response:", resp)
            action_code = resp["action"]
            time_info = resp["time_info"]
            time_info["net_true"] = time_total_true - time_info["total"]
            time_info["total_true"] = time_total_true

            # 记录动作
            free_action.append(action_code)
            with open(os.path.join(traj_save_dir, "free_action", '0.json'), 'w') as f:
                json.dump(free_action, f)
            free_time_info.append(time_info)
            with open(os.path.join(traj_save_dir, "free_time_info", '0.json'), 'w') as f:
                json.dump(free_time_info, f)

            # 执行动作
            if action_code == 1: # 前进
                res = self.move_client(MoveDistanceRequest(distance=self.move_distance))
                if not res.success: rospy.logerr(f"前进失败: {res.message}")
            
            elif action_code == 2: # 左转
                res = self.rotate_client(RotateAngleRequest(angle=self.turn_angle_rad))
                if not res.success: rospy.logerr(f"左转失败: {res.message}")

            elif action_code == 3: # 右转
                res = self.rotate_client(RotateAngleRequest(angle=-self.turn_angle_rad))
                if not res.success: rospy.logerr(f"右转失败: {res.message}")
            
            elif action_code == 0: # 停止
                    rospy.loginfo("动作是 'stop', 任务结束。")
                    break
            else:
                rospy.logwarn(f"未知的动作编码: '{action_code}'")

    # --- 辅助函数 ---
    def quaternion_to_yaw(self, orientation):
        """将四元数转换为偏航角 (yaw)"""
        x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw

    def yaw_to_quaternion(self, yaw):
        """将偏航角 (yaw) 转换为四元数"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2)
        q.w = math.cos(yaw / 2)
        return q
    def compute_target_pose(self, current_pose, action_code):
        """
        根据当前位姿和动作码(action_code)，计算目标位姿。
        action_code: 
            1 = 前进0.25米
            2 = 左转15度
            3 = 右转15度
            0 = 停止
        """
        target_pose = PoseStamped()
        target_pose.header.frame_id = "map"
        target_pose.header.stamp = rospy.Time.now()

        # 当前位姿
        x = current_pose.position.x
        y = current_pose.position.y
        yaw = self.quaternion_to_yaw(current_pose.orientation)

        # 根据动作计算
        if action_code == 1:  # 前进0.25米
            step = 0.25
            x += step * math.cos(yaw)
            y += step * math.sin(yaw)

        elif action_code == 2:  # 左转15度
            yaw += math.radians(15)

        elif action_code == 3:  # 右转15度
            yaw -= math.radians(15)

        elif action_code == 0:
            rospy.loginfo("动作是 'stop', 任务结束。")
            return None

        # 转回四元数
        q = self.yaw_to_quaternion(yaw)

        # 填充结果
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = current_pose.position.z
        target_pose.pose.orientation = q

        return target_pose.pose

    def normalize_angle(self, angle):
        """将角度规范化到 [-pi, pi] 范围内"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    def shutdown(self):
        """节点关闭前的清理工作"""
        rospy.loginfo("正在关闭 PathInference...")
        if hasattr(self, 'realsense'):
            self.realsense.close()
        rospy.loginfo("清理完成。")
    def pose_to_dict(self, pose):
        """
        将 ROS 的 Pose 对象转换为包含 yaw 的字典。
        
        参数:
            pose (geometry_msgs.msg.Pose): ROS Pose 对象

        返回:
            dict: 包含 position, orientation 和 yaw 的字典
        """
        return {
            'position': {
                'x': pose.position.x,
                'y': pose.position.y,
                'z': pose.position.z
            },
            'orientation': {
                'x': pose.orientation.x,
                'y': pose.orientation.y,
                'z': pose.orientation.z,
                'w': pose.orientation.w
            },
            'yaw': self.quaternion_to_yaw(pose.orientation),
            'yaw_angle': self.quaternion_to_yaw(pose.orientation) * 180 / math.pi
        }
if __name__ == '__main__':

    mode_arg = sys.argv[1]
    info_arg = sys.argv[2]
    executor = None
    if mode_arg=="val":
        executor = PathInference(info=info_arg)
        executor.base_data_dir = 'navtj_extend/val' # 存放原始轨迹数据的目录
        executor.base_save_dir = 'vlnce_test/infer_val' # 存放推理时新产生数据的目录
        rospy.on_shutdown(executor.shutdown) # 注册关闭钩子
        executor.run()
    elif mode_arg=="dynamic_val":
        executor = PathInference(info=info_arg)
        executor.base_data_dir = 'navtj_extend/val' # 存放原始轨迹数据的目录
        executor.base_save_dir = 'vlnce_test/dynamic_val' # 存放动态行人实体实验数据的目录
        if len(sys.argv) >= 4:
            with open(sys.argv[3], "r") as f:
                executor.experiment_metadata = json.load(f)
            executor.save_id = executor.experiment_metadata.get("trial_id", executor.info)
        rospy.on_shutdown(executor.shutdown) # 注册关闭钩子
        executor.run()
    else:
        if mode_arg=="free":
            language = "en"
        elif mode_arg=="free_cn":
            language = "cn"
        executor = PathInference(info=info_arg)
        executor.base_data_dir = '' # 存放原始轨迹数据的目录
        executor.base_save_dir = 'vlnce_test/free' # 存放推理时新产生数据的目录
        rospy.on_shutdown(executor.shutdown) # 注册关闭钩子
        executor.run_free(language)

