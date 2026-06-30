#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人端 VLN-CE HTTP 客户端
============================

功能:
  - 与服务端 Flask HTTP Server 通信获取动作预测
  - 支持 Brain 指令优化模式（自动更新优化后的指令）
  - 4动作/6动作空间自适应
  - 模拟相机/真实相机双模式
  - 支持内网穿透代理

用法:
  真实机器人:
    python robot_vln_client.py --server-url http://10.0.0.100:32146 --mode real

  模拟测试:
    python robot_vln_client.py --server-url http://localhost:32146 --mode simulate --image-dir ./test_images

HTTP API:
  POST /reset_hiddens  → 初始化 RNN hidden states + Brain episode
  POST /predict_action  → 发送观测，返回动作
"""

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests


# =============================================================================
# 动作映射
# =============================================================================

ACTION_4_MAP = {
    0: "STOP",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
}

ACTION_6_MAP = {
    0: "STOP",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "PAUSE",
    5: "BACKWARD",
}


# =============================================================================
# 机器人控制接口（用户需根据实际机器人硬件实现）
# =============================================================================

class RobotController:
    """机器人底盘控制抽象接口 — 用户根据实际机器人实现"""

    def __init__(self, use_6_actions: bool = False):
        self.use_6_actions = use_6_actions
        self.action_map = ACTION_6_MAP if use_6_actions else ACTION_4_MAP

    def stop(self):
        """停止运动"""
        print("  [Robot] STOP")
        # TODO: 你的机器人停止控制代码
        # 例如: ros_publisher.publish(Twist(linear=0, angular=0))

    def move_forward(self, speed: float = 0.25, duration: float = 0.25):
        """前进"""
        print(f"  [Robot] FORWARD (speed={speed}, duration={duration}s)")
        # TODO: 你的机器人前进控制代码

    def turn_left(self, angular_speed: float = 0.5, duration: float = 0.25):
        """左转"""
        print(f"  [Robot] LEFT (angular_speed={angular_speed}, duration={duration}s)")
        # TODO: 你的机器人左转控制代码

    def turn_right(self, angular_speed: float = 0.5, duration: float = 0.25):
        """右转"""
        print(f"  [Robot] RIGHT (angular_speed={angular_speed}, duration={duration}s)")
        # TODO: 你的机器人右转控制代码

    def pause(self, duration: float = 1.0):
        """原地等待（仅6动作）"""
        print(f"  [Robot] PAUSE (duration={duration}s)")
        # TODO: 你的机器人暂停控制代码

    def move_backward(self, speed: float = -0.25, duration: float = 0.25):
        """后退（仅6动作）"""
        print(f"  [Robot] BACKWARD (speed={speed}, duration={duration}s)")
        # TODO: 你的机器人后退控制代码

    def execute_action(self, action_id: int, **kwargs):
        """根据 action_id 执行动作"""
        action_name = self.action_map.get(action_id, f"UNKNOWN_{action_id}")
        print(f"\n  >>> Action: {action_name} (id={action_id})")

        if action_id == 0:
            self.stop()
        elif action_id == 1:
            self.move_forward(**kwargs)
        elif action_id == 2:
            self.turn_left(**kwargs)
        elif action_id == 3:
            self.turn_right(**kwargs)
        elif action_id == 4:
            if self.use_6_actions:
                self.pause(**kwargs)
            else:
                print(f"  [WARN] PAUSE not in 4-action space, treating as STOP")
                self.stop()
        elif action_id == 5:
            if self.use_6_actions:
                self.move_backward(**kwargs)
            else:
                print(f"  [WARN] BACKWARD not in 4-action space, treating as STOP")
                self.stop()
        else:
            print(f"  [WARN] Unknown action ID: {action_id}")

        return action_name


# =============================================================================
# 相机接口
# =============================================================================

class CameraInterface:
    """相机抽象接口 — 用户根据实际相机实现"""

    def __init__(self, image_dir: str = None):
        """
        Args:
            image_dir: 模拟模式下读取图像的目录
        """
        self.image_dir = image_dir
        self._sim_image_list = []
        self._sim_idx = 0

        if image_dir:
            self._sim_image_list = sorted(
                [str(p) for p in Path(image_dir).glob("*.jpg")]
                + [str(p) for p in Path(image_dir).glob("*.png")]
            )
            print(f"[Camera] Simulate mode: found {len(self._sim_image_list)} images in {image_dir}")

    def get_rgb_jpeg(self) -> bytes:
        """获取当前 RGB 图像 (JPEG bytes)"""
        if self._sim_image_list:
            # 模拟模式: 从目录循环读取
            img_path = self._sim_image_list[self._sim_idx % len(self._sim_image_list)]
            self._sim_idx += 1
            with open(img_path, "rb") as f:
                return f.read()

        # 真实相机模式: 用户需要实现
        # TODO: 接入你的真实相机
        # 例如 ROS:
        #   from sensor_msgs.msg import Image
        #   from cv_bridge import CvBridge
        #   cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        #   _, jpeg_bytes = cv2.imencode(".jpg", cv_image)
        #   return jpeg_bytes.tobytes()
        raise NotImplementedError(
            "请实现 get_rgb_jpeg(): 接入真实相机或使用 --image-dir 模拟模式"
        )

    def get_depth_bytes(self) -> Optional[bytes]:
        """获取当前深度图像 (PNG bytes, 可选)"""
        # 深度不是必需的，如果无需深度返回 None
        return None

    def close(self):
        """释放相机资源"""
        pass


# =============================================================================
# HTTP 客户端核心
# =============================================================================

class VLNClient:
    """VLN-CE 服务端 HTTP 客户端"""

    def __init__(
        self,
        server_url: str,
        use_6_actions: bool = False,
        timeout: float = 30.0,
        verbose: bool = True,
    ):
        self.server_url = server_url.rstrip("/")
        self.use_6_actions = use_6_actions
        self.timeout = timeout
        self.verbose = verbose

        self.session = requests.Session()
        # 连接池配置
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=4,
            max_retries=3,
        )
        self.session.mount("http://", adapter)

        self._current_ep_id: str = ""
        self._current_instruction: str = ""
        self._step_count: int = 0
        self._brain_enabled: bool = False
        self._brain_modifications: int = 0

    def start_episode(self, ep_id: str, instruction: str) -> bool:
        """
        开始新的导航 episode。

        Args:
            ep_id: episode 标识符
            instruction: 导航指令文本

        Returns:
            True if successful
        """
        self._current_ep_id = ep_id
        self._current_instruction = instruction
        self._step_count = 0
        self._brain_modifications = 0

        t_start = time.time()
        try:
            resp = self.session.post(
                f"{self.server_url}/reset_hiddens",
                data={
                    "ep_id": ep_id,
                    "inst": instruction,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            self._brain_enabled = data.get("brain_enabled", False)
            elapsed = (time.time() - t_start) * 1000

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"[Episode Start] ID: {ep_id}")
                print(f"  Instruction: {instruction}")
                print(f"  Brain enabled: {self._brain_enabled}")
                print(f"  Reset latency: {elapsed:.0f}ms")
                print(f"{'='*60}")
            return data.get("status") == "success"
        except Exception as e:
            print(f"[ERROR] Failed to start episode: {e}")
            return False

    def predict_action(
        self,
        rgb_jpeg: bytes,
        depth_bytes: Optional[bytes] = None,
        goal_x: float = 0.0,
        goal_y: float = 0.0,
        compass: float = 0.0,
    ) -> Dict:
        """
        发送当前帧观测，获取动作预测。

        Args:
            rgb_jpeg: RGB 图像 JPEG bytes
            depth_bytes: 深度图像 bytes (可选)
            goal_x: 相对目标的 X 坐标
            goal_y: 相对目标的 Y 坐标
            compass: 罗盘航向 (弧度)

        Returns:
            {
                "action": int,
                "action_name": str,
                "instruction_modified": bool,
                "optimized_instruction": str | None,
                "pedestrian_detected": bool,
                "pedestrian_count": int,
                "time_info": dict,
            }
        """
        self._step_count += 1

        files = {"rgb": ("frame.jpg", rgb_jpeg, "image/jpeg")}
        if depth_bytes is not None:
            files["depth"] = ("depth.png", depth_bytes, "image/png")

        try:
            resp = self.session.post(
                f"{self.server_url}/predict_action",
                data={
                    "ep_id": self._current_ep_id,
                    "inst": self._current_instruction,
                    "goal_x": str(goal_x),
                    "goal_y": str(goal_y),
                    "compass": str(compass),
                },
                files=files,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("status") != "success":
                print(f"[ERROR] predict_action failed: {result.get('message', 'unknown')}")
                return {"action": 0, "action_name": "STOP", "error": result.get("message")}

            action_id = result["action"]
            action_name = (ACTION_6_MAP if self.use_6_actions else ACTION_4_MAP).get(
                action_id, f"UNKNOWN_{action_id}"
            )

            # 处理 Brain 指令修改
            if result.get("instruction_modified"):
                self._brain_modifications += 1
                new_inst = result.get("optimized_instruction", "")
                if new_inst:
                    old_inst = self._current_instruction
                    self._current_instruction = new_inst
                    if self.verbose:
                        print(f"\n  [Brain] Instruction modified (#{self._brain_modifications})!")
                        print(f"    Old: {old_inst[:80]}...")
                        print(f"    New: {new_inst[:80]}...")

            if self.verbose:
                time_info = result.get("time_info", {})
                ped_str = ""
                if result.get("pedestrian_detected"):
                    ped_str = f" | Pedestrians: {result.get('pedestrian_count', 0)}"
                brain_str = ""
                if result.get("instruction_modified"):
                    brain_str = " | [BRAIN-MODIFIED]"
                print(
                    f"  Step {self._step_count:3d} | Action: {action_name:8s} (id={action_id})"
                    f" | Total: {time_info.get('total', 0)*1000:.0f}ms"
                    f" | Model: {time_info.get('model', 0)*1000:.0f}ms"
                    f"{ped_str}{brain_str}"
                )

            return {
                "action": action_id,
                "action_name": action_name,
                "instruction_modified": result.get("instruction_modified", False),
                "optimized_instruction": result.get("optimized_instruction"),
                "pedestrian_detected": result.get("pedestrian_detected", False),
                "pedestrian_count": result.get("pedestrian_count", 0),
                "time_info": result.get("time_info", {}),
            }

        except requests.exceptions.Timeout:
            print(f"[ERROR] predict_action timeout after {self.timeout}s, returning STOP")
            return {"action": 0, "action_name": "STOP", "error": "timeout"}
        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Connection failed: {e}")
            return {"action": 0, "action_name": "STOP", "error": "connection"}
        except Exception as e:
            print(f"[ERROR] predict_action failed: {e}")
            traceback.print_exc()
            return {"action": 0, "action_name": "STOP", "error": str(e)}

    def get_current_instruction(self) -> str:
        """获取当前指令（Brain 可能已修改过）"""
        return self._current_instruction

    @property
    def brain_modifications(self) -> int:
        return self._brain_modifications

    @property
    def step_count(self) -> int:
        return self._step_count

    def close(self):
        self.session.close()


# =============================================================================
# GPS / 定位接口
# =============================================================================

class LocalizationInterface:
    """定位接口 — 提供 goal_x, goal_y, compass"""

    def __init__(self, goal_x: float = 5.0, goal_y: float = 0.0, compass: float = 0.0):
        """
        Args:
            goal_x: 目标在机器人前方的距离 (米)
            goal_y: 目标在机器人左右的偏移 (米, 正值=右侧)
            compass: 机器人当前的罗盘航向 (弧度)
        """
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.compass = compass

    def update(self):
        """
        更新定位数据。
        真实机器人应在此处从 ROS/sensor 读取数据。
        """
        # TODO: 从你的定位系统更新
        # 例如: self.compass = ros_imu.yaw
        pass

    def get_gps_compass_dict(self) -> Dict[str, float]:
        return {
            "goal_x": self.goal_x,
            "goal_y": self.goal_y,
            "compass": self.compass,
        }


# =============================================================================
# 主循环
# =============================================================================

def run_navigation_loop(
    client: VLNClient,
    camera: CameraInterface,
    localization: LocalizationInterface,
    robot: RobotController,
    max_steps: int = 500,
    step_delay: float = 0.25,
    ep_id: str = "mission_001",
    instruction: str = "Go straight and turn left at the end of the hallway",
):
    """
    主导航循环。

    1. 调用 /reset_hiddens 初始化
    2. 每帧:
       a. 获取相机图像
       b. 获取 GPS/compass
       c. 调用 /predict_action
       d. 执行动作
       e. 如果 action=STOP，退出
    3. 打印统计
    """
    # 1. 开始 episode
    if not client.start_episode(ep_id, instruction):
        print("[FATAL] Failed to start episode, aborting")
        return

    t_episode_start = time.time()

    try:
        for step in range(max_steps):
            # 2a. 获取相机图像
            t_img_start = time.time()
            try:
                rgb_jpeg = camera.get_rgb_jpeg()
                depth_bytes = camera.get_depth_bytes()
            except NotImplementedError as e:
                print(f"[FATAL] Camera not available: {e}")
                print("  Run with --mode simulate --image-dir <path> for testing")
                break
            t_img = (time.time() - t_img_start) * 1000

            # 2b. 获取定位
            localization.update()
            loc = localization.get_gps_compass_dict()

            # 2c. 获取动作预测
            result = client.predict_action(
                rgb_jpeg=rgb_jpeg,
                depth_bytes=depth_bytes,
                goal_x=loc["goal_x"],
                goal_y=loc["goal_y"],
                compass=loc["compass"],
            )

            if "error" in result:
                print(f"  [WARN] Prediction error: {result['error']}, retrying...")
                time.sleep(1.0)
                continue

            action_id = result["action"]

            # 2d. 执行动作
            robot.execute_action(action_id)

            # 2e. 检查 STOP
            if action_id == 0:
                print(f"\n  >>> Goal reached! Episode completed at step {step + 1}")
                break

            # 控制帧率
            elapsed = (time.time() - t_episode_start) - (step + 1) * step_delay
            if elapsed < 0:
                time.sleep(-elapsed)

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
    finally:
        t_total = time.time() - t_episode_start
        print(f"\n{'='*60}")
        print(f"[Episode Summary]")
        print(f"  ID: {ep_id}")
        print(f"  Steps: {client.step_count}")
        print(f"  Total time: {t_total:.1f}s")
        print(f"  Brain modifications: {client.brain_modifications}")
        print(f"  Final instruction: {client.get_current_instruction()[:100]}...")
        print(f"{'='*60}")
        robot.stop()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLN-CE 机器人客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 真实机器人模式 (需实现 RobotController 和 CameraInterface)
  python robot_vln_client.py --server-url http://10.0.0.100:32146 --mode real

  # 模拟测试模式 (从目录加载图像)
  python robot_vln_client.py --server-url http://localhost:32146 \\
      --mode simulate --image-dir ./test_images

  # 内网穿透模式
  python robot_vln_client.py --server-url http://myrobot.frp.xyz:32146 --mode real
        """,
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:32146",
        help="服务端 URL (默认: http://localhost:32146)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["real", "simulate"],
        default="simulate",
        help="运行模式: real=真实机器人, simulate=模拟测试",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="模拟模式下加载图像的目录",
    )
    parser.add_argument(
        "--ep-id",
        type=str,
        default="mission_001",
        help="Episode ID",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Go straight to the end of the hallway and turn left",
        help="导航指令",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        choices=[4, 6],
        default=4,
        help="动作空间大小: 4 或 6 (需与服务端checkpoint一致)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="最大步数",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.25,
        help="每步之间的最小延迟 (秒)",
    )
    parser.add_argument(
        "--goal-x",
        type=float,
        default=5.0,
        help="目标相对 X 坐标 (米)",
    )
    parser.add_argument(
        "--goal-y",
        type=float,
        default=0.0,
        help="目标相对 Y 坐标 (米)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP 请求超时 (秒)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="详细输出",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式",
    )

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    print(f"\n{'='*60}")
    print(f"VLN-CE Robot Client")
    print(f"{'='*60}")
    print(f"  Server: {args.server_url}")
    print(f"  Mode: {args.mode}")
    print(f"  Actions: {args.num_actions}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    # 初始化组件
    use_6_actions = args.num_actions == 6

    client = VLNClient(
        server_url=args.server_url,
        use_6_actions=use_6_actions,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    camera = CameraInterface(image_dir=args.image_dir)

    localization = LocalizationInterface(
        goal_x=args.goal_x,
        goal_y=args.goal_y,
    )

    robot = RobotController(use_6_actions=use_6_actions)

    # 运行导航
    try:
        run_navigation_loop(
            client=client,
            camera=camera,
            localization=localization,
            robot=robot,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
            ep_id=args.ep_id,
            instruction=args.instruction,
        )
    finally:
        client.close()
        camera.close()


if __name__ == "__main__":
    main()
