# -*- coding: utf-8 -*-
# ==============================================================================
# 文件: pedestrian_detection.py
# 描述: 行人检测模块 - 提供统一的行人检测接口，支持多种检测器后端
# ==============================================================================

"""
PedestrianDetectionModule - 行人检测模块
=========================================

本模块提供统一的行人检测接口，支持以下检测器：
1. YOLOv8 (Ultralytics) - 轻量级高性能检测器
2. RT-DETR (HuggingFace Transformers) - 基于DETR的实时检测器

检测器启动选项可通过配置文件控制，不启用时不影响原有系统。

主要功能：
- 统一检测接口
- 支持CPU/GPU切换
- 检测结果格式标准化
- 可配置置信度阈值和输入尺寸
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DetectorType(Enum):
    """
    检测器类型枚举
    =================
    定义支持的行人检测器类型
    """
    YOLOV8N = "yolov8n"          # YOLOv8 Nano - 最轻量
    YOLOV8S = "yolov8s"          # YOLOv8 Small - 轻量
    YOLOV8M = "yolov8m"          # YOLOv8 Medium - 中等
    RTDETR_R18 = "rtdetr_r18"    # RT-DETR ResNet18vd
    RTDETR_R50 = "rtdetr_r50"    # RT-DETR ResNet50vd
    DISABLED = "disabled"        # 不启用检测


@dataclass
class PedestrianDetection:
    """
    行人检测结果数据结构
    ==========================
    存储单个行人检测的结果信息
    """
    bbox: List[float]           # 边界框坐标 [x1, y1, x2, y2]
    score: float                # 置信度分数
    center: List[float]         # 边界框中心点 [cx, cy]
    width: float                # 边界框宽度
    height: float               # 边界框高度
    relative_area: float        # 相对图像面积比例


@dataclass
class DetectionResult:
    """
    检测结果包装类
    ================
    包含一帧图像的所有检测结果
    """
    frame_id: int                          # 帧ID
    detection_list: List[PedestrianDetection]  # 检测到的行人列表
    num_detections: int                   # 检测数量
    elapsed_ms: float                      # 检测耗时(毫秒)
    success: bool                          # 检测是否成功
    image_shape: tuple                     # 原始图像尺寸 (H, W, C)
    has_pedestrian: bool                   # 是否检测到行人


class PedestrianDetector:
    """
    统一行人检测器接口
    ====================

    该类封装了多种行人检测器的调用接口，提供统一的结果格式输出。
    支持热切换检测器类型，检测器通过配置选项控制启用状态。

    Attributes:
        detector_type: 检测器类型
        device: 计算设备
        confidence: 置信度阈值
        input_size: 输入图像尺寸
        model: 检测器模型实例
        processor: 图像预处理实例 (RT-DETR专用)
    """

    def __init__(
        self,
        detector_type: str = "yolov8n",
        device: str = "cuda",
        confidence: float = 0.25,
        input_size: int = 640,
        checkpoint_path: Optional[str] = None,
        hf_model_id: Optional[str] = None,
    ):
        """
        初始化行人检测器

        Args:
            detector_type: 检测器类型字符串，支持 yolov8n/s/m, rtdetr_r18, rtdetr_r50, disabled
            device: 计算设备，"cuda"或"cpu"
            confidence: 置信度阈值，低于此值的检测结果将被过滤
            input_size: 模型输入尺寸，较大的值可能提高精度但降低速度
            checkpoint_path: YOLO模型权重路径，若为None则自动下载
            hf_model_id: HuggingFace模型ID (RT-DETR专用)
        """
        self.detector_type = DetectorType(detector_type.lower())
        self.device = self._get_device(device)
        self.confidence = confidence
        self.input_size = input_size
        self.checkpoint_path = checkpoint_path
        self.hf_model_id = hf_model_id

        self.model = None
        self.processor = None

        # 初始化模型
        self._initialize_model()

    def _get_device(self, device: str) -> torch.device:
        """
        获取计算设备

        Args:
            device: 设备字符串

        Returns:
            torch.device对象
        """
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _initialize_model(self) -> None:
        """
        根据检测器类型初始化对应的模型

        该方法根据self.detector_type创建对应的检测器模型实例。
        DISABLED类型不加载任何模型。
        """
        if self.detector_type == DetectorType.DISABLED:
            return

        if self.detector_type in [DetectorType.YOLOV8N, DetectorType.YOLOV8S, DetectorType.YOLOV8M]:
            self._init_yolo_model()
        elif self.detector_type in [DetectorType.RTDETR_R18, DetectorType.RTDETR_R50]:
            self._init_rtdetr_model()
        else:
            raise ValueError(f"不支持的检测器类型: {self.detector_type}")

    # 默认 YOLO 权重搜索路径（含本地 pretrained_model 目录）
    _DEFAULT_YOLO_PATHS = [
        "pretrained_model/yolov8n-seg.pt",
        "pretrained_model/yolov8s-seg.pt",
        "pretrained_model/yolov8m-seg.pt",
    ]

    def _init_yolo_model(self) -> None:
        from ultralytics import YOLO

        model_name = self.detector_type.value
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            self.model = YOLO(self.checkpoint_path)
        else:
            # 按 detector_type 推断文件名
            seg_name_map = {
                "yolov8n": "pretrained_model/yolov8n-seg.pt",
                "yolov8s": "pretrained_model/yolov8s-seg.pt",
                "yolov8m": "pretrained_model/yolov8m-seg.pt",
            }
            seg_file = seg_name_map.get(model_name)
            if seg_file and Path(seg_file).exists():
                self.model = YOLO(seg_file)
            else:
                self.model = YOLO(f"{model_name}-seg.pt")

        self.model.to(self.device)

    def _init_rtdetr_model(self) -> None:
        try:
            from transformers import AutoImageProcessor, RTDetrForObjectDetection
        except ImportError:
            raise ImportError(
                "RT-DETR 需要安装 transformers>=4.49 且支持 RTDetrForObjectDetection。"
                "建议使用 YOLO 检测器（yolov8n/s/m），无需额外依赖。"
            )

        model_mapping = {
            DetectorType.RTDETR_R18: "PekingU/rtdetr_r18vd",
            DetectorType.RTDETR_R50: "PekingU/rtdetr_r50vd",
        }

        model_id = self.hf_model_id or model_mapping.get(
            self.detector_type, "PekingU/rtdetr_r18vd"
        )

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def detect(
        self,
        image: np.ndarray,
        frame_id: int = 0,
    ) -> DetectionResult:
        """
        对输入图像执行行人检测

        Args:
            image: 输入图像，numpy数组，格式为RGB (H, W, 3)
            frame_id: 当前帧ID，用于追踪

        Returns:
            DetectionResult: 检测结果对象

        Note:
            输入图像应为RGB格式，YOLO和RT-DETR内部会进行格式转换
        """
        if self.detector_type == DetectorType.DISABLED:
            return DetectionResult(
                frame_id=frame_id,
                detection_list=[],
                num_detections=0,
                elapsed_ms=0.0,
                success=True,
                image_shape=image.shape,
                has_pedestrian=False,
            )

        if self.detector_type in [DetectorType.YOLOV8N, DetectorType.YOLOV8S, DetectorType.YOLOV8M]:
            return self._detect_yolo(image, frame_id)
        else:
            return self._detect_rtdetr(image, frame_id)

    def _detect_yolo(
        self,
        image: np.ndarray,
        frame_id: int,
    ) -> DetectionResult:
        """
        YOLO检测器执行

        Args:
            image: RGB格式图像
            frame_id: 帧ID

        Returns:
            DetectionResult检测结果
        """
        start_time = time.perf_counter()

        # 执行推理 - 只检测person类(class 0)
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.input_size,
            device=self.device,
            classes=[0],  # person only
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detection_list = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            image_h, image_w = image.shape[:2]

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()

                x1, y1, x2, y2 = xyxy
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                relative_area = (w * h) / (image_w * image_h)

                detection_list.append(
                    PedestrianDetection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        score=float(conf),
                        center=[float(cx), float(cy)],
                        width=float(w),
                        height=float(h),
                        relative_area=float(relative_area),
                    )
                )

        return DetectionResult(
            frame_id=frame_id,
            detection_list=detection_list,
            num_detections=len(detection_list),
            elapsed_ms=elapsed_ms,
            success=True,
            image_shape=image.shape,
            has_pedestrian=len(detection_list) > 0,
        )

    def _detect_rtdetr(
        self,
        image: np.ndarray,
        frame_id: int,
    ) -> DetectionResult:
        """
        RT-DETR检测器执行

        Args:
            image: RGB格式图像
            frame_id: 帧ID

        Returns:
            DetectionResult检测结果
        """
        from PIL import Image

        start_time = time.perf_counter()

        pil_image = Image.fromarray(image)
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [pil_image.size[::-1]],
            device=self.device
        )  # (h, w)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence,
            target_sizes=target_sizes,
        )[0]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detection_list = []
        image_h, image_w = image.shape[:2]

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            label_name = self.model.config.id2label[int(label)]
            if label_name != "person":
                continue

            x1, y1, x2, y2 = box.cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            relative_area = (w * h) / (image_w * image_h)

            detection_list.append(
                PedestrianDetection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    score=float(score),
                    center=[float(cx), float(cy)],
                    width=float(w),
                    height=float(h),
                    relative_area=float(relative_area),
                )
            )

        return DetectionResult(
            frame_id=frame_id,
            detection_list=detection_list,
            num_detections=len(detection_list),
            elapsed_ms=elapsed_ms,
            success=True,
            image_shape=image.shape,
            has_pedestrian=len(detection_list) > 0,
        )

    def format_for_brain(self, result: DetectionResult) -> Dict[str, Any]:
        """
        将检测结果格式化为适合Brain模型输入的字典

        Args:
            result: 检测结果对象

        Returns:
            包含行人信息的字典，可直接拼接到prompt中
        """
        if result.num_detections == 0:
            return {
                "pedestrian_detected": False,
                "pedestrian_count": 0,
                "pedestrian_info": "No pedestrians detected",
                "spatial_info": "",
                "warning_level": "safe",  # safe, caution, danger
            }

        # 计算警告等级
        max_area = max(d.relative_area for d in result.detection_list)
        warning_level = "safe"
        if max_area > 0.15:
            warning_level = "danger"
        elif max_area > 0.05:
            warning_level = "caution"

        # 空间位置分析
        spatial_parts = []
        for det in result.detection_list:
            cx, cy = det.center
            # 归一化到[0, 1]
            cx_norm, cy_norm = cx / result.image_shape[1], cy / result.image_shape[0]

            if cx_norm < 0.33:
                h_pos = "Left"
            elif cx_norm < 0.67:
                h_pos = "Center"
            else:
                h_pos = "Right"

            if cy_norm < 0.4:
                v_pos = "Far"
            elif cy_norm < 0.7:
                v_pos = "Medium"
            else:
                v_pos = "Near"

            spatial_parts.append(
                f"Person #{det.bbox} (Confidence:{det.score:.2f}, Position:{h_pos}{v_pos}, "
                f"Relative Area:{det.relative_area:.3f})"
            )

        return {
            "pedestrian_detected": True,
            "pedestrian_count": result.num_detections,
            "pedestrian_info": "; ".join(spatial_parts),
            "spatial_info": f"Detected {result.num_detections} pedestrian(s)",
            "warning_level": warning_level,
            "detection_latency_ms": result.elapsed_ms,
            "raw_detections": [
                {
                    "bbox": d.bbox,
                    "confidence": d.score,
                    "relative_area": d.relative_area,
                }
                for d in result.detection_list
            ],
        }

    def cleanup(self) -> None:
        """
        清理检测器资源

        释放GPU内存，删除模型引用
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """
        析构函数，确保资源释放
        """
        self.cleanup()


class PedestrianDetectionManager:
    """
    行人检测管理器
    ===============

    统一管理多个行人检测实例，支持批量处理和缓存。
    支持同步/异步检测模式。

    该管理器用于训练/评估时的检测任务协调，
    可以在多个worker间共享检测器实例。

    Attributes:
        detector: 主检测器实例
        batch_size: 批处理大小
        cache_enabled: 是否启用结果缓存
        async_enabled: 是否启用异步检测
        _executor: 异步检测的线程池
    """

    def __init__(
        self,
        enabled: bool = True,
        detector_type: str = "yolov8n",
        device: str = "cuda",
        confidence: float = 0.25,
        async_enabled: bool = False,
        **kwargs,
    ):
        """
        初始化检测管理器

        Args:
            enabled: 是否启用检测功能
            detector_type: 检测器类型
            device: 计算设备
            confidence: 置信度阈值
            async_enabled: 是否启用异步检测模式
            **kwargs: 其他传递给PedestrianDetector的参数
        """
        self.enabled = enabled
        self.async_enabled = async_enabled
        self.detector: Optional[PedestrianDetector] = None
        self._executor = None
        self._pending_futures: Dict[int, Any] = {}  # frame_id -> Future
        self._results_cache: Dict[int, Dict[str, Any]] = {}  # frame_id -> result

        # ========== 耗时统计 ==========
        self._total_detection_time: float = 0.0  # 总检测耗时（秒）
        self._detection_count: int = 0  # 检测次数
        self._skipped_count: int = 0  # 跳过检测次数（使用缓存）

        if self.enabled:
            self.detector = PedestrianDetector(
                detector_type=detector_type,
                device=device,
                confidence=confidence,
                **kwargs,
            )

            # 为异步检测创建线程池
            if self.async_enabled:
                from concurrent.futures import ThreadPoolExecutor
                # 使用2个线程，允许并行检测和主训练循环
                self._executor = ThreadPoolExecutor(max_workers=2)

    def detect_frame(
        self,
        image: np.ndarray,
        frame_id: int = 0,
    ) -> Dict[str, Any]:
        """
        检测单帧图像（同步模式）

        Args:
            image: 输入图像
            frame_id: 帧ID

        Returns:
            格式化后的检测结果字典（包含 detection_time_ms 字段）
        """
        import time
        start_time = time.perf_counter()

        if not self.enabled or self.detector is None:
            return {
                "pedestrian_detected": False,
                "pedestrian_count": 0,
                "pedestrian_info": "Pedestrian detection disabled",
                "warning_level": "disabled",
                "detection_time_ms": 0.0,
            }

        result = self.detector.detect(image, frame_id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 统计耗时
        self._total_detection_time += elapsed_ms / 1000.0  # 转换为秒
        self._detection_count += 1

        formatted = self.detector.format_for_brain(result)
        formatted["detection_time_ms"] = elapsed_ms
        return formatted

    def _detect_in_thread(
        self,
        image: np.ndarray,
        frame_id: int,
    ) -> Dict[str, Any]:
        """在线程中执行检测"""
        import time
        start_time = time.perf_counter()

        if self.detector is None:
            return {
                "pedestrian_detected": False,
                "pedestrian_count": 0,
                "pedestrian_info": "Detector not initialized",
                "warning_level": "error",
                "detection_time_ms": 0.0,
            }
        result = self.detector.detect(image, frame_id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 统计耗时（异步模式下在提交时记录）
        formatted = self.detector.format_for_brain(result)
        formatted["detection_time_ms"] = elapsed_ms
        return formatted

    def detect_async_submit(
        self,
        image: np.ndarray,
        frame_id: int = 0,
    ) -> None:
        """
        提交异步检测任务（非阻塞）

        将检测任务提交到线程池，不等待结果。
        结果需要通过 detect_async_wait() 获取。

        Args:
            image: 输入图像
            frame_id: 帧ID
        """
        if not self.enabled or self.detector is None:
            return

        if self._executor is not None:
            # 提交到线程池
            future = self._executor.submit(self._detect_in_thread, image.copy(), frame_id)
            self._pending_futures[frame_id] = future
        else:
            # 如果没有线程池，直接执行
            result = self._detect_in_thread(image, frame_id)
            self._results_cache[frame_id] = result

    def detect_async_wait(
        self,
        frame_id: int,
        timeout: float = 0.0,
    ) -> Dict[str, Any]:
        """
        等待并获取异步检测结果（阻塞）

        Args:
            frame_id: 帧ID
            timeout: 等待超时（秒），0表示不等待

        Returns:
            格式化后的检测结果字典（包含 detection_time_ms 字段）
        """
        import time
        wait_start = time.perf_counter()

        if not self.enabled or self.detector is None:
            return {
                "pedestrian_detected": False,
                "pedestrian_count": 0,
                "pedestrian_info": "Pedestrian detection disabled",
                "warning_level": "disabled",
                "detection_time_ms": 0.0,
            }

        # 先检查缓存（同步模式的结果）
        if frame_id in self._results_cache:
            return self._results_cache.pop(frame_id)

        # 检查pending futures
        if frame_id in self._pending_futures:
            future = self._pending_futures.pop(frame_id)
            try:
                result = future.result(timeout=timeout if timeout > 0 else None)
                # 记录等待时间
                wait_time_ms = (time.perf_counter() - wait_start) * 1000.0
                result["wait_time_ms"] = wait_time_ms
                # 统计实际检测耗时（由线程记录在 result 中）
                if "detection_time_ms" in result:
                    self._total_detection_time += result["detection_time_ms"] / 1000.0
                    self._detection_count += 1
                return result
            except Exception as e:
                print(f"[AsyncDetection] Error waiting for frame {frame_id}: {e}")
                return {
                    "pedestrian_detected": False,
                    "pedestrian_count": 0,
                    "pedestrian_info": f"Async error: {str(e)[:50]}",
                    "warning_level": "error",
                    "detection_time_ms": 0.0,
                }

        # 如果没有pending结果，返回默认值
        return {
            "pedestrian_detected": False,
            "pedestrian_count": 0,
            "pedestrian_info": "No pending detection",
            "warning_level": "unknown",
            "detection_time_ms": 0.0,
        }

    def detect_async_wait_all(
        self,
        frame_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        等待多个帧的检测结果

        Args:
            frame_ids: 帧ID列表

        Returns:
            frame_id -> 检测结果的字典
        """
        results = {}
        for fid in frame_ids:
            results[fid] = self.detect_async_wait(fid)
        return results

    def is_available(self) -> bool:
        """
        检查检测器是否可用

        Returns:
            bool: 检测器是否已初始化且可用
        """
        return self.enabled and self.detector is not None

    def get_stats(self) -> Dict[str, Any]:
        """
        获取检测统计信息

        Returns:
            包含统计信息的字典：
            - total_time_s: 总检测耗时（秒）
            - detection_count: 检测次数
            - avg_time_ms: 平均检测耗时（毫秒）
            - skipped_count: 跳过检测次数
            - cache_hit_rate: 缓存命中率
        """
        avg_time_ms = (self._total_detection_time / self._detection_count * 1000.0
                       if self._detection_count > 0 else 0.0)
        total_frames = self._detection_count + self._skipped_count
        cache_hit_rate = (self._skipped_count / total_frames * 100.0
                          if total_frames > 0 else 0.0)

        return {
            "total_time_s": self._total_detection_time,
            "detection_count": self._detection_count,
            "avg_time_ms": avg_time_ms,
            "skipped_count": self._skipped_count,
            "cache_hit_rate": cache_hit_rate,
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._total_detection_time = 0.0
        self._detection_count = 0
        self._skipped_count = 0

    def shutdown(self) -> None:
        """
        关闭检测管理器，释放所有资源
        """
        # 关闭线程池
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        # 清理pending futures
        self._pending_futures.clear()
        self._results_cache.clear()

        # 关闭检测器
        if self.detector is not None:
            self.detector.cleanup()
            self.detector = None
