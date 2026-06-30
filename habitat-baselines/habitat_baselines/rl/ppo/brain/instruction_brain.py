# -*- coding: utf-8 -*-
"""指令优化Brain模块 - 检测到行人时优化导航指令"""
import base64
import copy
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from habitat import logger


def normalize_instruction(instruction: str) -> str:
    """规范化指令字符串，移除多余的换行符和空格
    
    Args:
        instruction: 原始指令字符串
        
    Returns:
        规范化后的指令字符串
    """
    if not instruction:
        return ""
    # 移除换行符，替换为空格
    result = instruction.replace('\n', ' ')
    # 移除多余的空格（连续空格替换为单个空格）
    result = re.sub(r'\s+', ' ', result)
    # 移除首尾空格
    result = result.strip()
    return result

class BrainCallMode(Enum):
    """Brain调用模式"""
    LOCAL_HF = "local_hf"  # 本地HuggingFace transformers调用（默认）
    LOCAL_API = "local_api"  # 本地API服务器（如vLLM、TGI）
    REMOTE_API = "remote_api"  # 远程API（如QWen API、OpenAI API）

class BrainModelType(Enum):
    # Gemma系列
    GEMMA4_E2B = "gemma4_e2b"
    GEMMA4_E4B = "gemma4_e4b"
    # Qwen3-VL系列（视觉语言模型）
    QWEN3_VL_2B = "qwen3_vl_2b"
    QWEN3_VL_4B = "qwen3_vl_4b"
    QWEN3_VL_8B = "qwen3_vl_8b"
    QWEN3_VL_32B = "qwen3_vl_32b"
    QWEN3_VL = "qwen3_vl"  # 别名
    # Qwen3-VL-FP8系列（视觉语言模型 FP8量化版）
    QWEN3_VL_2B_FP8 = "qwen3_vl_2b_fp8"
    QWEN3_VL_4B_FP8 = "qwen3_vl_4b_fp8"
    QWEN3_VL_8B_FP8 = "qwen3_vl_8b_fp8"
    # Qwen3.5系列（纯文本语言模型）
    QWEN3_0_6B = "qwen3_0_6b"
    QWEN3_5_0_8B = "qwen3_5_0_8b"
    QWEN3_5_2B = "qwen3_5_2b"
    QWEN3_5_4B = "qwen3_5_4b"
    QWEN3_5_9B = "qwen3_5_9b"
    QWEN3_5_27B = "qwen3_5_27b"
    QWEN3_5_35B = "qwen3_5_35b"  # 可选扩展
    # Qwen2.5-VL系列
    QWEN2_5_VL_3B = "qwen2_5_vl_3b"
    QWEN2_5_VL_7B = "qwen2_5_vl_7b"
    QWEN2_5_VL_72B = "qwen2_5_vl_72b"
    # LLaVA系列
    LLAVA_V1_5_7B = "llava_v1_5_7b"
    LLAVA_V1_6_7B = "llava_v1_6_7b"
    LLAVA_NEXT_7B = "llava_next_7b"
    LLAVA_NEXT_34B = "llava_next_34b"
    # GLM-V系列（智谱AI视觉语言模型）
    GLM_4_6V = "glm_4_6v"
    GLM_4_6V_FLASH = "glm_4_6v_flash"
    GLM_4_6V_FP8 = "glm_4_6v_fp8"
    GLM_4_5V = "glm_4_5v"
    GLM_4_5V_FP8 = "glm_4_5v_fp8"
    GLM_4_1V_9B_THINKING = "glm_4_1v_9b_thinking"
    GLM_4_1V_9B_BASE = "glm_4_1v_9b_base"
    # Gemini系列（Google视觉语言模型）
    GEMINI_2_0_FLASH = "gemini_2_0_flash"
    GEMINI_2_0_FLASH_LITE = "gemini_2_0_flash_lite"
    GEMINI_2_5_FLASH = "gemini_2_5_flash"
    GEMINI_2_5_PRO = "gemini_2_5_pro"
    GEMINI_2_5_FLASH_LITE = "gemini_2_5_flash_lite"
    DISABLED = "disabled"

class InstructionModifier(Enum):
    ORIGINAL = "original"
    APPEND_WARNING = "append_warning"
    REPLACE = "replace"
    ADD_HINT = "add_hint"

@dataclass
class FrameRecord:
    frame_id: int
    image_path: Optional[str] = None
    image_array: Optional[Any] = None
    pedestrian_detected: bool = False
    pedestrian_count: int = 0
    pedestrian_info: str = ""
    pedestrian_bbox: Optional[List[List[float]]] = None  # 行人边界框 [[x1,y1,x2,y2], ...]
    pedestrian_trajectory: Optional[List[Dict]] = None  # 历史行人轨迹 [{"bbox": [...], "relative_pos": (...)}]
    action: Optional[str] = None
    action_id: int = 1  # 动作ID: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT, 4=WAIT, 5=BACKWARD
    instruction: str = ""
    timestamp: float = 0.0

@dataclass
class InstructionOptimizationResult:
    original_instruction: str = ""
    optimized_instruction: str = ""
    modifier_type: InstructionModifier = InstructionModifier.ORIGINAL
    confidence: float = 0.0
    reasoning: str = ""
    safety_level: str = "normal"
    pedestrian_warning: bool = False
    warning_message: str = ""
    should_modify: bool = False
    raw_response: str = ""
    inference_time_ms: float = 0.0  # 推理耗时（毫秒）

@dataclass
class EpisodeRecord:
    episode_id: str
    start_time: str
    original_instruction: str
    frames: List[FrameRecord] = field(default_factory=list)
    instruction_modifications: List[Dict] = field(default_factory=list)
    total_frames: int = 0
    frames_with_pedestrian: int = 0
    brain_calls: int = 0
    instruction_modifications_count: int = 0

class InstructionBrain:
    """指令优化Brain - 帧级记录 + 条件触发 + 指令优化"""
    
    # 类级别定义属性
    @property
    def frame_history(self) -> List['FrameRecord']:
        """返回当前episode的frame_history（向后兼容）"""
        if self.current_episode_id and self.current_episode_id in self._frame_histories:
            return self._frame_histories[self.current_episode_id]
        return []
    
    def get_frame_history(self, episode_id: str) -> List['FrameRecord']:
        """获取指定episode的frame_history
        
        Args:
            episode_id: episode ID
            
        Returns:
            指定episode的帧历史列表
        """
        if episode_id and episode_id in self._frame_histories:
            return self._frame_histories[episode_id]
        return []
    
    def get_env_episode_id(self, env_idx: int) -> Optional[str]:
        """获取指定环境的当前episode_id
        
        Args:
            env_idx: 环境索引
            
        Returns:
            该环境的当前episode_id，如果不存在则返回None
        """
        return self._env_current_episode_ids.get(env_idx)
    
    def resolve_episode_id(self, episode_id: Optional[str] = None, env_idx: Optional[int] = None) -> Optional[str]:
        """解析episode_id，优先使用传入的episode_id，否则尝试从env_idx获取
        
        优先级：
        1. 如果提供了episode_id，直接使用
        2. 如果提供了env_idx，从_env_current_episode_ids获取
        3. 如果都没有，使用current_episode_id
        
        Args:
            episode_id: 指定的episode_id
            env_idx: 环境索引
            
        Returns:
            解析后的episode_id
        """
        if episode_id is not None:
            return episode_id
        if env_idx is not None and env_idx in self._env_current_episode_ids:
            return self._env_current_episode_ids[env_idx]
        return self.current_episode_id
    
    MODEL_IDS = {
        # Gemma系列
        BrainModelType.GEMMA4_E2B: "google/gemma-4-E2B", 
        BrainModelType.GEMMA4_E4B: "google/gemma-4-E4B", 
        # Qwen3-VL系列（视觉语言模型）  420 图片和信息不保存 prompt zeroshot e4/2
        BrainModelType.QWEN3_VL_2B: "Qwen/Qwen3-VL-2B-Instruct",
        BrainModelType.QWEN3_VL_4B: "Qwen/Qwen3-VL-4B-Instruct",
        BrainModelType.QWEN3_VL_8B: "Qwen/Qwen3-VL-8B-Instruct",
        BrainModelType.QWEN3_VL_32B: "Qwen/Qwen3-VL-32B-Instruct",
        BrainModelType.QWEN3_VL: "Qwen/Qwen3-VL-8B-Instruct",  # 默认使用8B
        # Qwen3-VL-FP8系列（FP8量化版，来自ModelScope）
        BrainModelType.QWEN3_VL_2B_FP8: "Qwen/Qwen3-VL-2B-Instruct-FP8",
        BrainModelType.QWEN3_VL_4B_FP8: "Qwen/Qwen3-4B-Instruct-2507-FP8",
        BrainModelType.QWEN3_VL_8B_FP8: "Qwen/Qwen3-VL-8B-Instruct-FP8",
        # Qwen3.5系列（纯文本语言模型）
        BrainModelType.QWEN3_0_6B: "Qwen/Qwen3-0.6B",
        BrainModelType.QWEN3_5_0_8B: "Qwen/Qwen3.5-0.8B",
        BrainModelType.QWEN3_5_2B: "Qwen/Qwen3.5-2B",
        BrainModelType.QWEN3_5_4B: "Qwen/Qwen3.5-4B",
        BrainModelType.QWEN3_5_9B: "Qwen/Qwen3.5-9B",
        BrainModelType.QWEN3_5_27B: "Qwen/Qwen3.5-27B",
        BrainModelType.QWEN3_5_35B: "Qwen/Qwen3.5-35B",
        # Qwen2.5-VL系列
        BrainModelType.QWEN2_5_VL_3B: "Qwen/Qwen2.5-VL-3B-Instruct",
        BrainModelType.QWEN2_5_VL_7B: "Qwen/Qwen2.5-VL-7B-Instruct",
        BrainModelType.QWEN2_5_VL_72B: "Qwen/Qwen2.5-VL-72B-Instruct",
        # LLaVA
        BrainModelType.LLAVA_V1_5_7B: "llava-hf/llava-v1.5-7b-hf",
        BrainModelType.LLAVA_V1_6_7B: "llava-hf/llava-v1.6-mistral-7b-hf",
        BrainModelType.LLAVA_NEXT_7B: "llava-hf/llava-next-7b-hf",
        BrainModelType.LLAVA_NEXT_34B: "llava-hf/llava-next-34b-hf",
        # GLM-V系列（智谱AI视觉语言模型）
        BrainModelType.GLM_4_6V: "zai-org/GLM-4.6V", 
        BrainModelType.GLM_4_6V_FLASH: "zai-org/GLM-4.6V-Flash", 
        BrainModelType.GLM_4_6V_FP8: "zai-org/GLM-4.6V-FP8", 
        BrainModelType.GLM_4_5V: "zai-org/GLM-4.5V",
        BrainModelType.GLM_4_5V_FP8: "zai-org/GLM-4.5V-FP8",
        BrainModelType.GLM_4_1V_9B_THINKING: "zai-org/GLM-4.1V-9B-Thinking",
        BrainModelType.GLM_4_1V_9B_BASE: "zai-org/GLM-4.1V-9B-Base",
        # Gemini系列（Google视觉语言模型）
        BrainModelType.GEMINI_2_0_FLASH: "gemini-2.0-flash",
        BrainModelType.GEMINI_2_0_FLASH_LITE: "gemini-2.0-flash-lite",
        BrainModelType.GEMINI_2_5_FLASH: "gemini-2.5-flash",
        BrainModelType.GEMINI_2_5_PRO: "gemini-2.5-pro", 
        BrainModelType.GEMINI_2_5_FLASH_LITE: "gemini-2.5-flash-lite",
    }

    def __init__(self, model_type: str = "qwen3_vl", device: str = "cuda", model_id: Optional[str] = None,
                 model_path: Optional[str] = None, enable_reasoning: bool = True, max_history_frames: int = 5,
                 max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9,
                 cache_dir: Optional[str] = None, torch_dtype: str = "bfloat16",
                 save_frames: bool = True, output_dir: str = "./brain_records",
                 log_prompt: bool = True, save_prompt_to_file: bool = True,
                 save_frame_images: bool = True, frame_images_root: str = "./brain_records/frame_images",
                 # API调用相关参数
                 call_mode: str = "local_hf",  # 调用模式: local_hf, local_api, remote_api
                 api_base_url: Optional[str] = None,  # API服务器地址，如 "http://localhost:8000/v1"
                 api_model_name: Optional[str] = None,  # API模型名称，如 "Qwen/Qwen3-VL-8B-Instruct"
                 api_api_key: Optional[str] = None,  # API密钥，远程API需要
                 brain_call_confidence: float = 0.7,  # 调用Brain的置信度阈值
                 num_actions: int = 6,  # 动作空间大小: 4 (STOP/FORWARD/LEFT/RIGHT) 或 6 (含WAIT/BACKWARD)
                 min_brain_call_interval: int = 3,  # Brain调用最小间隔（帧），避免对几乎相同的帧重复调用VLM
                 # 定期调用Brain参数（即使无行人检测）
                 call_brain_periodically: bool = False,  # 是否定期调用Brain（适用于无行人训练环境）
                 call_brain_interval: int = 10,  # 定期调用间隔（帧）
                 ):
        self.model_type = BrainModelType(model_type.lower())
        self.model_id = model_id  # HuggingFace模型ID
        self.model_path = model_path  # 本地模型路径
        self.device = torch.device("cuda") if device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
        self.enable_reasoning = enable_reasoning
        self.max_history_frames = max_history_frames
        self.cache_dir = cache_dir
        self.save_frames = save_frames
        self.output_dir = output_dir
        self.model = None
        self.processor = None

        # API调用相关
        self.call_mode = BrainCallMode(call_mode.lower())
        self.api_base_url = api_base_url
        self.api_model_name = api_model_name
        self.api_api_key = api_api_key or "EMPTY"
        self.api_client = None

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self.generation_config = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "do_sample": temperature > 0}

        # VLM generate() 超时设置（秒）：防止模型推理挂起导致所有 rank NCCL watchdog 超时
        # 设置为 NCCL_TIMEOUT 的约 1/4，留足余地进入 _update_agent()
        self._vlm_generation_timeout = float(os.environ.get("VLM_GENERATION_TIMEOUT", 120.0))
        self.episode_records: Dict[str, EpisodeRecord] = {}
        self.current_episode_id: Optional[str] = None
        # 修复：改为按episode_id索引的字典，支持多环境并行
        self._frame_histories: Dict[str, List[FrameRecord]] = {}
        
        # 行人轨迹历史也改为按episode索引
        self._pedestrian_trajectory_histories: Dict[str, List[Dict]] = {}
        
        # 多环境并行支持：为每个env维护独立的current_episode_id
        # key: env_idx (int), value: episode_id (str)
        self._env_current_episode_ids: Dict[int, str] = {}

        # Zero-Shot评估专用: Prompt打印和保存配置
        self.log_prompt = log_prompt
        self.save_prompt_to_file = save_prompt_to_file
        self.save_frame_images = save_frame_images
        self.frame_images_root = frame_images_root

        # Brain调用置信度阈值
        self.brain_call_confidence = brain_call_confidence

        # Brain调用节流：避免对连续帧重复调用VLM
        self.min_brain_call_interval = min_brain_call_interval
        self._last_brain_call_frame: Dict[str, int] = {}  # episode_id -> last frame_id when brain was called
        self._last_brain_call_result: Dict[str, InstructionOptimizationResult] = {}  # episode_id -> cached result

        # 定期调用Brain（即使无行人检测），适用于无行人训练环境
        self.call_brain_periodically = call_brain_periodically
        self.call_brain_interval = call_brain_interval
        self._last_periodic_call_frame: Dict[str, int] = {}  # episode_id -> last frame_id for periodic call

        # 动作空间大小: 4 (STOP/FORWARD/LEFT/RIGHT) 或 6 (含WAIT/BACKWARD)
        self.num_actions = num_actions

        # 在初始化时创建所有需要的目录
        self._create_directories()

        # 统计
        self._brain_call_count: int = 0
        self._total_inference_time: float = 0.0  # 总推理耗时（秒）
        self._total_wait_time: float = 0.0  # 总等待耗时（秒）
        self._inference_count: int = 0  # 推理次数
        self._empty_cache_counter: int = 0  # 周期性CUDA缓存清理计数器

        self._initialize_model(model_id)

    def _create_directories(self) -> None:
        """创建所有需要的目录"""
        # 创建主输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        # 创建帧图像根目录
        if self.save_frame_images:
            os.makedirs(self.frame_images_root, exist_ok=True)
        # prompt记录目录在保存时创建（_save_prompt_to_file）
        # episode记录目录在保存时创建（save_episode_record）

    def _initialize_model(self, model_id: Optional[str] = None) -> None:
        if self.model_type == BrainModelType.DISABLED:
            return
        
        # API模式不需要加载本地模型
        if self.call_mode in (BrainCallMode.LOCAL_API, BrainCallMode.REMOTE_API):
            self._init_api_client()
            return
        
        try:
            if self._is_vision_model():
                self._init_vision_model(model_id)
            else:
                self._init_text_model(model_id)
        except Exception as e:
            print(f"[InstructionBrain] Model loading failed: {e}")
            self.model = None
            self.processor = None

    def _init_api_client(self) -> None:
        """初始化API客户端"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要安装openai库 (pip install openai)")
        
        # 确定API基础URL
        if self.call_mode == BrainCallMode.REMOTE_API:
            # 远程API需要用户指定
            if not self.api_base_url:
                raise ValueError("Remote API mode requires api_base_url")
            # API密钥必须有
            if not self.api_api_key or self.api_api_key == "EMPTY":
                raise ValueError("Remote API mode requires valid api_api_key")
        else:
            # 本地API
            if not self.api_base_url:
                # 默认本地地址
                self.api_base_url = "http://localhost:8000/v1"
        
        # 确定模型名称
        if not self.api_model_name:
            self.api_model_name = self.model_id or self.MODEL_IDS.get(self.model_type, "Qwen/Qwen3-VL-8B-Instruct")
        
        self.api_client = OpenAI(
            api_key=self.api_api_key,
            base_url=self.api_base_url
        )
        
        print(f"[InstructionBrain] API client initialized:")
        print(f"  - Call mode: {self.call_mode.value}")
        print(f"  - Base URL: {self.api_base_url}")
        print(f"  - Model name: {self.api_model_name}")

    def _is_vision_model(self) -> bool:
        """判断是否为视觉语言模型"""
        vision_models = [
            # Qwen3-VL系列
            BrainModelType.QWEN3_VL, BrainModelType.QWEN3_VL_2B,
            BrainModelType.QWEN3_VL_4B, BrainModelType.QWEN3_VL_8B,
            BrainModelType.QWEN3_VL_32B,
            # Qwen3-VL-FP8系列
            BrainModelType.QWEN3_VL_2B_FP8, BrainModelType.QWEN3_VL_4B_FP8,
            BrainModelType.QWEN3_VL_8B_FP8,
            # Qwen2.5-VL系列
            BrainModelType.QWEN2_5_VL_3B, BrainModelType.QWEN2_5_VL_7B,
            BrainModelType.QWEN2_5_VL_72B,
            # LLaVA系列
            BrainModelType.LLAVA_V1_6_7B,
            BrainModelType.LLAVA_V1_5_7B,
            BrainModelType.LLAVA_NEXT_7B,
            BrainModelType.LLAVA_NEXT_34B,
            # GLM-V系列（智谱AI视觉语言模型）
            BrainModelType.GLM_4_6V,
            BrainModelType.GLM_4_6V_FLASH,
            BrainModelType.GLM_4_6V_FP8,
            BrainModelType.GLM_4_5V,
            BrainModelType.GLM_4_5V_FP8,
            BrainModelType.GLM_4_1V_9B_THINKING,
            BrainModelType.GLM_4_1V_9B_BASE,
            # Gemma系列是纯文本模型，不在视觉模型列表中
            BrainModelType.GEMMA4_E2B,
            BrainModelType.GEMMA4_E4B,
            # Gemini系列（Google视觉语言模型）
            BrainModelType.GEMINI_2_0_FLASH,
            BrainModelType.GEMINI_2_0_FLASH_LITE,
            BrainModelType.GEMINI_2_5_FLASH,
            BrainModelType.GEMINI_2_5_PRO,
            BrainModelType.GEMINI_2_5_FLASH_LITE,
        ]
        return self.model_type in vision_models

    def _is_gemini_model(self) -> bool:
        """判断是否为Gemini模型"""
        gemini_models = [
            BrainModelType.GEMINI_2_0_FLASH,
            BrainModelType.GEMINI_2_0_FLASH_LITE,
            BrainModelType.GEMINI_2_5_FLASH,
            BrainModelType.GEMINI_2_5_PRO,
            BrainModelType.GEMINI_2_5_FLASH_LITE,
        ]
        return self.model_type in gemini_models

    def _init_vision_model(self, model_id: Optional[str] = None) -> None:
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError:
            raise ImportError("需要安装transformers库 (pip install transformers)")

        # 优先使用本地路径，其次使用传入的model_id，最后使用MODEL_IDS中的默认值
        if self.model_path and os.path.exists(self.model_path):
            actual_model_id = self.model_path
            print(f"[InstructionBrain] Loading vision model from local: {actual_model_id}")
        elif self.model_path:
            # 路径可能不存在，打印警告但仍然尝试
            print(f"[InstructionBrain] Warning: Local path does not exist, trying HuggingFace: {self.model_path}")
            actual_model_id = model_id or self.model_id or self.MODEL_IDS.get(self.model_type, "Qwen/Qwen3-VL-8B-Instruct")
            print(f"[InstructionBrain] Trying to load from HuggingFace: {actual_model_id}")
        else:
            actual_model_id = model_id or self.model_id or self.MODEL_IDS.get(self.model_type, "Qwen/Qwen3-VL-8B-Instruct")
            print(f"[InstructionBrain] Loading vision model from HuggingFace: {actual_model_id}")

        print(f"[InstructionBrain] Model config:")
        print(f"  - Actual model ID: {actual_model_id}")
        print(f"  - Model type: {self.model_type.value}")
        print(f"  - Cache dir: {self.cache_dir}")
        print(f"  - Data type: {self.torch_dtype}")

        self.processor = AutoProcessor.from_pretrained(
            actual_model_id, 
            cache_dir=self.cache_dir, 
            trust_remote_code=True
        )
        # 修复：在分布式训练中使用明确的设备映射，避免NCCL通信超时
        # device_map="auto" 可能在多GPU环境下导致模型层分布在不同GPU，造成集体通信超时
        import torch.distributed as dist
        if dist.is_initialized():
            # 分布式训练：使用当前进程的本地设备
            local_rank = dist.get_rank() % torch.cuda.device_count()
            self.model = AutoModelForVision2Seq.from_pretrained(
                actual_model_id, 
                cache_dir=self.cache_dir, 
                torch_dtype=self.torch_dtype, 
                device_map={"": f"cuda:{local_rank}"}, 
                trust_remote_code=True
            )
        else:
            # 非分布式训练：可以使用 device_map="auto"
            self.model = AutoModelForVision2Seq.from_pretrained(
                actual_model_id, 
                cache_dir=self.cache_dir, 
                torch_dtype=self.torch_dtype, 
                device_map="auto", 
                trust_remote_code=True
            )
        self.model.eval()
        print(f"[InstructionBrain] Vision model ({self.model_type.value}) loaded successfully")

    def _init_text_model(self, model_id: Optional[str] = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("需要安装transformers库")

        if self.model_path and os.path.exists(self.model_path):
            actual_model_id = self.model_path
            print(f"[InstructionBrain] Loading text model from local: {actual_model_id}")
        else:
            actual_model_id = model_id or self.model_id or self.MODEL_IDS.get(self.model_type, "google/gemma-4-E2B")
            print(f"[InstructionBrain] Loading text model from HuggingFace: {actual_model_id}")

        self.processor = AutoTokenizer.from_pretrained(actual_model_id, cache_dir=self.cache_dir)
        # 修复：在分布式训练中使用明确的设备映射，避免NCCL通信超时
        import torch.distributed as dist
        if dist.is_initialized():
            # 分布式训练：使用当前进程的本地设备
            local_rank = dist.get_rank() % torch.cuda.device_count()
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_id, 
                cache_dir=self.cache_dir, 
                torch_dtype=self.torch_dtype, 
                device_map={"": f"cuda:{local_rank}"}
            )
        else:
            # 非分布式训练：可以使用 device_map="auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_id, 
                cache_dir=self.cache_dir, 
                torch_dtype=self.torch_dtype, 
                device_map="auto"
            )
        self.model.eval()

    def start_episode(self, episode_id: str, original_instruction: str, env_idx: Optional[int] = None) -> None:
        """开始新的episode，确保状态正确转换

        Args:
            episode_id: episode ID
            original_instruction: 原始指令
            env_idx: 环境索引（可选）。如果提供，会在_env_current_episode_ids中记录，
                     用于支持多环境并行训练时区分不同环境的episode。
        """
        # 不自动结束之前的episode！
        # 在多环境并行训练中，不同环境可能有不同的活跃episode
        # 结束episode应该由trainer显式调用end_episode()，而不是在这里自动结束
        
        # 规范化原始指令：移除换行符和多余空格
        normalized_instruction = normalize_instruction(original_instruction)
        
        # 检查是否已存在该episode的记录
        if episode_id in self.episode_records:
            existing_ep = self.episode_records[episode_id]
            # 如果已有记录的原始指令不同，说明确实是不同的episode
            if existing_ep.original_instruction != normalized_instruction:
                # 先保存旧记录，避免数据丢失
                if len(existing_ep.frames) > 0:
                    old_path = self.save_episode_record(episode_id)
                    logger.info(f"[InstructionBrain] Episode {episode_id} already exists ({len(existing_ep.frames)} frames, "
                              f"instruction changed), auto-saved old record to: {old_path}")
                else:
                    logger.info(f"[InstructionBrain] Episode {episode_id} already exists (0 frames), "
                              f"discarding empty old record")
                # 清理旧状态的缓存
                if episode_id in self._frame_histories:
                    del self._frame_histories[episode_id]
                if episode_id in self._pedestrian_trajectory_histories:
                    del self._pedestrian_trajectory_histories[episode_id]
                if episode_id in self._last_brain_call_frame:
                    del self._last_brain_call_frame[episode_id]
                if episode_id in self._last_brain_call_result:
                    del self._last_brain_call_result[episode_id]
                # 重新创建新的episode记录
                self.episode_records[episode_id] = EpisodeRecord(
                    episode_id=episode_id,
                    start_time=datetime.now().isoformat(),
                    original_instruction=normalized_instruction
                )
                logger.info(f"[InstructionBrain] Episode {episode_id} re-created (instruction changed, old record saved)")
            else:
                # 相同的episode_id和相同的指令，可能是重复调用，直接使用现有记录
                logger.info(f"[InstructionBrain] Episode {episode_id} already exists, reusing existing record")
        else:
            # 创建新的episode记录
            self.episode_records[episode_id] = EpisodeRecord(
                episode_id=episode_id,
                start_time=datetime.now().isoformat(),
                original_instruction=normalized_instruction
            )
            logger.info(f"[InstructionBrain] Created new episode record for {episode_id}")
        
        # 更新全局current_episode_id（保持向后兼容）
        self.current_episode_id = episode_id
        
        # 如果提供了env_idx，更新该env的current_episode_id映射
        if env_idx is not None:
            self._env_current_episode_ids[env_idx] = episode_id
            logger.info(f"[InstructionBrain] Set env{env_idx} current_episode_id = {episode_id}")
        
        # 确保frame_history存在
        if episode_id not in self._frame_histories:
            self._frame_histories[episode_id] = []
        if episode_id not in self._pedestrian_trajectory_histories:
            self._pedestrian_trajectory_histories[episode_id] = []
        
        logger.info(f"[InstructionBrain] Started episode {episode_id}: {normalized_instruction[:80]}...")

    def record_frame(self, frame_id: int, image: np.ndarray, action: str, instruction: str, 
                    pedestrian_info: Dict[str, Any], image_path: Optional[str] = None, 
                    action_id: int = 1, episode_id: Optional[str] = None, env_idx: Optional[int] = None) -> None:
        """记录帧数据
        
        Args:
            frame_id: 帧ID
            image: 图像数据
            action: 当前动作
            instruction: 当前指令
            pedestrian_info: 行人检测信息
            image_path: 图像路径（可选）
            action_id: 动作ID（可选，默认1=FORWARD）
            episode_id: 指定的episode ID（可选）。如果为None，则使用current_episode_id。
                        推荐传入明确的episode_id以支持多环境并行训练。
            env_idx: 环境索引（可选）。如果提供，会优先使用该环境对应的episode_id。
                     这是在多环境并行训练中正确区分不同环境episode的关键参数。
        
        注意：如果指定的episode_id不存在于records中，说明episode尚未开始，会静默跳过。
        确保在调用此方法前已调用start_episode()。
        """
        # 调试日志：记录record_frame被调用（已注释）
        # if not hasattr(self, '_record_frame_entry_count'):
        #     self._record_frame_entry_count = 0
        # self._record_frame_entry_count += 1
        # if self._record_frame_entry_count <= 30:
        #     env_ep_id = self._env_current_episode_ids.get(env_idx) if env_idx is not None else None
        #     logger.info(f"[InstructionBrain.record_frame] ENTRY #{self._record_frame_entry_count}: "
        #                f"episode_id={episode_id}, env_idx={env_idx}, env_episode_id={env_ep_id}, "
        #                f"current_episode_id={self.current_episode_id}, "
        #                f"frame_id={frame_id}, image={'None' if image is None else f'shape={image.shape}'}")
        
        # 优先使用env_idx对应的episode_id（如果提供），其次使用传入的episode_id，最后使用current_episode_id
        target_episode_id = self.resolve_episode_id(episode_id, env_idx)
        
        if target_episode_id is None:
            # 调试：跟踪因episode_id为None而跳过的帧数
            # 只在首次跳过时记录，避免日志刷屏
            if not hasattr(self, '_frame_skip_count'):
                self._frame_skip_count = 0
                self._frame_skip_first_episode_id = None
            self._frame_skip_count += 1
            
            # 如果跳过的帧数超过10帧，说明可能存在bug（正常情况应该很少）
            if self._frame_skip_count > 10:
                # 重置计数器，避免无限警告
                skip_ep_id = getattr(self, '_frame_skip_first_episode_id', 'unknown')
                logger.warning(
                    f"[InstructionBrain] record_frame skipped {self._frame_skip_count} times "
                    f"(first episode_id was: {skip_ep_id}). "
                    f"This may indicate a synchronization issue between trainer and brain. "
                    f"Check that start_episode() is called before record_frame()."
                )
                self._frame_skip_count = 0  # 重置计数器
            return
        
        # 获取目标episode的记录
        episode = self.episode_records.get(target_episode_id)
        if episode is None:
            # 重要：这是帧没有被记录的主要原因！
            # 检查 current_episode_id 和 records 中的 keys
            current_id = self.current_episode_id
            records_keys = list(self.episode_records.keys())
            
            logger.warning(f"[RecordFrame-ERROR] Episode '{target_episode_id}' not found in episode_records!")
            logger.warning(f"  - target_episode_id: {target_episode_id}")
            logger.warning(f"  - current_episode_id: {current_id}")
            logger.warning(f"  - episode_records keys: {records_keys}")
            
            # 详细检查：是不是被覆盖了？
            for key in records_keys:
                ep = self.episode_records[key]
                logger.warning(f"    Episode '{key}': {len(ep.frames)} frames, start_time={ep.start_time}")
            return
        
        # 调试日志：记录record_frame调用（已注释）
        # if not hasattr(self, '_record_frame_call_count'):
        #     self._record_frame_call_count = 0
        # self._record_frame_call_count += 1

        # 获取episode对象
        episode = self.episode_records.get(target_episode_id)

        # 调试日志（已注释）
        # if self._record_frame_call_count <= 30 or self._record_frame_call_count % 100 == 0:
        #     logger.info(f"[InstructionBrain] record_frame #{self._record_frame_call_count}: episode={target_episode_id}, frame_id={frame_id}, "
        #                f"episode={'FOUND' if episode else 'NOT FOUND'}, "
        #                f"episode.frames count={len(episode.frames) if episode else 'N/A'}, "
        #                f"episode.total_frames={episode.total_frames if episode else 'N/A'}, "
        #                f"all_records_keys={list(self.episode_records.keys())}")
        
        # 提取行人边界框
        pedestrian_bbox = None
        raw_detections = pedestrian_info.get("raw_detections", [])
        if raw_detections:
            pedestrian_bbox = [det.get("bbox", []) for det in raw_detections]
        
        # 获取目标episode的frame_history
        frame_history = self._frame_histories.get(target_episode_id, [])
        
        # 更新行人轨迹历史
        if pedestrian_info.get("pedestrian_detected", False) and raw_detections:
            traj_entry = {
                "frame_id": frame_id,
                "bboxes": pedestrian_bbox,
                "count": pedestrian_info.get("pedestrian_count", 0),
            }
            self._pedestrian_trajectory_histories.setdefault(target_episode_id, []).append(traj_entry)
            # 保持最多10帧的历史
            traj_hist = self._pedestrian_trajectory_histories.get(target_episode_id, [])
            if len(traj_hist) > 10:
                self._pedestrian_trajectory_histories[target_episode_id] = traj_hist[-10:]
        
        # 获取当前episode的原始指令（用于无行人时返回）
        episode = self.episode_records.get(target_episode_id) if target_episode_id else None
        original_instr = episode.original_instruction if episode else instruction
        
        # 规范化指令：移除换行符和多余空格
        normalized_instruction = normalize_instruction(instruction)
        
        frame_record = FrameRecord(
            frame_id=frame_id,
            image_path=image_path,
            # 始终存储图像用于VLM历史帧（内存由max_history_frames限制，每帧~196KB）
            # 当frame_history超过max_history_frames时，旧帧被丢弃并自动GC
            # end_episode()时整个history被清理，确保内存不会泄漏
            image_array=image.copy() if image is not None else None,
            pedestrian_detected=pedestrian_info.get("pedestrian_detected", False),
            pedestrian_count=pedestrian_info.get("pedestrian_count", 0),
            pedestrian_info=self._format_pedestrian_info(pedestrian_info, original_instr),
            pedestrian_bbox=pedestrian_bbox,
            pedestrian_trajectory=self._pedestrian_trajectory_histories.get(target_episode_id, []).copy() if target_episode_id in self._pedestrian_trajectory_histories else None,
            action=action, 
            action_id=action_id,
            instruction=normalized_instruction,  # 使用规范化后的指令
            timestamp=time.time()
        )
        frame_history.append(frame_record)
        if len(frame_history) > self.max_history_frames:
            self._frame_histories[target_episode_id] = frame_history[-self.max_history_frames:]
        else:
            self._frame_histories[target_episode_id] = frame_history
        episode = self.episode_records.get(target_episode_id)
        if episode:
            # 调试日志（已注释）
            # if not hasattr(self, '_last_episode_id'):
            #     self._last_episode_id = {}
            # last_ep = self._last_episode_id.get(target_episode_id)
            # if last_ep is not None and id(episode) != last_ep:
            #     logger.warning(f"[RecordFrame-WARN] Episode {target_episode_id} object changed! "
            #                  f"old_id={id(last_ep)}, new_id={id(episode)}")
            # self._last_episode_id[target_episode_id] = id(episode)

            # 记录调用前的 frames 长度
            frames_before = len(episode.frames)
            episode.frames.append(frame_record)
            episode.total_frames += 1

            # 调试日志（已注释）
            # if not hasattr(self, '_frame_record_debug'):
            #     self._frame_record_debug = {}
            # key = f"{target_episode_id}_frames"
            # if key not in self._frame_record_debug:
            #     self._frame_record_debug[key] = 0
            # self._frame_record_debug[key] += 1
            # if self._frame_record_debug[key] <= 30 or self._frame_record_debug[key] % 50 == 0:
            #     logger.info(f"[RecordFrame-Debug] Episode {target_episode_id}: frame #{self._frame_record_debug[key]} recorded, "
            #               f"frames_before={frames_before}, frames_after={len(episode.frames)}, total_frames={episode.total_frames}, "
            #               f"episode_obj_id={id(episode)}")
            if pedestrian_info.get("pedestrian_detected"):
                episode.frames_with_pedestrian += 1
        else:
            logger.warning(f"[RecordFrame-WARN] Episode {target_episode_id} not found in episode_records! "
                         f"Available keys: {list(self.episode_records.keys())}")

    def _format_pedestrian_info(self, ped_info: Dict[str, Any], original_instruction: str = "") -> str:
        """格式化行人信息用于显示和LLM输入

        无行人时返回简洁的 "No pedestrians detected"，
        有行人时返回包含人数、位置和置信度的描述性文本。
        此方法用于JSON保存的pedestrian_info字段和VLM prompt。

        Args:
            ped_info: 行人检测信息字典
            original_instruction: 原始指令（已废弃，不再使用）
        """
        if not ped_info.get("pedestrian_detected", False):
            return "No pedestrians detected"

        # 有行人时：构建紧凑的单行描述文本
        ped_count = ped_info.get("pedestrian_count", 0)
        warning_level = ped_info.get("warning_level", "unknown")
        raw_detections = ped_info.get("raw_detections", [])

        parts = [f"{ped_count} pedestrian(s) detected"]

        if raw_detections:
            # 添加每个行人的简要位置和置信度信息
            det_parts = []
            for i, det in enumerate(raw_detections):
                bbox = det.get("bbox", [])
                conf = det.get("confidence", 0)
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    if cx < 85:
                        h_pos = "L"
                    elif cx < 170:
                        h_pos = "C"
                    else:
                        h_pos = "R"
                    det_parts.append(f"P{i+1}:{h_pos}({conf:.0%})")
                else:
                    det_parts.append(f"P{i+1}:{conf:.0%}")
            if det_parts:
                parts.append("[" + ", ".join(det_parts) + "]")

        if warning_level and warning_level != "unknown":
            parts.append(f"level:{warning_level}")

        return " ".join(parts)

    def should_call_brain(self, pedestrian_info: Dict[str, Any], episode_id: Optional[str] = None,
                          frame_id: int = -1, force: bool = False) -> bool:
        """判断是否应该调用brain模型（检查行人检测、置信度和定期调用）

        规则：
        1. 有行人检测 + 置信度足够 → 调用
        2. 启用了定期调用 + 距上次调用 ≥ call_brain_interval → 调用
        3. force=True → 强制调用

        注意：节流（throttling）由 optimize_instruction() 内部处理，
        节流期间会返回缓存的优化结果。

        Args:
            pedestrian_info: 行人检测信息
            episode_id: episode ID（用于检查定期调用状态）
            frame_id: 当前帧ID（用于定期调用判断）
            force: 是否强制调用（忽略所有阈值）
        """
        if not pedestrian_info.get("pedestrian_detected", False):
            # 无行人时，检查是否需要定期调用
            if force:
                return True
            if self.call_brain_periodically and episode_id and frame_id >= 0:
                last_periodic = self._last_periodic_call_frame.get(episode_id, -self.call_brain_interval)
                if (frame_id - last_periodic) >= self.call_brain_interval:
                    return True
            return False

        # 置信度检查：只有最高置信度 ≥ brain_call_confidence 才调用VLM
        # YOLO 用 pedestrian_confidence (0.25) 做初筛，brain_call_confidence (0.7) 决定是否真调用
        if not force:
            raw_detections = pedestrian_info.get("raw_detections", [])
            if raw_detections:
                max_conf = max((d.get("confidence", 0) for d in raw_detections), default=0)
                if max_conf < self.brain_call_confidence:
                    # 置信度不足，但仍检查是否需要定期调用
                    if self.call_brain_periodically and episode_id and frame_id >= 0:
                        last_periodic = self._last_periodic_call_frame.get(episode_id, -self.call_brain_interval)
                        if (frame_id - last_periodic) >= self.call_brain_interval:
                            return True
                    return False

        return True

    @torch.no_grad()
    def optimize_instruction(self, original_instruction: str, current_frame: np.ndarray, history_frames: Optional[List[FrameRecord]] = None,
                             pedestrian_info: Optional[Dict[str, Any]] = None, episode_id: Optional[str] = None,
                             env_idx: Optional[int] = None, frame_id: int = -1) -> InstructionOptimizationResult:
        """优化指令

        Args:
            original_instruction: 原始指令
            current_frame: 当前帧图像
            history_frames: 历史帧列表
            pedestrian_info: 行人检测信息
            episode_id: 指定的episode ID（可选）。如果为None，则使用current_episode_id。
            env_idx: 环境索引（可选）。如果提供，会优先使用该环境对应的episode_id。
                     这是在多环境并行训练中正确区分不同环境episode的关键参数。
            frame_id: 当前帧ID（用于节流判断，-1表示不启用节流）

        Returns:
            InstructionOptimizationResult，包含 inference_time_ms 字段表示推理耗时（毫秒）
        """
        import time
        call_start_time = time.perf_counter()

        # 使用resolve_episode_id解析正确的episode_id（优先env_idx，其次episode_id，最后current_episode_id）
        target_episode_id = self.resolve_episode_id(episode_id, env_idx)

        # 无行人时：检查是否需要定期调用
        has_pedestrian = pedestrian_info and pedestrian_info.get("pedestrian_detected", False)
        if not has_pedestrian:
            # 定期调用检查：即使无行人，每隔 call_brain_interval 帧也调用VLM做指令优化
            if self.call_brain_periodically and target_episode_id and frame_id >= 0:
                last_periodic = self._last_periodic_call_frame.get(target_episode_id, -self.call_brain_interval)
                if (frame_id - last_periodic) >= self.call_brain_interval:
                    # 记录定期调用帧号，然后继续往下执行VLM调用
                    self._last_periodic_call_frame[target_episode_id] = frame_id
                    # 不return，继续执行VLM调用逻辑
                else:
                    return InstructionOptimizationResult(
                        original_instruction=original_instruction,
                        optimized_instruction=original_instruction,
                        modifier_type=InstructionModifier.ORIGINAL,
                        confidence=1.0,
                        reasoning="No pedestrians in view, using original instruction",
                        should_modify=False,
                        inference_time_ms=0.0
                    )
            else:
                return InstructionOptimizationResult(
                    original_instruction=original_instruction,
                    optimized_instruction=original_instruction,
                    modifier_type=InstructionModifier.ORIGINAL,
                    confidence=1.0,
                    reasoning="No pedestrians in view, using original instruction",
                    should_modify=False,
                    inference_time_ms=0.0
                )

        # 置信度不足时：返回原指令（YOLO检出了但置信度不够高，不值得调用VLM）
        # 但如果定期调用到期，仍然允许调用
        raw_detections = pedestrian_info.get("raw_detections", []) if pedestrian_info else []
        if raw_detections and has_pedestrian:
            max_conf = max((d.get("confidence", 0) for d in raw_detections), default=0)
            if max_conf < self.brain_call_confidence:
                # 定期调用检查
                if self.call_brain_periodically and target_episode_id and frame_id >= 0:
                    last_periodic = self._last_periodic_call_frame.get(target_episode_id, -self.call_brain_interval)
                    if (frame_id - last_periodic) >= self.call_brain_interval:
                        self._last_periodic_call_frame[target_episode_id] = frame_id
                        # 不return，继续执行VLM调用
                    else:
                        return InstructionOptimizationResult(
                            original_instruction=original_instruction,
                            optimized_instruction=original_instruction,
                            modifier_type=InstructionModifier.ORIGINAL,
                            confidence=1.0,
                            reasoning=f"Pedestrian confidence ({max_conf:.2f}) below threshold ({self.brain_call_confidence})",
                            should_modify=False,
                            inference_time_ms=0.0
                        )
                else:
                    return InstructionOptimizationResult(
                        original_instruction=original_instruction,
                        optimized_instruction=original_instruction,
                        modifier_type=InstructionModifier.ORIGINAL,
                        confidence=1.0,
                        reasoning=f"Pedestrian confidence ({max_conf:.2f}) below threshold ({self.brain_call_confidence})",
                        should_modify=False,
                        inference_time_ms=0.0
                    )

        # 节流检查：距上次调用不足 min_brain_call_interval 帧时，返回缓存的优化结果
        # 这是关键修复：之前 should_call_brain() 包含了节流检查并返回 False，
        # 导致在上面的早期返回中直接返回了原始指令。现在节流检查移到这里，
        # 确保在节流窗口内返回的是缓存的上一次优化结果，而不是原始指令。
        if target_episode_id and self.min_brain_call_interval > 0 and frame_id >= 0:
            last_frame = self._last_brain_call_frame.get(target_episode_id, -1)
            if (frame_id - last_frame) < self.min_brain_call_interval:
                cached = self._last_brain_call_result.get(target_episode_id)
                if cached is not None:
                    # 返回缓存的优化结果（不实际调用VLM，节省推理时间）
                    # 使用 dataclasses.replace 避免修改缓存中的原始 inference_time_ms
                    from dataclasses import replace as _dc_replace
                    return _dc_replace(cached, inference_time_ms=0.0)
                # 如果没有缓存结果（首次遇到行人但还没调用过VLM），
                # 继续往下执行实际的VLM调用

        episode = self.episode_records.get(target_episode_id) if target_episode_id else None
        if episode:
            episode.brain_calls += 1

        result = None

        # API模式调用
        if self.call_mode in (BrainCallMode.LOCAL_API, BrainCallMode.REMOTE_API):
            if self.api_client is None:
                result = self._fallback_optimization(original_instruction, pedestrian_info)
                self._record_modification(episode, original_instruction, result)
            else:
                try:
                    result = self._optimize_with_api(original_instruction, current_frame, history_frames, pedestrian_info, episode_id=target_episode_id)
                    self._record_modification(episode, original_instruction, result)
                except Exception as e:
                    print(f"[InstructionBrain] API call failed: {e}")
                    result = self._fallback_optimization(original_instruction, pedestrian_info)
                    self._record_modification(episode, original_instruction, result)
        # 本地HuggingFace模式
        # 模型不可用时使用回退逻辑
        elif self.model is None or self.processor is None:
            result = self._fallback_optimization(original_instruction, pedestrian_info)
            self._record_modification(episode, original_instruction, result)
        else:
            try:
                if self._is_vision_model():
                    result = self._optimize_with_vision_model(original_instruction, current_frame, history_frames, pedestrian_info, episode_id=target_episode_id)
                else:
                    result = self._optimize_with_text_model(original_instruction, self._format_pedestrian_info(pedestrian_info, original_instruction) if pedestrian_info else "", episode_id=target_episode_id)
                self._record_modification(episode, original_instruction, result)
            except Exception as e:
                print(f"[InstructionBrain] Optimization failed: {e}")
                result = self._fallback_optimization(original_instruction, pedestrian_info)
                self._record_modification(episode, original_instruction, result)

        # 记录耗时并缓存结果
        if result is not None:
            elapsed_ms = (time.perf_counter() - call_start_time) * 1000.0
            result.inference_time_ms = elapsed_ms

            # 累计统计
            self._total_inference_time += elapsed_ms / 1000.0  # 转换为秒
            self._inference_count += 1

            # 缓存优化结果，避免对连续帧重复调用VLM
            if target_episode_id and self.min_brain_call_interval > 0 and frame_id >= 0:
                self._last_brain_call_frame[target_episode_id] = frame_id
                self._last_brain_call_result[target_episode_id] = result

            # 同步更新定期调用计数器（VLM被实际调用时，重置定期调用计时器）
            if target_episode_id and frame_id >= 0:
                self._last_periodic_call_frame[target_episode_id] = frame_id

            # GPU内存清理：每50次VLM推理后释放未使用的缓存（避免全局CUDA同步阻塞NCCL操作）
            if self.device.type == "cuda":
                self._empty_cache_counter += 1
                if self._empty_cache_counter % 50 == 0:
                    torch.cuda.empty_cache()

        return result

    def _record_modification(self, episode: Optional[EpisodeRecord], original: str, result: InstructionOptimizationResult) -> None:
        if episode is None or not result.should_modify:
            return
        episode.instruction_modifications.append({"original": original, "optimized": result.optimized_instruction,
                                                   "modifier_type": result.modifier_type.value, "confidence": result.confidence, "safety_level": result.safety_level})
        episode.instruction_modifications_count += 1

    def _optimize_with_api(self, original_instruction: str, current_frame: np.ndarray, history_frames: Optional[List[FrameRecord]],
                           pedestrian_info: Optional[Dict[str, Any]], episode_id: Optional[str] = None) -> InstructionOptimizationResult:
        """通过API调用进行指令优化（支持Gemini和其他OpenAI兼容API）"""
        from PIL import Image
        import io
        
        self._brain_call_count += 1
        # 使用传入的episode_id或current_episode_id
        target_episode_id = episode_id if episode_id is not None else self.current_episode_id
        call_id = f"ep{target_episode_id}_call{self._brain_call_count}"
        
        # 解码原始指令
        decoded_instruction = self._decode_instruction_for_display(original_instruction)
        
        # 仅在episode结束时保存调试图像（通过配置控制）
        # 历史帧图像使用内存中的 image_array，不再每次调用都保存到磁盘
        if self.save_frame_images and current_frame is not None:
            # 仅保存带标注的当前帧图像用于调试
            if pedestrian_info and pedestrian_info.get("pedestrian_detected", False):
                annotated_image = self._save_annotated_image(current_frame, call_id, pedestrian_info)
        
        # 构建消息
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        # 构建用户消息（支持图像）
        user_content = []
        
        # 添加历史帧图像（使用内存中的 image_array，不保存到磁盘）
        if history_frames:
            for frame in history_frames[-self.max_history_frames:]:
                if frame.image_array is not None:
                    hist_img = Image.fromarray(frame.image_array)
                    # 将图像转为base64
                    buffered = io.BytesIO()
                    hist_img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                    })
                # 添加历史帧的文本信息
                frame_info = f"[History Frame {frame.frame_id}] Action: {frame.action or 'N/A'}, Pedestrian: {frame.pedestrian_info}"
                user_content.append({"type": "text", "text": frame_info})
        
        # 添加当前帧图像
        if current_frame is not None:
            img_pil = Image.fromarray(current_frame)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })
        
        # 添加上下文信息（无行人时返回原始指令，有行人时返回空字符串让LLM自主生成）
        # 检查是否有行人
        has_pedestrian = pedestrian_info and pedestrian_info.get("pedestrian_detected", False)
        ped_count = pedestrian_info.get("pedestrian_count", 0) if has_pedestrian else 0
        
        # 动作空间上下文（根据 num_actions 动态生成）
        action_context = self._get_action_context_text()
        context_text = (
            f"=== NAVIGATION TASK ===\n"
            f"Original Instruction: {decoded_instruction}\n"
            f"\n=== PEDESTRIAN DETECTION ===\n"
            f"Status: {'Pedestrian detected' if has_pedestrian else 'No pedestrians detected'}\n"
            f"Count: {ped_count}\n"
            f"\n=== TASK ===\n"
            f"Based on the images and original instruction, generate an appropriate navigation instruction.\n"
            f"If there are pedestrians, consider safety and generate modified instructions that avoid them.\n"
            f"If no pedestrians, you can follow or simplify the original instruction.\n"
            f"\n{action_context}"
        )
        user_content.append({"type": "text", "text": context_text})
        messages.append({"role": "user", "content": user_content})
        
        # 构建增强的上下文信息（包含历史轨迹）
        enhanced_context = self._build_trajectory_context(history_frames, pedestrian_info)
        
        # 将增强上下文添加到user message中
        user_content.append({"type": "text", "text": enhanced_context})
        
        # 构建完整prompt文本（用于打印和保存）
        # 注意：历史帧图像路径不再传递，因为现在使用内存中的图像数组
        full_prompt_text = self._build_full_prompt_text(
            original_instruction=decoded_instruction,
            history_frames=history_frames,
            pedestrian_info=pedestrian_info,
            history_image_paths=[],  # 空列表，图像信息通过 frame.image_array 提供
            call_id=call_id,
        )
        
        # 打印prompt（如果启用）
        if self.log_prompt:
            self._print_prompt(full_prompt_text, call_id, decoded_instruction, history_frames, pedestrian_info)
        
        # 保存prompt到文件（如果启用）
        if self.save_prompt_to_file:
            self._save_prompt_to_file(full_prompt_text, call_id)
        
        # 调用API
        try:
            if self._is_gemini_model():
                # Gemini特定调用方式（使用中转API服务）
                response = self._call_gemini_api(messages)
            else:
                # 标准OpenAI兼容API调用
                chat_response = self.api_client.chat.completions.create(
                    model=self.api_model_name,
                    messages=messages,
                    temperature=self.generation_config.get("temperature", 0.7),
                    max_tokens=self.generation_config.get("max_new_tokens", 512),
                )
                response = chat_response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")
        
        # 记录模型响应（已注释）
        # response_len = len(response)
        # response_bytes = len(response.encode('utf-8'))
        # logger.info(f"[Brain-Response] Call #{self._brain_call_count}: {response_len} chars / {response_bytes} bytes")
        # logger.info(f"[Brain-Response Full] {response}")
        
        return self._parse_optimization_response(decoded_instruction, response, pedestrian_info)

    def _call_gemini_api(self, messages: List[Dict]) -> str:
        """
        调用Gemini API（通过OpenAI兼容中转服务）
        
        参考 gemini.py 中的实现，使用 /v1/chat/completions 端点
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # 构建content（兼容OpenAI格式）
        user_message = messages[-1]  # 最后一条是user消息
        content = user_message.get("content", [])
        
        # 构建请求体
        body = {
            "model": self.api_model_name,
            "messages": messages,
            "temperature": self.generation_config.get("temperature", 0.7),
            "max_tokens": self.generation_config.get("max_new_tokens", 512),
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        
        # 构建URL
        base_url = self.api_base_url.rstrip('/') if self.api_base_url else "https://api.openai120.com"
        url = f"{base_url}/v1/chat/completions"
        
        # 创建带重试的session
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        try:
            resp = session.post(url, headers=headers, json=body, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            
            # 提取响应内容
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            elif "candidates" in data and len(data["candidates"]) > 0:
                # Gemini原生格式
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                raise RuntimeError(f"Unexpected API response format: {data}")
        finally:
            session.close()

    def _generate_with_timeout(self, inputs, generation_kwargs):
        """
        带超时的 model.generate() 调用。

        使用 daemon thread 检测超时。如果生成时间超过超时限制，
        抛出 TimeoutError 让调用者使用 fallback 响应。

        NOTE: Python daemon thread 不能真正杀死正在运行的 CUDA kernel，
        但可以防止调用者无限期阻塞，从而使 rank 能及时进入 NCCL 同步点。
        残存的 CUDA kernel 将在后续 CUDA 同步时被 Fix 7 的错误检查捕获。
        """
        import queue
        import threading

        result_queue = queue.Queue(maxsize=1)
        exception_queue = queue.Queue(maxsize=1)

        def _generate_worker():
            try:
                outputs = self.model.generate(**inputs, **generation_kwargs)
                result_queue.put(outputs)
            except Exception as e:
                exception_queue.put(e)

        worker = threading.Thread(target=_generate_worker, daemon=True)
        worker.start()
        worker.join(timeout=self._vlm_generation_timeout)

        if worker.is_alive():
            raise TimeoutError(
                f"VLM generate() exceeded {self._vlm_generation_timeout}s timeout. "
                "Returning fallback response to avoid NCCL watchdog cascade."
            )

        # 检查 worker 是否有异常
        if not exception_queue.empty():
            raise exception_queue.get_nowait()

        if not result_queue.empty():
            return result_queue.get_nowait()

        raise TimeoutError("VLM generate() completed but produced no output.")

    def _optimize_with_vision_model(self, original_instruction: str, current_frame: np.ndarray, history_frames: Optional[List[FrameRecord]],
                                   pedestrian_info: Optional[Dict[str, Any]], episode_id: Optional[str] = None) -> InstructionOptimizationResult:
        from PIL import Image

        self._brain_call_count += 1
        # 使用传入的episode_id或current_episode_id
        target_episode_id = episode_id if episode_id is not None else self.current_episode_id
        call_id = f"ep{target_episode_id}_call{self._brain_call_count}"

        # 解码原始指令（如果是字节数组）
        decoded_instruction = self._decode_instruction_for_display(original_instruction)

        # 仅在episode结束时保存调试图像（通过配置控制）
        # 历史帧图像使用内存中的 image_array，不再每次调用都保存到磁盘
        if self.save_frame_images and current_frame is not None:
            # 仅保存带标注的当前帧图像用于调试
            if pedestrian_info and pedestrian_info.get("pedestrian_detected", False):
                annotated_image = self._save_annotated_image(current_frame, call_id, pedestrian_info)
            else:
                annotated_image = None

        messages = [{"role": "system", "content": self._build_system_prompt()}]
        user_content = []

        # 构建历史帧信息（使用内存中的 image_array，不保存到磁盘）
        if history_frames:
            for frame in history_frames[-self.max_history_frames:]:
                if frame.image_array is not None:
                    hist_img = Image.fromarray(frame.image_array)
                    user_content.append({"type": "image", "image": hist_img})
                # 添加历史帧的文本信息（使用简洁格式）
                if frame.pedestrian_detected:
                    ped_info = f"{frame.pedestrian_count} pedestrian(s)"
                else:
                    ped_info = "no pedestrian"
                frame_info = f"[History Frame {frame.frame_id}] Action: {frame.action or 'N/A'}, Pedestrian: {ped_info}"
                user_content.append({"type": "text", "text": frame_info})

        current_img = Image.fromarray(current_frame)
        user_content.append({"type": "image", "image": current_img})

        # 检查行人检测状态（与API路径保持一致的格式）
        has_pedestrian = pedestrian_info and pedestrian_info.get("pedestrian_detected", False)
        ped_count = pedestrian_info.get("pedestrian_count", 0) if has_pedestrian else 0

        # 动作空间上下文（根据 num_actions 动态生成）
        action_context = self._get_action_context_text()
        context_text = (
            f"=== NAVIGATION TASK ===\n"
            f"Original Instruction: {decoded_instruction}\n"
            f"\n=== PEDESTRIAN DETECTION ===\n"
            f"Status: {'Pedestrian detected' if has_pedestrian else 'No pedestrians detected'}\n"
            f"Count: {ped_count}\n"
            f"\n{action_context}"
        )
        user_content.append({"type": "text", "text": context_text})
        messages.append({"role": "user", "content": user_content})
        
        # 构建增强的上下文信息（包含历史轨迹）
        enhanced_context = self._build_trajectory_context(history_frames, pedestrian_info)
        
        # 将增强上下文添加到messages中（作为额外的user消息）
        messages.append({"role": "user", "content": enhanced_context})

        # 构建完整prompt文本（用于打印和保存）
        # 注意：历史帧图像路径不再传递，因为现在使用内存中的图像数组
        full_prompt_text = self._build_full_prompt_text(
            original_instruction=decoded_instruction,
            history_frames=history_frames,
            pedestrian_info=pedestrian_info,
            history_image_paths=[],  # 空列表，图像信息通过 frame.image_array 提供
            call_id=call_id,
        )

        # 打印prompt（如果启用）
        if self.log_prompt:
            self._print_prompt(full_prompt_text, call_id, decoded_instruction, history_frames, pedestrian_info)

        # 保存prompt到文件（如果启用）
        if self.save_prompt_to_file:
            self._save_prompt_to_file(full_prompt_text, call_id)

        # 调用模型推理
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # 从user_content中提取所有图像
        images_for_processor = []
        for item in user_content:
            if item.get("type") == "image" and "image" in item:
                images_for_processor.append(item["image"])
        
        inputs = self.processor(text=text, images=images_for_processor, return_tensors="pt", padding=True).to(self.device)
        try:
            outputs = self._generate_with_timeout(
                inputs,
                dict(**self.generation_config, pad_token_id=self.processor.tokenizer.eos_token_id)
            )
        except TimeoutError as e:
            logger.warning(f"[InstructionBrain] {e}")
            return self._fallback_optimization(original_instruction, pedestrian_info)

        # 安全地解码生成的文本
        # outputs形状: [batch_size, seq_len] 或 [batch_size, num_return_sequences, seq_len]
        # inputs["input_ids"]形状: [batch_size, input_len] 或 [input_len] (单样本)
        input_ids_shape = inputs["input_ids"].shape

        # 计算输入序列长度
        if len(input_ids_shape) == 1:
            # 一维情况：单个序列 [seq_len]
            input_len = input_ids_shape[0]
        else:
            # 二维或更高：使用序列长度维度（通常是dim 1）
            input_len = input_ids_shape[1]

        # 获取生成的token IDs
        # outputs[0] 取第一个batch的第一个生成结果
        generated_ids = outputs[0]
        # 如果outputs是三维的 [batch, num_seq, seq_len]，需要额外索引
        if len(generated_ids.shape) > 1:
            generated_ids = generated_ids[0]

        # 只解码新生成的部分（跳过输入的prompt部分）
        # 注意：generate()的输出可能已经包含输入，取决于模型和参数
        # 大多数情况下，outputs只包含新生成的token
        # 但如果outputs包含完整序列(input+generated)，需要切片
        if len(generated_ids) > input_len:
            # 如果生成的序列比输入长，说明outputs包含了input+generated
            response = self.processor.decode(generated_ids[input_len:], skip_special_tokens=True)
        else:
            # outputs只包含生成的token，直接全部解码
            response = self.processor.decode(generated_ids, skip_special_tokens=True)

        # 记录模型响应长度（已注释）
        # response_len = len(response)
        # response_bytes = len(response.encode('utf-8'))
        # logger.info(f"[Brain-Response] Call #{self._brain_call_count}: {response_len} chars / {response_bytes} bytes")
        # 完整打印响应（已注释）
        # logger.info(f"[Brain-Response Full] {response}")

        return self._parse_optimization_response(original_instruction, response, pedestrian_info)

    def _build_full_prompt_text(
        self,
        original_instruction: str,
        history_frames: Optional[List[FrameRecord]],
        pedestrian_info: Optional[Dict[str, Any]],
        history_image_paths: List[str],
        call_id: str,
    ) -> str:
        """构建完整的prompt文本（用于打印和保存）"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"[Brain Call ID]: {call_id}")
        lines.append("=" * 80)

        # 系统提示
        lines.append("\n【System Prompt】")
        lines.append(self._build_system_prompt())

        # 动作约束说明（根据 num_actions 动态生成）
        lines.append("\n【Robot Action Constraints】")
        action_lines = self._get_action_full_prompt_lines()
        lines.extend(action_lines)

        # 历史帧轨迹信息（包括行人检测和动作）
        lines.append("\n[Navigation Trajectory History]")
        lines.append(f"Recent {len(history_frames[-self.max_history_frames:]) if history_frames else 0} frames (max {self.max_history_frames} frames):")
        if history_frames:
            recent_frames = history_frames[-self.max_history_frames:]
            # 构建轨迹格式
            trajectory_parts = []
            for i, frame in enumerate(recent_frames):
                # 动作标签
                action_label = self._action_id_to_label(frame.action_id)
                # 行人状态
                ped_status = "✓ pedestrians" if frame.pedestrian_detected else "○ no pedestrian"
                trajectory_parts.append(f"Frame {frame.frame_id}: {action_label}, {ped_status}")
            
            # 使用轨迹格式输出
            lines.append("\n  Trajectory Summary:")
            lines.append(f"    {' → '.join(trajectory_parts)}")
            
            # 详细历史帧信息
            lines.append("\n  Detailed Frame Information:")
            for i, frame in enumerate(recent_frames):
                # 从 frame 对象判断是否有图像数据
                has_image = frame.image_array is not None
                img_path = history_image_paths[i] if i < len(history_image_paths) and history_image_paths[i] else ("(in-memory)" if has_image else "N/A")
                action_label = self._action_id_to_label(frame.action_id)
                
                lines.append(f"\n    [Frame {frame.frame_id}]")
                lines.append(f"      Action: {action_label}")
                lines.append(f"      Pedestrian: {frame.pedestrian_info}")
                if frame.pedestrian_bbox:
                    lines.append(f"      BBox: {frame.pedestrian_bbox}")
                lines.append(f"      Image: {img_path} {'(base64 in API call)' if has_image else ''}")
        else:
            lines.append("  No history frames available.")

        # 当前帧信息
        lines.append("\n[Current Frame Info]")
        # 显示原始指令字符串（如果是指令数组则解码）
        display_instruction = self._decode_instruction_for_display(original_instruction)
        ped_info_str = self._format_pedestrian_info(pedestrian_info, display_instruction) if pedestrian_info else 'None'
        annotated_img_path = os.path.join(self.frame_images_root, f"{call_id}_annotated.jpg")
        lines.append(f"  - Original Instruction: {display_instruction}")
        lines.append(f"  - Pedestrian Detection: {ped_info_str}")
        # 标注图像仅在有行人时保存
        lines.append(f"  - Annotated Frame Image: {annotated_img_path}" if pedestrian_info and pedestrian_info.get("pedestrian_detected") else f"  - Annotated Frame Image: N/A (no pedestrian)")
        
        # 添加图像说明
        lines.append("\n[Image Description]")
        lines.append("  The robot's current first-person view from its onboard camera.")
        lines.append("  - Image dimensions: Same as current frame")
        lines.append("  - Content: Indoor scene with possible pedestrians highlighted with bounding boxes")
        lines.append("  - Bounding box format: [x1, y1, x2, y2] representing top-left and bottom-right corners")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    
    def _action_id_to_label(self, action_id: int) -> str:
        """将动作ID转换为可读标签（根据 num_actions 动态调整）"""
        if self.num_actions == 4:
            action_labels = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
        else:
            action_labels = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT", 4: "WAIT", 5: "BACKWARD"}
        return action_labels.get(action_id, f"UNKNOWN_{action_id}")

    def _build_trajectory_context(
        self,
        history_frames: Optional[List[FrameRecord]],
        pedestrian_info: Optional[Dict[str, Any]],
    ) -> str:
        """
        构建导航轨迹上下文信息，包含历史帧的动作和行人状态
        
        Returns:
            格式化的轨迹字符串，描述最近几帧的动作和行人状态
        """
        if not history_frames:
            return ""
        
        recent_frames = history_frames[-self.max_history_frames:]
        
        lines = []
        lines.append("\n=== NAVIGATION TRAJECTORY CONTEXT ===")
        lines.append(f"Recent {len(recent_frames)} frames trajectory:")
        
        # 构建轨迹摘要
        trajectory_parts = []
        for frame in recent_frames:
            action_label = self._action_id_to_label(frame.action_id)
            if frame.pedestrian_detected:
                ped_info = f"[{frame.pedestrian_count} pedestrian(s)]"
            else:
                ped_info = "[no pedestrian]"
            trajectory_parts.append(f"{action_label}{ped_info}")
        
        lines.append("  " + " → ".join(trajectory_parts))
        
        # 详细帧信息
        lines.append("\nFrame-by-frame details:")
        for i, frame in enumerate(recent_frames):
            action_label = self._action_id_to_label(frame.action_id)
            lines.append(f"  Frame {frame.frame_id}: {action_label}")
            if frame.pedestrian_detected:
                # 使用简洁格式显示行人信息
                lines.append(f"    - Pedestrian detected: {frame.pedestrian_count} person(s)")
                if frame.pedestrian_bbox:
                    lines.append(f"    - BBox: {frame.pedestrian_bbox}")
            else:
                lines.append(f"    - No pedestrian detected")
        
        # 当前帧行人信息
        if pedestrian_info and pedestrian_info.get("pedestrian_detected", False):
            lines.append(f"\nCurrent frame pedestrian status:")
            lines.append(f"  - Count: {pedestrian_info.get('pedestrian_count', 0)}")
            lines.append(f"  - Warning: {pedestrian_info.get('warning_level', 'unknown')}")
            raw_dets = pedestrian_info.get("raw_detections", [])
            if raw_dets:
                for j, det in enumerate(raw_dets):
                    conf = det.get("confidence", 0)
                    bbox = det.get("bbox", [])
                    lines.append(f"  - Pedestrian {j+1}: conf={conf:.2f}, bbox={bbox}")
        
        lines.append("=" * 50)
        return "\n".join(lines)

    def _decode_instruction_for_display(self, instruction: Any) -> str:
        """解码指令用于显示"""
        if isinstance(instruction, str):
            return instruction
        elif isinstance(instruction, (list, tuple)):
            return "".join(str(x) for x in instruction)
        elif hasattr(instruction, 'tolist'):
            # numpy array 或 tensor
            arr = instruction.tolist()
            if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], (int, float)):
                # 字节数组 - 解码为字符串
                non_zero = [x for x in arr if x != 0]
                try:
                    return bytes(non_zero).decode('utf-8', errors='ignore').strip()
                except:
                    return str(arr[:50]) + "..."
            return str(arr)
        return str(instruction)

    def _print_prompt(
        self,
        full_prompt_text: str,
        call_id: str,
        original_instruction: str,
        history_frames: Optional[List[FrameRecord]],
        pedestrian_info: Optional[Dict[str, Any]],
    ) -> None:
        """Print prompt content"""
        print("\n" + "=" * 80)
        print(f"[Brain Call #{self._brain_call_count} | ID: {call_id}]")
        print("=" * 80)

        # Decode instruction for display
        decoded_instruction = self._decode_instruction_for_display(original_instruction)

        # Print history frame info
        print("\n[History Frames]")
        if history_frames:
            for frame in history_frames[-self.max_history_frames:]:
                ped_info = frame.pedestrian_info[:80] if frame.pedestrian_info else 'N/A'
                has_image = "(in-memory)" if frame.image_array is not None else "(no image)"
                print(f"  Frame {frame.frame_id}: action={frame.action}, pedestrian={ped_info}...")
                print(f"    -> Image: {has_image}")
        else:
            print("  (No history)")

        # Print current frame pedestrian info
        print("\n[Pedestrian Detection]")
        ped_info_str = self._format_pedestrian_info(pedestrian_info, decoded_instruction) if pedestrian_info else 'None'
        print(f"  {ped_info_str}")

        # Print decoded instruction
        print("\n[Original Instruction (Decoded)]")
        print(f"  {decoded_instruction}")

        print("\n" + "=" * 80)

    def _save_prompt_to_file(self, full_prompt_text: str, call_id: str) -> str:
        """保存prompt到文件"""
        prompt_dir = os.path.join(self.output_dir, "prompt_records")
        os.makedirs(prompt_dir, exist_ok=True)
        filepath = os.path.join(prompt_dir, f"prompt_{call_id}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_prompt_text)
        print(f"[Brain Prompt] Saved prompt record: {filepath}")
        return filepath

    def _save_frame_image(self, image_array: np.ndarray, call_id: str, frame_type: str) -> str:
        """保存帧图像到文件"""
        os.makedirs(self.frame_images_root, exist_ok=True)
        filename = f"{call_id}_{frame_type}.jpg"
        filepath = os.path.join(self.frame_images_root, filename)
        try:
            from PIL import Image
            if image_array.dtype != np.uint8:
                img = Image.fromarray(image_array.astype(np.uint8))
            else:
                img = Image.fromarray(image_array)
            img.save(filepath)
        except Exception as e:
            print(f"[Brain] Failed to save frame image {filepath}: {e}")
            filepath = ""
        return filepath

    def _save_annotated_image(self, image_array: np.ndarray, call_id: str, pedestrian_info: Dict[str, Any]) -> str:
        """保存带行人检测框的标注图像"""
        os.makedirs(self.frame_images_root, exist_ok=True)
        filename = f"{call_id}_annotated.jpg"
        filepath = os.path.join(self.frame_images_root, filename)
        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2

            # 确保图像格式正确
            if image_array.dtype != np.uint8:
                img_np = image_array.astype(np.uint8)
            else:
                img_np = image_array.copy()

            # 如果是RGB格式，转换为BGR用于OpenCV绘图
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_np

            # 获取行人检测框并绘制
            raw_detections = pedestrian_info.get("raw_detections", [])
            if raw_detections:
                for det in raw_detections:
                    bbox = det.get("bbox", [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                        conf = det.get("confidence", 0)
                        # 根据置信度设置颜色
                        color = (0, 255, 0)  # 绿色
                        if conf < 0.5:
                            color = (0, 255, 255)  # 黄色
                        if conf < 0.3:
                            color = (0, 165, 255)  # 橙色
                        # 绘制边界框
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                        # 添加标签
                        label = f"Person: {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img_bgr, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 转换回RGB并保存
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.save(filepath)
            logger.info(f"[Brain-Annotated Image] Saved annotated image: {filepath}")
        except Exception as e:
            print(f"[Brain] Failed to save annotated image {filepath}: {e}")
            filepath = ""
        return filepath

    def _optimize_with_text_model(self, original_instruction: str, pedestrian_info_text: str, episode_id: Optional[str] = None) -> InstructionOptimizationResult:
        from PIL import Image
        import io

        self._brain_call_count += 1
        # 使用传入的episode_id或current_episode_id
        target_episode_id = episode_id if episode_id is not None else self.current_episode_id
        call_id = f"ep{target_episode_id}_call{self._brain_call_count}"

        # 构建消息格式（支持聊天模板）
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_text_prompt(original_instruction, pedestrian_info_text)}
        ]

        # 检查tokenizer是否支持聊天模板
        if hasattr(self.processor, 'apply_chat_template') and callable(self.processor.apply_chat_template):
            # 使用聊天模板
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text, return_tensors="pt", padding=True).to(self.device)
        else:
            # 传统方式：直接用processor处理prompt
            prompt = self._build_text_prompt(original_instruction, pedestrian_info_text)
            inputs = self.processor(prompt, return_tensors="pt", padding=True).to(self.device)

        try:
            outputs = self._generate_with_timeout(
                inputs,
                dict(**self.generation_config, pad_token_id=self.processor.tokenizer.eos_token_id)
            )
        except TimeoutError as e:
            logger.warning(f"[InstructionBrain] {e}")
            return self._fallback_optimization(original_instruction, None)

        # 安全地解码生成的文本
        input_ids_shape = inputs["input_ids"].shape

        # 计算输入序列长度
        if len(input_ids_shape) == 1:
            input_len = input_ids_shape[0]
        else:
            input_len = input_ids_shape[1]

        # 获取生成的token IDs
        generated_ids = outputs[0]
        if len(generated_ids.shape) > 1:
            generated_ids = generated_ids[0]

        # 判断outputs是否包含输入部分
        if len(generated_ids) > input_len:
            response = self.processor.decode(generated_ids[input_len:], skip_special_tokens=True)
        else:
            response = self.processor.decode(generated_ids, skip_special_tokens=True)

        return self._parse_optimization_response(original_instruction, response, None)

    # ── 动作空间相关提示词辅助方法 ──
    def _get_action_system_text(self) -> str:
        """根据 num_actions 返回系统提示词中的动作空间描述"""
        if self.num_actions == 4:
            return (
                "## ROBOT ACTION SYSTEM (4 Actions)\n"
                "The robot can only execute these discrete actions:\n"
                "- **Action 0 (STOP)**: Robot comes to a complete stop and terminates navigation\n"
                "- **Action 1 (FORWARD)**: Robot moves forward 0.25 meters\n"
                "- **Action 2 (LEFT)**: Robot turns left 15 degrees\n"
                "- **Action 3 (RIGHT)**: Robot turns right 15 degrees\n"
                "\n"
                "## NAVIGATION TRAJECTORY FORMAT (4 Actions for 4-Frame History)\n"
                'When analyzing trajectories, actions are described as (example with 4 actions to match 4-frame history):\n'
                '- "forward 0.25m ×1 → turn right 15° ×1 → forward 0.25m ×1 → stop ×1"\n'
                "This means: move forward 0.25m, rotate right 15°, move forward 0.25m, then stop."
            )
        else:
            return (
                "## ROBOT ACTION SYSTEM (6 Actions)\n"
                "The robot can only execute these discrete actions:\n"
                "- **Action 0 (STOP)**: Robot comes to a complete stop and terminates navigation\n"
                "- **Action 1 (FORWARD)**: Robot moves forward 0.25 meters\n"
                "- **Action 2 (LEFT)**: Robot turns left 15 degrees\n"
                "- **Action 3 (RIGHT)**: Robot turns right 15 degrees\n"
                "- **Action 4 (WAIT)**: Robot pauses in place for a moment, then can continue with next action\n"
                "- **Action 5 (BACKWARD)**: Robot moves backward 0.25 meters\n"
                "\n"
                "## SPECIAL: WHEN TO USE WAIT (Action 4)\n"
                "Use WAIT when:\n"
                "1. A pedestrian is moving toward you and you need time to assess their trajectory\n"
                "2. The pedestrian is in an unpredictable path and stopping gives them space to pass\n"
                "3. You want the robot to pause and let a pedestrian cross safely\n"
                "4. Immediate movement would cause a collision, so you wait for the pedestrian to clear\n"
                "\n"
                'Example: "WAIT for the person to cross, then continue forward when the path is clear."\n'
                "Do NOT overuse WAIT - only when stopping is safer than turning or moving forward.\n"
                "\n"
                "## NAVIGATION TRAJECTORY FORMAT (5 Actions for 5-Frame History)\n"
                "When analyzing trajectories, actions are described as (example with 5 actions to match 5-frame history):\n"
                '- "forward 0.25m ×1 → turn right 15° ×1 → forward 0.25m ×1 → wait ×1 → stop ×1"\n'
                "This means: move forward 0.25m, rotate right 15°, move forward 0.25m, pause briefly, then stop."
            )

    def _get_action_context_text(self) -> str:
        """根据 num_actions 返回用户消息中的动作空间上下文（用于 vision_model 和 API 调用）"""
        if self.num_actions == 4:
            return (
                "=== ROBOT ACTION SYSTEM (4 Actions) ===\n"
                "Action types:\n"
                "  - 0 (STOP): Robot comes to a complete stop and terminates navigation\n"
                "  - 1 (FORWARD): Robot moves forward 0.25 meters\n"
                "  - 2 (LEFT): Robot turns left 15 degrees\n"
                "  - 3 (RIGHT): Robot turns right 15 degrees\n"
                "\n"
                "=== TRAJECTORY FORMAT (4 Actions for 4-Frame History) ===\n"
                "Example: 'forward 0.25m ×1 → turn right 15° ×1 → forward 0.25m ×1 → stop ×1'\n"
                "This means: move forward 0.25m, rotate right 15°, move forward 0.25m, then stop.\n"
                "\n"
                "IMPORTANT: Generate instructions using natural language that maps to these discrete actions."
            )
        else:
            return (
                "=== ROBOT ACTION SYSTEM (6 Actions) ===\n"
                "Action types:\n"
                "  - 0 (STOP): Robot comes to a complete stop and terminates navigation\n"
                "  - 1 (FORWARD): Robot moves forward 0.25 meters\n"
                "  - 2 (LEFT): Robot turns left 15 degrees\n"
                "  - 3 (RIGHT): Robot turns right 15 degrees\n"
                "  - 4 (WAIT): Robot pauses in place for a moment, then can continue with next action\n"
                "  - 5 (BACKWARD): Robot moves backward 0.25 meters\n"
                "\n"
                "=== WHEN TO USE WAIT (Action 4) ===\n"
                "Use WAIT when:\n"
                "  1. A pedestrian is moving toward you and you need time to assess their trajectory\n"
                "  2. The pedestrian is in an unpredictable path and stopping gives them space to pass\n"
                "  3. You want the robot to pause and let a pedestrian cross safely\n"
                "  4. Immediate movement would cause a collision, so you wait for the pedestrian to clear\n"
                "\n"
                "=== TRAJECTORY FORMAT (5 Actions for 5-Frame History) ===\n"
                "Example: 'forward 0.25m ×1 → turn right 15° ×1 → forward 0.25m ×1 → wait ×1 → stop ×1'\n"
                "This means: move forward 0.25m, rotate right 15°, move forward 0.25m, pause briefly, then stop.\n"
                "\n"
                "IMPORTANT: Generate instructions using natural language that maps to these discrete actions."
            )

    def _get_action_full_prompt_lines(self) -> List[str]:
        """根据 num_actions 返回 _build_full_prompt_text 中的动作空间描述行"""
        lines = []
        if self.num_actions == 4:
            lines.append("  Robot Action System (4 Actions):")
            lines.append("    - Action 0 (STOP): Robot comes to a complete stop and terminates navigation")
            lines.append("    - Action 1 (FORWARD): Robot moves forward 0.25 meters")
            lines.append("    - Action 2 (LEFT): Robot turns left 15 degrees")
            lines.append("    - Action 3 (RIGHT): Robot turns right 15 degrees")
            lines.append("  ")
            lines.append("  【TRAJECTORY FORMAT (4 Actions for 4-Frame History)】")
            lines.append('    "forward 0.25m ×1 → turn right 15° ×1 → forward 0.25m ×1 → stop ×1"')
            lines.append("    means: move forward 0.25m, rotate right 15°, move forward 0.25m, then stop.")
        else:
            lines.append("  Robot Action System (6 Actions):")
            lines.append("    - Action 0 (STOP): Robot comes to a complete stop and terminates navigation")
            lines.append("    - Action 1 (FORWARD): Robot moves forward 0.25 meters")
            lines.append("    - Action 2 (LEFT): Robot turns left 15 degrees")
            lines.append("    - Action 3 (RIGHT): Robot turns right 15 degrees")
            lines.append("    - Action 4 (WAIT): Robot pauses in place for a moment, then can continue")
            lines.append("    - Action 5 (BACKWARD): Robot moves backward 0.25 meters")
            lines.append("  ")
            lines.append("  【WHEN TO USE WAIT (Action 4)】")
            lines.append("    Use WAIT when:")
            lines.append("      1. A pedestrian is moving toward you and you need time to assess their trajectory")
            lines.append("      2. The pedestrian is in an unpredictable path and stopping gives them space to pass")
            lines.append("      3. You want the robot to pause and let a pedestrian cross safely")
            lines.append("      4. Immediate movement would cause a collision, so you wait for the pedestrian to clear")
            lines.append("  ")
            lines.append("  【TRAJECTORY FORMAT (5 Actions for 5-Frame History)】")
            lines.append('    "forward 0.25m ×1 → turn right 15° ×1 → forward 0.25m ×1 → wait ×1 → stop ×1"')
            lines.append("    means: move forward 0.25m, rotate right 15°, move forward 0.25m, pause briefly, then stop.")
        lines.append("  Important: Instructions must use natural language compatible with these discrete actions.")
        return lines

    def _build_system_prompt(self) -> str:
        action_system = self._get_action_system_text()
        return (
            'You are a Social Navigation Oracle for Vision-and-Language Navigation (VLN) in human-populated dynamic environments. Your role is to provide on-demand cognitive assistance that elevates the robot\'s "Social EQ"—enabling fluid, etiquette-aware interactions rather than mechanical obstacle avoidance.\n'
            '\n'
            + action_system +
            '\n'
            '## YOUR PRIMARY TASK\n'
            'When pedestrians are detected, you MUST generate actionable navigation guidance that safely avoids the pedestrian WHILE explicitly retaining the original global navigation goal. \n'
            '\n'
            '## INSTRUCTION STRUCTURE: CONCISE AVOIDANCE FIRST, GLOBAL GOAL SECOND\n'
            'To prevent the robot from forgetting its destination, your `optimized_instruction` MUST follow this exact structure:\n'
            '[Concise Pedestrian Warning/Avoidance] + [Preserved Global Navigation Task]\n'
            '\n'
            '- GOOD: "AVOID: Person on left, shift right. Then, continue forward through the corridor, and turn right at the blue vase."\n'
            '- BAD (Forgets global goal): "AVOID: Pedestrian on left, turn right to go around them."\n'
            '- BAD (Warning at the end): "Continue forward through the corridor, then turn right. Watch out for pedestrians."\n'
            '- BAD (Too wordy): "I see a pedestrian walking on the left side of the path, so you should carefully turn right to bypass them safely, and after that..."\n'
            '\n'
            '## OUTPUT FORMAT (JSON ONLY - NO OTHER TEXT)\n'
            'Your output MUST be a valid JSON object with exactly these fields:\n'
            '{\n'
            '  "optimized_instruction": "Actionable navigation instruction strictly following the [Avoidance] + [Global Goal] structure",\n'
            '  "modifier_type": "ORIGINAL" or "REPLACE" or "APPEND_WARNING",\n'
            '  "confidence": score from 0.0 to 1.0,\n'
            '  "reasoning": "Specific explanation of what changed and why",\n'
            '  "safety_level": "safe" or "caution" or "danger",\n'
            '  "should_modify": true or false\n'
            '}\n'
            '\n'
            '## DECISION RULES FOR should_modify (BE GENEROUS - prefer modification):\n'
            '- Set should_modify=true whenever you add ANY specific navigation guidance\n'
            '- Set should_modify=true when you can provide a BETTER route around pedestrians\n'
            '- Set should_modify=true when you detect positional changes in pedestrians\n'
            '- Only set should_modify=false if the path is completely clear and the original instruction is perfect\n'
            '\n'
            '## GUIDELINES FOR EFFECTIVE NAVIGATION INSTRUCTIONS:\n'
            '1. **FIRST (Brief Avoidance)**: State a SHORT, actionable avoidance maneuver (e.g., "AVOID: Person ahead, take a wide left."). Keep this under 15 words.\n'
            '2. **SECOND (Global Goal)**: Explicitly append the original navigation task so the robot knows where to go after bypassing the person.\n'
            '3. Use natural directional language: "turn left/right", "go straight", "move forward", "stop at"\n'
            '4. Reference visible landmarks: "turn left at the doorway", "continue past the bookshelf"\n'
            '5. Keep the total instruction clear, actionable, and balanced (20-60 words total).\n'
            '\n'
            '## EXAMPLES OF GOOD VS BAD OUTPUTS:\n'
            '\n'
            'Original: "Turn left and go to the kitchen"\n'
            'Pedestrian: Detected ahead blocking path\n'
            '\n'
            'BAD OUTPUT (Dropped Global Goal):\n'
            '{\n'
            '  "optimized_instruction": "AVOID: Person directly ahead blocking your path. Take a wide turn to the left to move past them.",\n'
            '  "modifier_type": "REPLACE",\n'
            '  "confidence": 0.85,\n'
            '  "reasoning": "Pedestrian blocks direct path. Provided specific detour route.",\n'
            '  "safety_level": "danger",\n'
            '  "should_modify": true\n'
            '}\n'
            '\n'
            'GOOD OUTPUT (Proper Structure):\n'
            '{\n'
            '  "optimized_instruction": "AVOID: Person ahead blocking path, take wide left. Then, turn left and go to the kitchen.",\n'
            '  "modifier_type": "REPLACE",\n'
            '  "confidence": 0.90,\n'
            '  "reasoning": "Prepended concise avoidance instructions while preserving the original goal to reach the kitchen.",\n'
            '  "safety_level": "danger",\n'
            '  "should_modify": true\n'
            '}\n'
            '\n'
            'Original: "Move forward into the living room"\n'
            'Pedestrian: Detected at medium distance on left side\n'
            '\n'
            'GOOD OUTPUT (Proper Structure):\n'
            '{\n'
            '  "optimized_instruction": "CAUTION: Person on left, shift right. Continue moving forward into the living room.",\n'
            '  "modifier_type": "APPEND_WARNING",\n'
            '  "confidence": 0.85,\n'
            '  "reasoning": "Added brief caution about pedestrian on the left, followed by the original goal.",\n'
            '  "safety_level": "caution",\n'
            '  "should_modify": true\n'
            '}'
        )

    def _build_text_prompt(self, instruction: str, ped_info: str) -> str:
        return f"""You are a Social Navigation Oracle for Vision-and-Language Navigation (VLN).
Given the original navigation instruction and pedestrian detection information, generate an optimized instruction that:
1. First provides specific avoidance guidance (e.g., "AVOID: Person ahead, shift left")
2. Then preserves the original navigation goal

Original instruction: {instruction}

Pedestrian detection info: {ped_info}

IMPORTANT OUTPUT FORMAT (JSON only):
{{"optimized_instruction": "[AVOID: specific action] Then, [original goal preserved]", "modifier_type": "ORIGINAL/APPEND_WARNING/REPLACE", "confidence": 0.0-1.0, "reasoning": "brief explanation", "safety_level": "safe/caution/danger", "should_modify": true/false}}

Key requirements:
- If pedestrians detected, prepend "AVOID:" guidance before the original goal
- Always preserve the original navigation goal in the response
- Set confidence based on how well you can provide specific avoidance guidance
- should_modify should be true when you add specific navigation guidance"""

    def _parse_optimization_response(self, original_instruction: str, response: str, pedestrian_info: Optional[Dict[str, Any]]) -> InstructionOptimizationResult:
        # 尝试多种方式解析JSON
        optimized_instruction = original_instruction
        modifier_type = InstructionModifier.ORIGINAL
        confidence = 0.5
        reasoning = ""
        safety_level = "normal"
        should_modify = False
        warning_message = ""

        # 方法1: 使用更宽松的正则表达式匹配JSON
        try:
            # 尝试找到包含 optimized_instruction 的JSON对象
            # 匹配从 { 到匹配的 optimized_instruction 字段
            json_pattern = r'\{[^}]*"optimized_instruction"\s*:\s*"([^"]*)"[^}]*\}'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                optimized_instruction = match.group(1)
                # 重要修复：正确解析 should_modify 字段，而不是无条件设置为 True
                should_modify_match = re.search(r'"should_modify"\s*:\s*(true|false)', response, re.IGNORECASE)
                if should_modify_match:
                    should_modify = should_modify_match.group(1).lower() == 'true'
                else:
                    # 如果模型没有返回 should_modify 字段，默认为 False
                    # 只有当优化后的指令与原始指令不同时才考虑修改
                    should_modify = (optimized_instruction.strip() != original_instruction.strip())
                modifier_type = InstructionModifier.APPEND_WARNING

                # 尝试提取其他字段
                conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', response)
                if conf_match:
                    confidence = float(conf_match.group(1))

                safety_match = re.search(r'"safety_level"\s*:\s*"([^"]+)"', response)
                if safety_match:
                    safety_level = safety_match.group(1)

                reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response)
                if reason_match:
                    reasoning = reason_match.group(1)
        except Exception:
            pass

        # 方法2: 如果上面失败，尝试完整JSON解析
        if not should_modify and optimized_instruction == original_instruction:
            try:
                # 尝试找到完整的JSON对象（处理嵌套）
                # 找到第一个 { 和最后一个 }
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    if "optimized_instruction" in parsed:
                        optimized_instruction = parsed.get("optimized_instruction", original_instruction)
                        # 重要修复：正确解析 should_modify 字段，而不是无条件设置为 True
                        if "should_modify" in parsed:
                            should_modify = bool(parsed.get("should_modify", False))
                        else:
                            # 如果模型没有返回 should_modify 字段，默认为 False
                            # 只有当优化后的指令与原始指令不同时才考虑修改
                            should_modify = (optimized_instruction.strip() != original_instruction.strip())
                        modifier_type = InstructionModifier.APPEND_WARNING
                        confidence = parsed.get("confidence", 0.5)
                        reasoning = parsed.get("reasoning", "")
                        safety_level = parsed.get("safety_level", "normal")
            except Exception:
                pass

        # 方法3: 如果仍然失败，使用关键词判断
        # 注意：不将原始模型响应直接作为指令！使用原始指令保底，
        # 只根据关键词设置 should_modify 和 safety_level 等元数据。
        if not should_modify:
            response_lower = response.lower()
            if "不修改" in response or "保持" in response or "original" in response_lower or "no change" in response_lower:
                modifier_type = InstructionModifier.ORIGINAL
                should_modify = False
            elif "警告" in response or "warning" in response_lower or "caution" in response_lower:
                modifier_type = InstructionModifier.APPEND_WARNING
                should_modify = True
                # 由于JSON解析失败，不能安全地提取指令文本
                # 保持原始指令不变（安全兜底），记录警告
                optimized_instruction = original_instruction
                reasoning = f"Model indicated warning/caution but JSON parsing failed. Using original instruction as fallback. Response preview: {response[:100]}"
                safety_level = "caution"
            else:
                # 默认情况，使用原始指令
                modifier_type = InstructionModifier.ORIGINAL
                should_modify = False

        # 获取安全等级
        if pedestrian_info and not should_modify:
            warning = pedestrian_info.get("warning_level", "safe")
            if warning == "danger":
                safety_level = "danger"
            elif warning == "caution":
                safety_level = "caution"

        # 如果需要修改，确保指令不为空
        if should_modify and not optimized_instruction:
            optimized_instruction = original_instruction
            should_modify = False

        # 限制优化指令的长度（防止超过512字节限制）
        max_len = 400  # 约400字符，UTF-8编码后约400字节
        max_len_bytes = 500  # 字节长度限制
        truncated = False
        original_optimized_len = len(optimized_instruction)
        original_optimized_bytes = len(optimized_instruction.encode('utf-8'))

        if original_optimized_bytes > max_len_bytes:
            truncated = True
            # 按字节截断，确保不切割中文字符
            encoded = optimized_instruction.encode('utf-8')
            if len(encoded) > max_len_bytes:
                # 从后往前找完整的UTF-8字符边界
                while len(encoded) > max_len_bytes:
                    encoded = encoded[:-1]
                optimized_instruction = encoded.decode('utf-8', errors='ignore')

        # 记录日志（已注释）
        # if truncated:
        #     logger.info(f"[Brain-Parsing] Instruction truncated:")
        #     logger.info(f"  Before truncation: {original_optimized_len} chars / {original_optimized_bytes} bytes")
        #     logger.info(f"  After truncation: {len(optimized_instruction)} chars / {len(optimized_instruction.encode('utf-8'))} bytes")
        #     logger.info(f"  Original instruction preview: {original_instruction[:100]}...")
        #     logger.info(f"  Optimized instruction preview: {optimized_instruction[:100]}...")

        # 添加解析结果日志（已注释）
        # logger.info(f"[Brain-Parsing] Parse result: should_modify={should_modify}, modifier_type={modifier_type.value}")
        # logger.info(f"[Brain-Parsing] Original instruction: {original_instruction[:100]}...")
        # logger.info(f"[Brain-Parsing] Optimized instruction: {optimized_instruction[:100]}...")

        return InstructionOptimizationResult(
            original_instruction=original_instruction,
            optimized_instruction=optimized_instruction,
            modifier_type=modifier_type,
            confidence=confidence,
            reasoning=reasoning,
            safety_level=safety_level,
            pedestrian_warning=should_modify,
            warning_message=warning_message,
            should_modify=should_modify,
            raw_response=response
        )

    def _fallback_optimization(self, original_instruction: str, pedestrian_info: Optional[Dict[str, Any]]) -> InstructionOptimizationResult:
        """回退优化：没有可用模型时，直接返回原始指令
        
        注意：不允许硬编码任何警告消息。行人相关的优化应该由视觉语言模型处理，
        而不是在这里添加简单的警告文本。
        """
        # 直接返回原始指令，不做任何修改
        return InstructionOptimizationResult(
            original_instruction=original_instruction,
            optimized_instruction=original_instruction,
            modifier_type=InstructionModifier.ORIGINAL,
            confidence=0.0,  # 没有模型优化，置信度为0
            reasoning="No optimization model available, using original instruction",
            safety_level="unknown",
            pedestrian_warning=False,
            warning_message="",
            should_modify=False,
            raw_response=""
        )

    def should_update_instruction(self, original: str, optimized: str) -> bool:
        """判断优化后的指令是否应该替换原始指令

        规则：
        1. 如果优化指令与原始指令相同，不替换
        2. 如果优化指令过短（< 8 字符），认为无效，不替换
        3. 如果优化指令长度 < 原始指令的 20% 且 < 30 字符，认为信息丢失过多，不替换
        4. 否则，可以替换
        """
        orig_stripped = original.strip()
        opt_stripped = optimized.strip()

        if orig_stripped == opt_stripped:
            return False
        # 拒绝过短的指令（如 "Stop." 之类的单字指令，可能是模型输出截断）
        if len(opt_stripped) < 8:
            return False
        # 只有当优化指令既短又包含信息量不足时才拒绝
        if len(opt_stripped) < len(orig_stripped) * 0.2 and len(opt_stripped) < 30:
            return False
        return True

    def save_episode_record(self, episode_id: Optional[str] = None) -> str:
        if episode_id is None:
            episode_id = self.current_episode_id
        if episode_id is None:
            return ""
        episode = self.episode_records.get(episode_id)
        if episode is None:
            logger.warning(f"[InstructionBrain] save_episode_record: episode {episode_id} not found in episode_records")
            return ""
        
        # 调试日志：打印episode记录的详细信息
        logger.info(f"[InstructionBrain] save_episode_record: episode_id={episode_id}, "
                   f"total_frames={episode.total_frames}, "
                   f"frames_in_episode.frames={len(episode.frames)}, "
                   f"brain_calls={episode.brain_calls}")
        
        record_for_save = {
            "episode_id": episode.episode_id,
            "start_time": episode.start_time,
            "end_time": datetime.now().isoformat(),
            "original_instruction": episode.original_instruction,
            "total_frames": episode.total_frames,
            "frames_with_pedestrian": episode.frames_with_pedestrian,
            "brain_calls": episode.brain_calls,
            "instruction_modifications_count": episode.instruction_modifications_count,
            "instruction_modifications": episode.instruction_modifications,
            "frame_summaries": [
                {
                    "frame_id": f.frame_id,
                    "pedestrian_detected": f.pedestrian_detected,
                    "pedestrian_count": f.pedestrian_count,
                    "pedestrian_info": f.pedestrian_info,  # 现在是简洁格式（"No pedestrians detected" 或 ""）
                    "pedestrian_bbox": f.pedestrian_bbox,
                    "action": f.action,
                    "action_id": f.action_id,
                    "instruction": f.instruction,  # 完整指令
                    "instruction_preview": f.instruction[:100] + "..." if len(f.instruction) > 100 else f.instruction
                }
                for f in episode.frames
            ]
        }
        save_dir = os.path.join(self.output_dir, "episode_records")
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"episode_{episode_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record_for_save, f, ensure_ascii=False, indent=2)
        print(f"[InstructionBrain] Episode record saved: {filepath}")
        return filepath

    def end_episode(self, episode_id: Optional[str] = None, save: bool = True, env_idx: Optional[int] = None) -> None:
        """结束指定episode，保存记录并清理状态

        Args:
            episode_id: episode ID，如果为None则使用current_episode_id或env_idx对应的episode_id
            save: 是否保存记录，默认True。设置为False可以避免重复保存
                  （当外部已经调用save_episode_record时使用）
            env_idx: 环境索引（可选）。如果提供，会清理该env对应的episode_id映射。
        """
        # 使用resolve_episode_id解析episode_id
        target_episode_id = self.resolve_episode_id(episode_id, env_idx)
        if target_episode_id is None:
            target_episode_id = self.current_episode_id
        
        # 调试日志
        logger.info(f"[InstructionBrain] end_episode called with episode_id={target_episode_id}, env_idx={env_idx}, "
                   f"current_episode_id={self.current_episode_id}, save={save}")
        logger.info(f"[InstructionBrain] _env_current_episode_ids: {self._env_current_episode_ids}")

        # 如果没有episode_id，直接清理状态
        if target_episode_id is None:
            logger.info("[InstructionBrain] end_episode called with no active episode, just cleaning state")
            self.current_episode_id = None
            self.frame_history = []
            self.pedestrian_trajectory_history = []
            return

        # 清理该env的episode_id映射（如果有）
        if env_idx is not None and env_idx in self._env_current_episode_ids:
            if self._env_current_episode_ids[env_idx] == target_episode_id:
                del self._env_current_episode_ids[env_idx]
                logger.info(f"[InstructionBrain] Removed env{env_idx} from _env_current_episode_ids")

        # 检查是否有其他活跃的episode（在删除当前episode之前）
        other_active = [eid for eid in self.episode_records.keys() if eid != target_episode_id]

        # 尝试使用传入的episode_id保存记录
        saved = False
        if save and target_episode_id in self.episode_records:
            self.save_episode_record(target_episode_id)
            saved = True
        elif target_episode_id not in self.episode_records:
            logger.warning(f"[InstructionBrain] end_episode: episode {target_episode_id} not found in records.")

        # 如果启用帧图像保存，在episode结束时一次性保存所有帧图像用于调试
        if saved and self.save_frame_images:
            self._save_episode_frames_for_debug(target_episode_id)

        # 清理该episode的状态（包括所有缓存）
        if target_episode_id in self._frame_histories:
            del self._frame_histories[target_episode_id]
        if target_episode_id in self._pedestrian_trajectory_histories:
            del self._pedestrian_trajectory_histories[target_episode_id]
        if target_episode_id in self.episode_records:
            del self.episode_records[target_episode_id]
        # 清理节流缓存
        if target_episode_id in self._last_brain_call_frame:
            del self._last_brain_call_frame[target_episode_id]
        if target_episode_id in self._last_brain_call_result:
            del self._last_brain_call_result[target_episode_id]
        # 清理定期调用计数器
        if target_episode_id in self._last_periodic_call_frame:
            del self._last_periodic_call_frame[target_episode_id]
        
        # 只有在没有其他活跃episode时才清空current_episode_id
        # 但也要检查_env_current_episode_ids是否为空
        if other_active or self._env_current_episode_ids:
            # 如果_env_current_episode_ids还有内容，current_episode_id指向其中一个
            if self._env_current_episode_ids:
                # 指向任意一个还有映射的env对应的episode
                any_env_idx = next(iter(self._env_current_episode_ids))
                self.current_episode_id = self._env_current_episode_ids[any_env_idx]
                logger.info(f"[InstructionBrain] Episode {target_episode_id} ended, current_episode_id set to {self.current_episode_id} (from env{any_env_idx})")
            else:
                self.current_episode_id = other_active[0] if other_active else None
                logger.info(f"[InstructionBrain] Episode {target_episode_id} ended, current_episode_id set to {self.current_episode_id}")
        else:
            self.current_episode_id = None
            logger.info(f"[InstructionBrain] Episode {target_episode_id} ended, all episodes complete, state cleaned")

    def _save_episode_frames_for_debug(self, episode_id: str) -> None:
        """在episode结束时保存所有帧图像用于调试（一次性保存，不是每次调用都保存）

        这是一种可选的调试模式，用于在评估/训练结束后保存关键帧的可视化，
        而不是像之前那样每次Brain调用都保存图像到磁盘。

        Args:
            episode_id: episode ID
        """
        frame_history = self._frame_histories.get(episode_id, [])
        if not frame_history:
            return

        # 创建episode专用的图像保存目录
        episode_img_dir = os.path.join(self.frame_images_root, f"episode_{episode_id}")
        os.makedirs(episode_img_dir, exist_ok=True)

        saved_count = 0
        for frame in frame_history:
            # 只保存包含行人的帧或有标注信息的帧
            if frame.pedestrian_detected or frame.pedestrian_bbox:
                # 保存原始图像
                if frame.image_array is not None:
                    try:
                        from PIL import Image
                        img_path = os.path.join(episode_img_dir, f"frame_{frame.frame_id}_pedestrian.jpg")
                        if frame.image_array.dtype != np.uint8:
                            img = Image.fromarray(frame.image_array.astype(np.uint8))
                        else:
                            img = Image.fromarray(frame.image_array)
                        img.save(img_path)
                        saved_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to save frame {frame.frame_id}: {e}")

        if saved_count > 0:
            logger.info(f"[Brain-Debug] Saved {saved_count} pedestrian frames to {episode_img_dir}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取Brain调用统计信息

        Returns:
            包含统计信息的字典：
            - total_inference_time_s: 总推理耗时（秒）
            - inference_count: 推理次数
            - avg_inference_time_ms: 平均推理耗时（毫秒）
        """
        avg_time_ms = (self._total_inference_time / self._inference_count * 1000.0
                       if self._inference_count > 0 else 0.0)

        return {
            "total_inference_time_s": self._total_inference_time,
            "inference_count": self._inference_count,
            "avg_inference_time_ms": avg_time_ms,
        }

    def cleanup(self) -> None:
        """清理所有状态，包括所有 episode 记录"""
        # 先结束所有活跃的 episode
        if self.current_episode_id is not None:
            self.end_episode(self.current_episode_id)

        # 清理所有 episode 记录
        self.episode_records.clear()

        # 清理frame histories
        self._frame_histories.clear()
        self._pedestrian_trajectory_histories.clear()

        # 清理节流缓存
        self._last_brain_call_frame.clear()
        self._last_brain_call_result.clear()
        self._last_periodic_call_frame.clear()
        self._env_current_episode_ids.clear()

        # 清理模型
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[InstructionBrain] Full cleanup completed")
