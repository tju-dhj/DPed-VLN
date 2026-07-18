#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
# 抑制Habitat渲染相关的警告
os.environ["GLOG_minloglevel"] = "2"  # 抑制INFO级别以下的日志
os.environ["MAGNUM_LOG"] = "quiet"   # 抑制Magnum渲染引擎的日志

# 设置日志级别
logging.getLogger("habitat").setLevel(logging.ERROR)
logging.getLogger("habitat_sim").setLevel(logging.ERROR)
logging.getLogger("magnum").setLevel(logging.ERROR)
logging.getLogger("glog").setLevel(logging.ERROR)
sys.path.append("dinov3")
# 抑制所有警告输出
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
import os
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

try:
    import clip
except ImportError:
    clip = None

# ==================== Long-CLIP路径支持 ====================
# 将Long-CLIP/model目录添加到sys.path，使longclip模块可导入
import sys
import os

# 计算Long-CLIP可能的位置（相对于当前文件）
_longclip_candidates = [
    # 相对于当前文件的三级上级目录 + "Long-CLIP"
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Long-CLIP"),
    # 相对于当前工作目录
    os.path.join(os.getcwd(), "Long-CLIP"),
    # 相对于脚本目录
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Long-CLIP"),
]

_longclip_root = None
_longclip_path = None
for path in _longclip_candidates:
    if os.path.exists(path) and os.path.isdir(path):
        _longclip_root = os.path.abspath(path)
        # 将 model/ 子目录加入 sys.path，以导入 longclip 模块
        _longclip_path = os.path.join(_longclip_root, "model")
        if os.path.exists(_longclip_path) and os.path.isdir(_longclip_path):
            if _longclip_path not in sys.path:
                sys.path.insert(0, _longclip_path)
            print(f"[Long-CLIP] Found at: {_longclip_root}")
            print(f"[Long-CLIP] Model path: {_longclip_path}")
        break

# 尝试导入longclip
try:
    from longclip import load as longclip_load
    from longclip import tokenize as longclip_tokenize
    _longclip_available = True
    print("[Long-CLIP] Module imported successfully")
except ImportError as e:
    _longclip_available = False
    longclip_load = None
    longclip_tokenize = None
    print(f"[Long-CLIP] Import failed: {e}")
# ============================================================

@baseline_registry.register_policy
class PointNavResNetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Keyword arguments:
        rnn_type: RNN layer type; one of ["GRU", "LSTM"]
        backbone: Visual encoder backbone; one of ["resnet18", "resnet50", "resneXt50", "se_resnet50", "se_resneXt50", "se_resneXt101", "resnet50_clip_avgpool", "resnet50_clip_attnpool"]
        """

        assert backbone in [
            "resnet18",
            "resnet50",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
            "resnet50_clip_avgpool",
            "resnet50_clip_attnpool",
            "resnet50_clip_text",
            "dinov3_text",
        ], f"{backbone} backbone is not recognized."

        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        # 从kwargs中获取clip_visual_sensors（如果有）和clip_model_type
        clip_visual_sensors = kwargs.get("clip_visual_sensors", None)
        clip_model_type = kwargs.get("clip_model_type", "longclip")  # 默认使用longclip
        
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                clip_visual_sensors=clip_visual_sensors,
                clip_model_type=clip_model_type,  # 传递CLIP模型类型
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        # 获取CLIP视觉传感器配置（如果有）
        clip_visual_sensors = None
        if hasattr(config.habitat_baselines.rl.ddppo, "clip_visual_sensors") and \
           config.habitat_baselines.rl.ddppo.clip_visual_sensors is not None:
            clip_visual_sensors = OmegaConf.to_container(
                config.habitat_baselines.rl.ddppo.clip_visual_sensors, resolve=True
            )

        # 获取CLIP模型类型配置（默认使用longclip）
        clip_model_type = getattr(
            config.habitat_baselines.rl.ddppo, "clip_model_type", "longclip"
        )

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
            clip_visual_sensors=clip_visual_sensors,
            clip_model_type=clip_model_type,  # 传递CLIP模型类型
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid and k != "oracle_humanoid_future_trajectory"
        ]
        self.key_needs_rescaling = {k: None for k in self.visual_keys}
        for k, v in observation_space.spaces.items():
            if v.dtype == np.uint8:
                self.key_needs_rescaling[k] = 1.0 / v.high.max()

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_channels
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            spatial_size_h = (
                observation_space.spaces[self.visual_keys[0]].shape[0] // 2
            )
            spatial_size_w = (
                observation_space.spaces[self.visual_keys[0]].shape[1] // 2
            )
            self.backbone = make_backbone(
                self._n_input_channels, baseplanes, ngroups
            )

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                )
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2)
            if self.key_needs_rescaling[k] is not None:
                obs_k = (
                    obs_k.float() * self.key_needs_rescaling[k]
                )  # normalize
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        pooling="attnpool",
    ):
        super().__init__()

        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid
        ]

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if not self.is_blind:
            if clip is None:
                raise ImportError(
                    "Need to install CLIP (run `pip install git+https://github.com/openai/CLIP.git@40f5484c1c74edd83cb9cf687c6ab92b28d8b656`)"
                )

            model, preprocess = clip.load("RN50")

            # expected input: C x H x W (np.uint8 in [0-255])
            self.preprocess = T.Compose(
                [
                    # resize and center crop to 224
                    preprocess.transforms[0],
                    preprocess.transforms[1],
                    # already tensor, but want float
                    T.ConvertImageDtype(torch.float),
                    # normalize with CLIP mean, std
                    preprocess.transforms[4],
                ]
            )
            # expected output: C x H x W (np.float32)

            self.backbone = model.visual

            if self.rgb and self.depth:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048,)  # type: Tuple
            elif pooling == "none":
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif pooling == "avgpool":
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
                )
                self.output_shape = (2048,)
            else:
                self.output_shape = (1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self.rgb:
            rgb_observations = observations["rgb"]
            rgb_observations = rgb_observations.permute(
                0, 3, 1, 2
            )  # BATCH x CHANNEL x HEIGHT X WIDTH
            rgb_observations = torch.stack(
                [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            rgb_x = self.backbone(rgb_observations).float()
            cnn_input.append(rgb_x)

        if self.depth:
            depth_observations = observations["depth"][
                ..., 0
            ]  # [BATCH x HEIGHT X WIDTH]
            ddd = torch.stack(
                [depth_observations] * 3, dim=1
            )  # [BATCH x 3 x HEIGHT X WIDTH]
            ddd = torch.stack(
                [
                    self.preprocess(
                        TF.convert_image_dtype(depth_map, torch.uint8)
                    )
                    for depth_map in ddd
                ]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            depth_x = self.backbone(ddd).float()
            cnn_input.append(depth_x)

        if self.rgb and self.depth:
            x = F.adaptive_avg_pool2d(cnn_input[0] + cnn_input[1], 1)
            x = x.flatten(1)
        else:
            x = torch.cat(cnn_input, dim=1)

        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        text_encoder_dim: int = 2048,
        fusion_method: str = "attention",
        clip_visual_sensors: Optional[dict] = None,  # CLIP传感器配置
        clip_model_type: str = "longclip",  # CLIP模型类型，默认longclip
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self.text_encoder_dim = text_encoder_dim
        self.fusion_method = fusion_method
        self.clip_model_type = clip_model_type  # 保存CLIP模型类型
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            # removing keys that correspond to goal sensors
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]
        # 排除不应该融合的传感器：人数传感器、定位传感器、GT动作传感器、指令传感器
        exclude_keys = [
            "human_num_sensor", 
            "localization_sensor", 
            "falcon_gt_action", 
            "agent_0_falcon_gt_action",
            "falcon_instruction",
            "agent_0_falcon_instruction"
        ]
        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys if len(observation_space.spaces[k].shape) == 1 and k not in exclude_keys
        ]
        print(f"[DEBUG _fuse_keys_1d] keys={self._fuse_keys_1d}")
        print(f"[DEBUG _fuse_keys_1d] shapes={[(k, observation_space.spaces[k].shape) for k in self._fuse_keys_1d]}")
        print(f"[DEBUG _fuse_keys_1d] rnn_input_size after fuse={rnn_input_size + sum(observation_space.spaces[k].shape[0] for k in self._fuse_keys_1d)}")
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32


        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observation_space.spaces:
                goal_observation_space = spaces.Dict(
                    {"rgb": observation_space.spaces[uuid]}
                )
                goal_visual_encoder = ResNetEncoder(
                    goal_observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(resnet, backbone),
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)

                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(goal_visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )
                setattr(self, f"{uuid}_fc", goal_visual_fc)

                rnn_input_size += hidden_size

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )

        if backbone == "dinov3_text":
            # 使用joint_cfg
            joint_cfg_path = getattr(config, "dinov3", None)
            if joint_cfg_path is not None and hasattr(joint_cfg_path, "joint_cfg"):
                dinov3_cfg = OmegaConf.load(joint_cfg_path.joint_cfg)
                vision_cfg = dinov3_cfg.get("vision_backbone_config", None)
                text_cfg = dinov3_cfg.get("text_backbone_config", None)
                bpe_path = dinov3_cfg.get("text_vocab_path_or_url", None)
                embed_dim = dinov3_cfg.get("embed_dim", 768)
            else:
                raise ValueError("dinov3.joint_cfg must be set in config!")

            # Build DINOv3 towers
            self._dinov3_embed_dim = embed_dim
            from dinov3.eval.text.vision_tower import build_vision_model
            from dinov3.eval.text.text_tower import build_text_model
            from dinov3.eval.text.tokenizer import get_tokenizer
            self._dinov3_vision = build_vision_model(
                embed_dim=self._dinov3_embed_dim,
                backbone_model_config=vision_cfg,
                freeze_backbone=True,
                num_head_blocks=0,
                blocks_drop_path=0.0,
                use_class_token=True,
                use_patch_tokens=False,
                patch_token_layer=-1,
                patch_tokens_pooler_type="mean",
                use_linear_projection=True,
            )
            self._dinov3_text = build_text_model(
                embed_dim=self._dinov3_embed_dim,
                backbone_model_config=text_cfg,
                freeze_backbone=True,
                num_head_blocks=0,
                head_blocks_is_causal=True,
                head_blocks_drop_prob=0.0,
                tokens_pooler_type="argmax",
                use_linear_projection=True,
            )
            self._dinov3_tokenizer = get_tokenizer(bpe_path)

            class Dinov3TextEncoder(nn.Module):
                def __init__(self, outer):
                    super().__init__()
                    self.outer = outer
                    self.output_shape = (outer._dinov3_embed_dim,)

                @property
                def is_blind(self):
                    return False

                def _get_text_list(self, observations):
                    for key in [
                        "agent_0_falcon_instruction",
                        "falcon_instruction",
                        "instruction_sensor",
                    ]:
                        if key in observations:
                            data = observations[key]
                            if isinstance(data, str):
                                return [data]
                            if isinstance(data, list) and len(data) > 0:
                                return [str(data[0])]
                            if isinstance(data, torch.Tensor):
                                npv = data.detach().cpu().numpy()
                                if npv.ndim == 2:
                                    npv = npv[0]
                                npv = npv.astype(np.uint8).flatten()
                                bs = bytes(npv.tolist())
                                try:
                                    s = bs.split(b"\x00")[0].decode("utf-8").strip()
                                except Exception:
                                    try:
                                        s = (
                                            bs.split(b"\x00")[0]
                                            .decode("latin-1")
                                            .strip()
                                        )
                                        s = "".join(
                                            c for c in s if 32 <= ord(c) <= 126
                                        )
                                    except Exception:
                                        s = ""
                                return [s if s else "Navigate to the target location."]
                    return ["Navigate to the target location."]

                def forward(self, observations: Dict[str, torch.Tensor]):
                    imgs = []
                    if "rgb" in observations:
                        rgb = observations["rgb"].permute(0, 3, 1, 2).float() / 255.0
                        imgs.append(rgb)
                    for k in [
                        "agent_0_articulated_agent_jaw_rgb",
                        "articulated_agent_jaw_rgb",
                        "overhead_front_rgb",
                        "third_rgb",
                    ]:
                        if k in observations:
                            rgb = observations[k].permute(0, 3, 1, 2).float() / 255.0
                            imgs.append(rgb)
                    # Optional depth
                    for dk in [
                        "depth",
                        "agent_0_articulated_agent_jaw_depth",
                        "articulated_agent_jaw_depth",
                        "overhead_front_depth",
                        "third_depth",
                    ]:
                        if dk in observations:
                            d = observations[dk][..., 0].unsqueeze(1).float()
                            d = d.clamp(min=0.0, max=1.0)
                            d3 = d.repeat(1, 3, 1, 1)
                            imgs.append(d3)

                    if len(imgs) == 0:
                        return None

                    # Run vision on each view, average features
                    feats = []
                    for img in imgs:
                        f, _, _ = self.outer._dinov3_vision(img)
                        feats.append(f)
                    visual_features = torch.stack(feats, dim=0).mean(dim=0)

                    # Text features
                    text_list = self._get_text_list(observations)
                    tokens = self.outer._dinov3_tokenizer.tokenize(text_list, 77).to(
                        visual_features.device
                    )
                    text_features = self.outer._dinov3_text(tokens)

                    # Simple cross-modal fusion: sum (both are embed_dim)
                    fused = visual_features + text_features
                    return fused

            self.visual_encoder = Dinov3TextEncoder(self)
            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Linear(self.visual_encoder.output_shape[0], hidden_size),
                    nn.ReLU(True),
                )
            self._visual_feature_size = hidden_size
        elif backbone.startswith("resnet50_clip"):
            if backbone in ["resnet50_clip_text", "resnet50_clip_attnpool"]:
                # 使用支持文本指令的ResNetCLIPTextEncoder
                # 从配置中获取CLIP传感器配置
                rgb_keys = None
                depth_keys = None
                visual_fusion_mode = "average"
                normalize_before_fusion = True
                
                if clip_visual_sensors is not None:
                    rgb_keys = clip_visual_sensors.get("rgb_keys", None)
                    depth_keys = clip_visual_sensors.get("depth_keys", None)
                    visual_fusion_mode = clip_visual_sensors.get("fusion_mode", "average")
                    normalize_before_fusion = clip_visual_sensors.get("normalize_before_fusion", True)
                
                self.visual_encoder = ResNetCLIPTextEncoder(
                    observation_space
                    if not force_blind_policy
                    else spaces.Dict({}),
                    pooling="attnpool",
                    text_encoder_dim=self.text_encoder_dim,
                    fusion_method=self.fusion_method,
                    rgb_sensor_keys=rgb_keys,
                    depth_sensor_keys=depth_keys,
                    visual_fusion_mode=visual_fusion_mode,
                    normalize_before_fusion=normalize_before_fusion,
                    clip_model_type=self.clip_model_type,  # 传递CLIP模型类型
                )
                # ResNetCLIPTextEncoder输出2048维，需要投影到hidden_size
                if not self.visual_encoder.is_blind:
                    self.visual_fc = nn.Sequential(
                        nn.Linear(
                            self.visual_encoder.output_shape[0], hidden_size
                        ),
                        nn.ReLU(True),
                    )
                # 设置视觉特征大小
                self._visual_feature_size = hidden_size
                print(f"PointNavResNetNet: CLIP架构，visual_feature_size = {self._visual_feature_size}")    
            else:
                self.visual_encoder = ResNetCLIPEncoder(
                    observation_space
                    if not force_blind_policy
                    else spaces.Dict({}),
                    pooling="avgpool" if "avgpool" in backbone else "attnpool",
                )
                if not self.visual_encoder.is_blind:
                    self.visual_fc = nn.Sequential(
                        nn.Linear(
                            self.visual_encoder.output_shape[0], hidden_size
                        ),
                        nn.ReLU(True),
                    )
                    self._visual_feature_size = hidden_size
        else:
            self.visual_encoder = ResNetEncoder(
                use_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            if not self.visual_encoder.is_blind:
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(self.visual_encoder.output_shape), hidden_size
                    ),
                    nn.ReLU(True),
                )
                self._visual_feature_size = hidden_size

        # 计算RNN输入维度
        total_rnn_input = (0 if self.is_blind else self._visual_feature_size) + rnn_input_size

        self.state_encoder = build_rnn_state_encoder(
            total_rnn_input,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        if not self.is_blind:
            # We CANNOT use observations.get() here because self.visual_encoder(observations)
            # is an expensive operation. Therefore, we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)

            # 如果有visual_fc层，则应用它（CLIP编码器需要投影到hidden_size）
            if hasattr(self, 'visual_fc') and self.visual_fc is not None:
                visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())


        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observations:
                goal_image = observations[uuid]

                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})

                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            # The mask means the previous action will be zero, an extra dummy action
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state

class ResNetCLIPTextEncoder(nn.Module):
    """
    支持文本指令和多视角传感器的ResNetCLIPEncoder
    - 支持通过配置文件指定RGB和Depth传感器键名
    - 支持多视角融合（average/concat/attention三种模式）
    - 使用多头注意力机制融合图像和文本指令信息
    - 支持Long-CLIP（默认，支持248 tokens）和标准CLIP（77 tokens）
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        pooling="attnpool",
        text_instruction_path: str = None,
        text_encoder_dim: int = 2048,  # 修改为与视觉特征维度一致
        fusion_method: str = "attention",  # 图像-文本融合方法
        rgb_sensor_keys: Optional[List[str]] = None,  # 配置文件指定的RGB传感器键名（不含agent_0_前缀）
        depth_sensor_keys: Optional[List[str]] = None,  # 配置文件指定的Depth传感器键名（不含agent_0_前缀）
        visual_fusion_mode: str = "average",  # 多视角融合模式: average/concat/attention
        normalize_before_fusion: bool = True,  # 融合前是否归一化
        clip_model_type: str = "longclip",  # CLIP模型类型: "longclip" 或 "clip"
    ):
        super().__init__()

        # 保存CLIP模型类型
        self.clip_model_type = clip_model_type

        # 保存多视角融合配置
        self.visual_fusion_mode = visual_fusion_mode
        self.normalize_before_fusion = normalize_before_fusion
        
        # 如果配置文件指定了传感器键名，使用配置的键名；否则使用默认查找顺序
        if rgb_sensor_keys is not None and len(rgb_sensor_keys) > 0:
            # 配置文件指定了RGB传感器，需要添加agent_0_前缀匹配
            rgb_keys = []
            for key in rgb_sensor_keys:
                # 支持带或不带前缀的键名
                if f"agent_0_{key}" in observation_space.spaces:
                    rgb_keys.append(f"agent_0_{key}")
                elif key in observation_space.spaces:
                    rgb_keys.append(key)
            self.configured_rgb_keys = rgb_keys
        else:
            # 使用默认查找顺序（向后兼容）
            rgb_keys = [
                "rgb",
                "agent_0_overhead_front_rgb",
                "overhead_front_rgb",
                "agent_0_articulated_agent_jaw_rgb",
                "articulated_agent_jaw_rgb",
                "agent_0_third_rgb",
                "third_rgb"
            ]
            self.configured_rgb_keys = None
        
        if depth_sensor_keys is not None and len(depth_sensor_keys) > 0:
            # 配置文件指定了Depth传感器，需要添加agent_0_前缀匹配
            depth_keys = []
            for key in depth_sensor_keys:
                # 支持带或不带前缀的键名
                if f"agent_0_{key}" in observation_space.spaces:
                    depth_keys.append(f"agent_0_{key}")
                elif key in observation_space.spaces:
                    depth_keys.append(key)
            self.configured_depth_keys = depth_keys
        else:
            # 使用默认查找顺序（向后兼容）
            depth_keys = [
                "depth",
                "agent_0_overhead_front_depth",
                "overhead_front_depth",
                "agent_0_articulated_agent_jaw_depth",
                "articulated_agent_jaw_depth",
                "agent_0_third_depth",
                "third_depth"
            ]
            self.configured_depth_keys = None
        
        self.rgb = any(k in observation_space.spaces for k in rgb_keys)
        self.depth = any(k in observation_space.spaces for k in depth_keys)
        self.text_instruction_path = text_instruction_path
        self.text_encoder_dim = text_encoder_dim
        self.fusion_method = fusion_method

        # Determine which visual observations are present
        self.visual_keys = [
            k
            for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and k != ImageGoalSensor.cls_uuid and k not in ["instruction_sensor", "gt_action_sensor", "falcon_instruction", "falcon_gt_action", "agent_0_falcon_instruction", "agent_0_falcon_gt_action", "oracle_humanoid_future_trajectory"]
        ]

        # Count total # of channels
        self._n_input_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.visual_keys
        )

        if not self.is_blind:
            if clip is None:
                raise ImportError(
                    "Need to install CLIP (run `pip install git+https://github.com/openai/CLIP.git@40f5484c1c74edd83cb9cf687c6ab92b28d8b656`)"
                )

            # 加载CLIP模型，包含视觉和文本编码器
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 根据clip_model_type决定加载哪种模型
            if self.clip_model_type == "longclip":
                if _longclip_available and longclip_load is not None:
                    # 构建checkpoint路径 - 尝试多个可能的位置
                    ckpt_candidates = []

                    # 1. 使用全局变量 _longclip_root（Long-CLIP根目录，如果找到）
                    if _longclip_root is not None:
                        ckpt_candidates.append(os.path.join(_longclip_root, "checkpoints", "longclip-L.pt"))
                        ckpt_candidates.append(os.path.join(_longclip_root, "checkpoints", "longclip-B.pt"))

                    # 2. 相对于当前文件的路径
                    ckpt_candidates.append(os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "Long-CLIP", "checkpoints", "longclip-L.pt"
                    ))
                    ckpt_candidates.append(os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "Long-CLIP", "checkpoints", "longclip-B.pt"
                    ))

                    # 3. 相对于工作目录的路径
                    ckpt_candidates.append(os.path.join(os.getcwd(), "Long-CLIP", "checkpoints", "longclip-L.pt"))
                    ckpt_candidates.append(os.path.join(os.getcwd(), "Long-CLIP", "checkpoints", "longclip-B.pt"))

                    # 查找第一个存在的checkpoint
                    ckpt_path = None
                    for candidate in ckpt_candidates:
                        if os.path.exists(candidate):
                            ckpt_path = candidate
                            print(f"[Long-CLIP] Found checkpoint at: {ckpt_path}")
                            break

                    try:
                        if ckpt_path is not None:
                            # 使用本地checkpoint加载
                            self.clip_model, preprocess = longclip_load(ckpt_path, device=device)
                            print(f"[ResNetCLIPTextEncoder] Using LongCLIP from local checkpoint")
                        else:
                            # 尝试使用HuggingFace名称
                            self.clip_model, preprocess = longclip_load("LongCLIP-L", device=device)
                            print(f"[ResNetCLIPTextEncoder] Using LongCLIP-L from HuggingFace/cache")
                        # LongCLIP checkpoint 使用 fp16 保存，转换为 fp32 以匹配 Habitat 的 fp32 输入
                        self.clip_model = self.clip_model.float()
                        print(f"[ResNetCLIPTextEncoder] LongCLIP model converted to float32")
                        self.use_long_clip = True
                    except Exception as e:
                        print(f"[ResNetCLIPTextEncoder] LongCLIP loading failed: {e}")
                        print("[ResNetCLIPTextEncoder] Falling back to standard CLIP")
                        self.clip_model, preprocess = clip.load("RN50", device=device)
                        self.use_long_clip = False
                else:
                    print("[ResNetCLIPTextEncoder] LongCLIP module not available, using standard CLIP")
                    self.clip_model, preprocess = clip.load("RN50", device=device)
                    self.use_long_clip = False
            else:
                # 使用标准CLIP
                self.clip_model, preprocess = clip.load("RN50", device=device)
                self.use_long_clip = False
                print(f"[ResNetCLIPTextEncoder] Using standard CLIP (77 tokens)")

            # 文本编码器
            self.text_encoder = self.clip_model.encode_text

            # 视觉编码器
            self.visual_encoder = self.clip_model.visual

            # expected input: C x H x W (np.uint8 in [0-255])
            self.preprocess = T.Compose(
                [
                    # resize and center crop to 224
                    preprocess.transforms[0],
                    preprocess.transforms[1],
                    # already tensor, but want float
                    T.ConvertImageDtype(torch.float),
                    # normalize with CLIP mean, std
                    preprocess.transforms[4],
                ]
            )

            # 统一视觉输出维度 - 从模型推断实际维度
            # CLIP RN50: 1024 (或 2048 不经过 attnpool)
            # LongCLIP-L (ViT-L/14): 768
            # LongCLIP-B (ViT-B/16): 512
            self._actual_text_dim = self.clip_model.text_projection.shape[1]
            # 获取视觉输出维度（尝试多个可能的属性）
            if hasattr(self.visual_encoder, 'output_dim'):
                self._actual_visual_dim = self.visual_encoder.output_dim
            elif hasattr(self.visual_encoder, 'proj'):
                self._actual_visual_dim = self.visual_encoder.proj.shape[1]
            else:
                # 回退：使用 text_dim（ViT架构下两者相同）
                self._actual_visual_dim = self._actual_text_dim
            visual_output_dim = self._actual_visual_dim
            print(f"[ResNetCLIPTextEncoder] Detected text_dim={self._actual_text_dim}, visual_dim={self._actual_visual_dim}")

            # 文本特征投影层 - 将CLIP文本特征投影到视觉维度
            # RN50: 1024 -> 2048 (或1024); ViT-L/14: 768 -> 768; ViT-B/16: 512 -> 512
            self.text_projection = nn.Linear(self._actual_text_dim, visual_output_dim)
            self.text_projection = self.text_projection.to(device)
            
            # 确保投影层可训练
            for param in self.text_projection.parameters():
                param.requires_grad = True
            
            # 多视角融合层（仅在有多个RGB或depth传感器时使用）
            num_rgb_sensors = len(self.configured_rgb_keys) if self.configured_rgb_keys else 1
            num_depth_sensors = len(self.configured_depth_keys) if self.configured_depth_keys else 1
            
            if self.visual_fusion_mode == "concat":
                # 拼接模式：需要投影层将拼接后的特征投影到2048维
                total_rgb_depth = (num_rgb_sensors if self.rgb else 0) + (num_depth_sensors if self.depth else 0)
                if total_rgb_depth > 1:
                    self.visual_fusion_proj = nn.Linear(visual_output_dim * total_rgb_depth, visual_output_dim).to(device)
                    for param in self.visual_fusion_proj.parameters():
                        param.requires_grad = True
                else:
                    self.visual_fusion_proj = None
            elif self.visual_fusion_mode == "attention":
                # 注意力模式：使用多头自注意力融合多视角特征
                total_rgb_depth = (num_rgb_sensors if self.rgb else 0) + (num_depth_sensors if self.depth else 0)
                if total_rgb_depth > 1:
                    self.visual_fusion_attn = nn.MultiheadAttention(
                        embed_dim=visual_output_dim, 
                        num_heads=8, 
                        batch_first=True
                    ).to(device)
                    for param in self.visual_fusion_attn.parameters():
                        param.requires_grad = True
                else:
                    self.visual_fusion_attn = None
            else:
                # 平均模式：不需要额外参数
                self.visual_fusion_proj = None
                self.visual_fusion_attn = None
            
            # 调试：打印投影层的可训练参数数量
            trainable_params = sum(p.numel() for p in self.text_projection.parameters() if p.requires_grad)
            # print(f"[ResNetCLIPTextEncoder] text_projection 可训练参数数量: {trainable_params}")
            
            # if self.configured_rgb_keys:
                # print(f"[ResNetCLIPTextEncoder] 配置的RGB传感器: {self.configured_rgb_keys}")
            # if self.configured_depth_keys:
                # print(f"[ResNetCLIPTextEncoder] 配置的Depth传感器: {self.configured_depth_keys}")
            # print(f"[ResNetCLIPTextEncoder] 多视角融合模式: {self.visual_fusion_mode}")
            
            # 视觉特征投影层 - 确保视觉特征维度一致
            # self.visual_projection = nn.Linear(visual_output_dim, visual_output_dim)
            # self.visual_projection = self.visual_projection.to(device)
            
            # 多头注意力融合层 - 用于动态融合视觉和文本特征
            # 调整num_heads以确保embed_dim能被整除
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=visual_output_dim, 
                num_heads=8,  # 2048 / 8 = 256，确保维度匹配
                batch_first=True
            )
            self.cross_modal_attention = self.cross_modal_attention.to(device)
            self._debug_input_print_count = 0
            
            # 输出投影层 - 将融合后的特征投影到最终维度
            self.output_projection = nn.Linear(visual_output_dim, visual_output_dim)
            self.output_projection = self.output_projection.to(device)

            # 冻结CLIP参数
            for param in self.clip_model.parameters():
                param.requires_grad_(False)
            for module in self.clip_model.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.clip_model.eval()
            
            # 设置输出形状
            self.output_shape = (visual_output_dim,)
            
            # 启用训练监控
            self._monitor_training = True

    @property
    def is_blind(self):
        return self._n_input_channels == 0

    def _decode_instruction_array(self, instruction_array) -> str:
        arr = np.asarray(instruction_array).astype(np.uint8).flatten()
        instruction_bytes = arr.tobytes()
        null_pos = instruction_bytes.find(b'\x00')
        if null_pos >= 0:
            instruction_bytes = instruction_bytes[:null_pos]
        try:
            instruction = instruction_bytes.decode('utf-8').strip()
        except UnicodeDecodeError:
            instruction = instruction_bytes.decode('latin-1').strip()
            instruction = ''.join(c for c in instruction if 32 <= ord(c) <= 126)
        return instruction if instruction else "Navigate to the target location."

    def _decode_instruction_batch(self, instruction_data):
        if isinstance(instruction_data, str):
            return [instruction_data]
        if isinstance(instruction_data, list):
            return [str(inst) if str(inst).strip() else "Navigate to the target location." for inst in instruction_data]

        if isinstance(instruction_data, torch.Tensor):
            instruction_np = instruction_data.detach().cpu().numpy()
        elif isinstance(instruction_data, np.ndarray):
            instruction_np = instruction_data
        else:
            print(f"Warning: instruction data type unexpected: {type(instruction_data)}")
            return None

        if instruction_np.ndim == 1:
            instruction_np = instruction_np.reshape(1, -1)
        elif instruction_np.ndim > 2:
            instruction_np = instruction_np.reshape(-1, instruction_np.shape[-1])

        return [self._decode_instruction_array(row) for row in instruction_np]

    def encode_text(self, text_instructions):
        """编码文本指令（带缓存：CLIP backbone frozen，缓存原始特征；每步重跑 text_projection）"""
        if text_instructions is None:
            return None

        # 处理空列表或空字符串的情况
        if isinstance(text_instructions, list) and len(text_instructions) == 0:
            return None

        if isinstance(text_instructions, list) and len(text_instructions) > 0:
            # 过滤掉空字符串
            text_instructions = [inst for inst in text_instructions if inst and len(inst.strip()) > 0]
            if len(text_instructions) == 0:
                return None

        # 初始化缓存和统计计数器
        if not hasattr(self, '_text_feature_cache'):
            self._text_feature_cache = {}
            self._cache_stats = {"hits": 0, "misses": 0, "encodes": 0}
            self._last_inst_fingerprints = [None] * len(text_instructions)
            self._step_counter = 0

        self._step_counter += 1
        batch_size = len(text_instructions)

        # 检查哪些指令需要编码
        raw_features = []  # 未投影的 CLIP 特征
        need_encode_indices = []
        need_encode_texts = []
        inst_changed_envs = []  # 追踪哪些 env 的指令变了

        for i, inst in enumerate(text_instructions):
            cache_key = inst  # 用指令字符串作为缓存键
            # 用 hash 检查指令是否与上一步相同（追踪 episode 切换）
            inst_hash = hash(cache_key) if cache_key else None
            if i < len(self._last_inst_fingerprints):
                if self._last_inst_fingerprints[i] != inst_hash:
                    inst_changed_envs.append(i)
                self._last_inst_fingerprints[i] = inst_hash
            elif i >= len(self._last_inst_fingerprints):
                # batch size 变化时扩展
                self._last_inst_fingerprints.extend([None] * (i - len(self._last_inst_fingerprints) + 1))
                self._last_inst_fingerprints[i] = inst_hash
                inst_changed_envs.append(i)

            if cache_key in self._text_feature_cache:
                raw_features.append(self._text_feature_cache[cache_key])
                self._cache_stats["hits"] += 1
            else:
                need_encode_indices.append(i)
                need_encode_texts.append(inst)
                raw_features.append(None)  # 占位
                self._cache_stats["misses"] += 1

        # 对新指令批量编码（CLIP backbone: 冻结, 重量级操作）
        if len(need_encode_texts) > 0:
            with torch.no_grad():
                device = next(self.clip_model.parameters()).device

                if self.use_long_clip:
                    from longclip import tokenize as longclip_tokenize
                    text_tokens = longclip_tokenize(need_encode_texts).to(device)
                else:
                    text_tokens = clip.tokenize(need_encode_texts, truncate=True).to(device)

                new_features = self.text_encoder(text_tokens).float()

            # 存入缓存
            for j, (idx, inst) in enumerate(zip(need_encode_indices, need_encode_texts)):
                feat = new_features[j:j+1]
                self._text_feature_cache[inst] = feat
                raw_features[idx] = feat
            self._cache_stats["encodes"] += len(need_encode_texts)

        # 每 500 步打印缓存统计 (已禁用)
        # if self._step_counter % 500 == 1 and self._step_counter > 1:
        #     stats = self._cache_stats
        #     total = stats["hits"] + stats["misses"]
        #     hit_rate = stats["hits"] / total * 100 if total > 0 else 0
        #     print(f"[TextCache] step={self._step_counter} | "
        #           f"cache_size={len(self._text_feature_cache)} | "
        #           f"hits={stats['hits']} misses={stats['misses']} encodes={stats['encodes']} | "
        #           f"hit_rate={hit_rate:.1f}%")

        # episode 切换时打印（已禁用）
        # if len(inst_changed_envs) > 0 and getattr(self, "_debug_input_print_count", 0) < 10:
        #     print(f"[TextCache] Episode transition detected: "
        #           f"envs={inst_changed_envs} | "
        #           f"cache_size={len(self._text_feature_cache)} | "
        #           f"new_encodes={len(need_encode_texts)}")

        # 拼接原始特征并投影（text_projection: 可训练, 轻量级操作）
        raw_features = torch.cat(raw_features, dim=0)  # [B, text_dim]
        text_features = self.text_projection(raw_features)

        return text_features

    def forward(self, observations: Dict[str, torch.Tensor], episode_ids: List[str] = None) -> torch.Tensor:
        if self.is_blind:
            return None
        #  debug_enabled = getattr(self, "_debug_input_print_count", 0) < 10
        debug_enabled = False  # 禁用调试输出
        if debug_enabled:
            print("[ResNetCLIPTextEncoder DEBUG] ===== forward input summary (step {}) =====".format(
                getattr(self, "_debug_input_print_count", 0)))
            print(f"[ResNetCLIPTextEncoder DEBUG] observation keys: {sorted(list(observations.keys()))}")
            for obs_key in sorted(observations.keys()):
                obs_value = observations[obs_key]
                if torch.is_tensor(obs_value):
                    shape = tuple(obs_value.shape)
                    dtype = obs_value.dtype
                    device = obs_value.device
                    try:
                        min_val = float(obs_value.min().item()) if obs_value.numel() > 0 else float("nan")
                        max_val = float(obs_value.max().item()) if obs_value.numel() > 0 else float("nan")
                        mean_val = float(obs_value.float().mean().item()) if obs_value.numel() > 0 else float("nan")
                    except Exception:
                        min_val = float("nan")
                        max_val = float("nan")
                        mean_val = float("nan")
                    print(
                        f"[ResNetCLIPTextEncoder DEBUG] obs[{obs_key}] "
                        f"shape={shape} dtype={dtype} device={device} min={min_val:.4f} max={max_val:.4f} mean={mean_val:.4f}"
                    )
                else:
                    print(f"[ResNetCLIPTextEncoder DEBUG] obs[{obs_key}] type={type(obs_value)}")

        # 处理视觉输入 - 支持多视角传感器
        rgb_features_list = []  # 存储多个RGB传感器的特征
        depth_features_list = []  # 存储多个Depth传感器的特征
        
        if self.rgb:
            # 如果配置文件指定了RGB传感器，使用配置的键名；否则使用第一个找到的
            if self.configured_rgb_keys:
                rgb_keys_to_use = self.configured_rgb_keys
            else:
                # 向后兼容：使用第一个找到的RGB键名
                rgb_keys_to_use = []
                for possible_key in [
                    "rgb",
                    "agent_0_overhead_front_rgb",
                    "overhead_front_rgb",
                    "agent_0_articulated_agent_jaw_rgb",
                    "articulated_agent_jaw_rgb",
                    "agent_0_third_rgb",
                    "third_rgb"
                ]:
                    if possible_key in observations:
                        rgb_keys_to_use = [possible_key]
                        break
            
            if len(rgb_keys_to_use) == 0:
                raise KeyError(f"No RGB sensor found in observations. Available keys: {list(observations.keys())}")
            
            # 处理所有配置的RGB传感器
            for rgb_key in rgb_keys_to_use:
                if rgb_key not in observations:
                    print(f"[WARNING] 配置的RGB传感器 '{rgb_key}' 不在observations中，跳过")
                    continue
                
                rgb_observations = observations[rgb_key]
                if debug_enabled:
                    print(
                        f"[ResNetCLIPTextEncoder DEBUG] RGB raw {rgb_key}: "
                        f"shape={tuple(rgb_observations.shape)} dtype={rgb_observations.dtype} device={rgb_observations.device}"
                    )
                    # Print per-channel stats to verify image varies between steps
                    if rgb_observations.shape[-1] >= 3:
                        for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                            ch = rgb_observations[0, ..., ch_idx].float()
                            print(
                                f"[ResNetCLIPTextEncoder DEBUG]   {ch_name} channel: "
                                f"min={ch.min().item():.1f} max={ch.max().item():.1f} mean={ch.mean().item():.1f}"
                            )
                rgb_observations = rgb_observations.permute(0, 3, 1, 2)  # BATCH x CHANNEL x HEIGHT X WIDTH
                rgb_observations = torch.stack(
                    [self.preprocess(rgb_image) for rgb_image in rgb_observations]
                )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
                if debug_enabled:
                    print(
                        f"[ResNetCLIPTextEncoder DEBUG] RGB preprocessed {rgb_key}: "
                        f"shape={tuple(rgb_observations.shape)} dtype={rgb_observations.dtype}"
                        f" min={rgb_observations.min().item():.4f} max={rgb_observations.max().item():.4f} mean={rgb_observations.mean().item():.4f}"
                    )
                
                try:
                    rgb_x = self.visual_encoder(rgb_observations).float()
                    rgb_features_list.append(rgb_x)
                except RuntimeError as e:
                    print(
                        f"[ResNetCLIPTextEncoder] RGB传感器 '{rgb_key}' 编码失败 - ",
                        f"shape={tuple(rgb_observations.shape)}, "
                        f"dtype={rgb_observations.dtype}, "
                        f"device={rgb_observations.device}",
                    )
                    raise

        if self.depth:
            # 如果配置文件指定了Depth传感器，使用配置的键名；否则使用第一个找到的
            if self.configured_depth_keys:
                depth_keys_to_use = self.configured_depth_keys
            else:
                # 向后兼容：使用第一个找到的Depth键名
                depth_keys_to_use = []
                for possible_key in [
                    "depth",
                    "agent_0_overhead_front_depth",
                    "overhead_front_depth",
                    "agent_0_articulated_agent_jaw_depth",
                    "articulated_agent_jaw_depth",
                    "agent_0_third_depth",
                    "third_depth"
                ]:
                    if possible_key in observations:
                        depth_keys_to_use = [possible_key]
                        break
            
            if len(depth_keys_to_use) == 0:
                raise KeyError(f"No depth sensor found in observations. Available keys: {list(observations.keys())}")
            
            # 处理所有配置的Depth传感器
            for depth_key in depth_keys_to_use:
                if depth_key not in observations:
                    print(f"[WARNING] 配置的Depth传感器 '{depth_key}' 不在observations中，跳过")
                    continue
                
                depth_observations = observations[depth_key][..., 0].float().clamp(0.0, 1.0)  # [BATCH x HEIGHT X WIDTH]
                # if debug_enabled:
                    # print(
                        # f"[ResNetCLIPTextEncoder DEBUG] Depth raw {depth_key}: "
                        # f"shape={tuple(observations[depth_key].shape)} dtype={observations[depth_key].dtype} device={observations[depth_key].device}"
                    # )
                    # print(
                        # f"[ResNetCLIPTextEncoder DEBUG] Depth squeezed {depth_key}: "
                        # f"shape={tuple(depth_observations.shape)} dtype={depth_observations.dtype}"
                    # )
                ddd = torch.stack([depth_observations] * 3, dim=1)  # [BATCH x 3 x HEIGHT X WIDTH]
                # if debug_enabled:
                    # print(
                        # f"[ResNetCLIPTextEncoder DEBUG] Depth expanded {depth_key}: "
                        # f"shape={tuple(ddd.shape)} dtype={ddd.dtype}"
                    # )
                ddd = torch.stack([
                    self.preprocess(TF.convert_image_dtype(depth_map, torch.uint8))
                    for depth_map in ddd
                ])  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
                # if debug_enabled:
                    # print(
                        # f"[ResNetCLIPTextEncoder DEBUG] Depth preprocessed {depth_key}: "
                        # f"shape={tuple(ddd.shape)} dtype={ddd.dtype}"
                    # )
                
                try:
                    depth_x = self.visual_encoder(ddd).float()
                    depth_features_list.append(depth_x)
                except RuntimeError as e:
                    print(
                        f"[ResNetCLIPTextEncoder] Depth传感器 '{depth_key}' 编码失败 - ",
                        f"shape={tuple(ddd.shape)}, dtype={ddd.dtype}, device={ddd.device}",
                    )
                    raise

        # 多视角特征融合
        all_visual_features = rgb_features_list + depth_features_list
        if debug_enabled:
            for feat_idx, feat in enumerate(all_visual_features):
                print(
                    f"[ResNetCLIPTextEncoder DEBUG] visual_feature[{feat_idx}] "
                    f"shape={tuple(feat.shape)} dtype={feat.dtype} device={feat.device}"
                    f" min={feat.min().item():.6f} max={feat.max().item():.6f} mean={feat.mean().item():.6f}"
                )
        
        if len(all_visual_features) == 0:
            raise RuntimeError("No visual features extracted from RGB or Depth sensors")
        elif len(all_visual_features) == 1:
            # 只有一个视角，直接使用
            visual_features = all_visual_features[0]
            if visual_features.dim() == 4:  # [B, C, H, W]
                visual_features = F.adaptive_avg_pool2d(visual_features, 1).flatten(1)
        else:
            # 多视角融合
            # 先将所有特征池化到相同维度 [B, 2048]
            pooled_features = []
            for feat in all_visual_features:
                if feat.dim() == 4:  # [B, C, H, W]
                    feat_pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                else:  # [B, C]
                    feat_pooled = feat
                
                # 可选：融合前归一化
                if self.normalize_before_fusion:
                    feat_pooled = F.normalize(feat_pooled, p=2, dim=1)
                
                pooled_features.append(feat_pooled)
            
            # 根据融合模式进行融合
            if self.visual_fusion_mode == "average":
                # 平均融合（内存友好）
                visual_features = torch.stack(pooled_features, dim=0).mean(dim=0)
            elif self.visual_fusion_mode == "concat":
                # 拼接后投影
                visual_features = torch.cat(pooled_features, dim=1)
                if self.visual_fusion_proj is not None:
                    visual_features = self.visual_fusion_proj(visual_features)
            elif self.visual_fusion_mode == "attention":
                # 自注意力融合
                # [num_views, B, 2048] -> [B, num_views, 2048]
                stacked_features = torch.stack(pooled_features, dim=1)  # [B, num_views, 2048]
                if self.visual_fusion_attn is not None:
                    attn_output, _ = self.visual_fusion_attn(
                        stacked_features, stacked_features, stacked_features
                    )
                    # 平均所有位置的注意力输出
                    visual_features = attn_output.mean(dim=1)  # [B, 2048]
                else:
                    visual_features = stacked_features.mean(dim=1)
            else:
                raise ValueError(f"Unknown visual fusion mode: {self.visual_fusion_mode}")

        if debug_enabled:
            print(
                f"[ResNetCLIPTextEncoder DEBUG] visual_features final "
                f"shape={tuple(visual_features.shape)} dtype={visual_features.dtype} device={visual_features.device}"
            )
            # self._debug_input_print_count += 1  # 已禁用计数

        # 处理文本指令 - 优先从observations中获取
        # 注意：仅在使用CLIP架构（backbone包含"resnet50_clip"）时才会有文本处理功能
        text_instructions = None
        
        # 首先尝试从observations中获取instruction（支持多种传感器键名，按优先级排序）
        instruction_key = None
        for key in [
            "agent_0_falcon_instruction",
            "falcon_instruction",
            "instruction_sensor"
        ]:
            if key in observations:
                instruction_key = key
                break
        
        if instruction_key is not None:
            # 每个 batch 元素分别解码，避免把第 0 条指令复制给整个 batch。
            text_instructions = self._decode_instruction_batch(observations[instruction_key])
        
        # 如果从observations中没有获取到，则回退到从文件加载
        if text_instructions is None and episode_ids is not None:
            text_instructions = []
            for episode_id in episode_ids:
                instruction = self.load_text_instruction(episode_id)
                text_instructions.append(instruction if instruction else "Navigate to the target location.")
        
        # 如果没有获取到任何文本指令，使用默认指令
        if text_instructions is None:
            text_instructions = ["Navigate to the target location."]
        
        # 监控指令变化和episode信息（只在指令变化或第一次运行时打印）
        instruction_changed = not hasattr(self, '_last_instruction') or self._last_instruction != text_instructions
        
        if instruction_changed:
            # print(f"\n=== 读取的文本指令 ===")
            # print(f"指令内容: {text_instructions}")
            # 判断指令来源
            source = "默认"
            # 检查多种可能的传感器键名
            instruction_key_for_source = None
            for key in ["agent_0_falcon_instruction", "falcon_instruction"]:
                if key in observations:
                    instruction_key_for_source = key
                    break
            
            if instruction_key_for_source is not None:
                try:
                    instruction_data = observations[instruction_key_for_source]
                    if isinstance(instruction_data, np.ndarray):
                        instruction_str = instruction_data.tobytes().decode('utf-8').rstrip('\x00')
                        if instruction_str and instruction_str != "Navigate to the target location.":
                            source = f"observations ({instruction_key_for_source})"
                    elif isinstance(instruction_data, torch.Tensor):
                        try:
                            instruction_np = instruction_data.detach().cpu().numpy() if instruction_data.is_cuda else instruction_data.numpy()
                            # 处理不同形状的tensor
                            if len(instruction_np.shape) == 2:
                                instruction_np = instruction_np[0]
                            elif len(instruction_np.shape) > 2:
                                instruction_np = instruction_np.flatten()
                            
                            # 转换为uint8类型
                            instruction_np = instruction_np.astype(np.uint8)
                            
                            instruction_bytes = instruction_np.tobytes()
                            null_pos = instruction_bytes.find(b'\x00')
                            if null_pos >= 0:
                                instruction_bytes = instruction_bytes[:null_pos]
                            
                            try:
                                instruction_str = instruction_bytes.decode('utf-8').strip()
                            except UnicodeDecodeError:
                                instruction_str = instruction_bytes.decode('latin-1').strip()
                                instruction_str = ''.join(c for c in instruction_str if 32 <= ord(c) <= 126)
                            
                            if instruction_str and instruction_str != "Navigate to the target location.":
                                source = f"observations ({instruction_key_for_source})"
                        except Exception:
                            pass
                except:
                    pass
            elif episode_ids is not None:
                source = "文件"
            
            # print(f"来源: {source}")
            # print(f"=== === ===\n")

        self._last_instruction = text_instructions

        # DEBUG: Print decoded instruction (已禁用)
        # if hasattr(self, "_debug_input_print_count") and self._debug_input_print_count < 10:
        #     print(f"[ResNetCLIPTextEncoder DEBUG] decoded instruction: [{text_instructions}]")
        #     print(f"[ResNetCLIPTextEncoder DEBUG] instruction changed since last step: {instruction_changed}")

        # 编码文本指令
        text_features = self.encode_text(text_instructions)

        if debug_enabled and text_features is not None:
            print(
                f"[ResNetCLIPTextEncoder DEBUG] text_features "
                f"shape={tuple(text_features.shape)} dtype={text_features.dtype}"
                f" min={text_features.min().item():.6f} max={text_features.max().item():.6f} mean={text_features.mean().item():.6f}"
            )

        if text_features is not None:
            # 使用多头注意力机制融合视觉和文本特征
            # 1. 投影特征到统一维度
            # visual_proj = self.visual_projection(visual_features)  # [B, 2048]
            text_proj = text_features  # [B, 2048] (已经在encode_text中投影)
            
            # 2. 准备注意力输入 - 视觉作为query，文本作为key和value
            visual_query = visual_features.unsqueeze(1)  # [B, 1, 2048]
            
            # 确保文本特征的批次大小与视觉特征匹配
            batch_size = visual_features.shape[0]
            if text_proj.shape[0] != batch_size:
                if text_proj.shape[0] == 1:
                    text_proj = text_proj.expand(batch_size, -1)
                else:
                    repeat_count = int(np.ceil(batch_size / text_proj.shape[0]))
                    text_proj = text_proj.repeat(repeat_count, 1)[:batch_size]
            
            text_key_value = text_proj.unsqueeze(1)  # [B, 1, 2048]
            
            # 3. 多头注意力融合
            fused_features, attention_weights = self.cross_modal_attention(
                query=visual_query,
                key=text_key_value,
                value=text_key_value
            )  # [B, 1, 2048]
            
            # 4. 输出投影和残差连接
            fused_features = fused_features.squeeze(1)  # [B, 2048]
            fused_features = self.output_projection(fused_features) + visual_features  # 残差连接
            
        else:
            # 没有文本指令时，只使用视觉特征
            fused_features = visual_features

        # 添加训练监控
        if hasattr(self, '_monitor_training') and self._monitor_training:
            # 只记录观察信息，不记录rewards（rewards在训练器层面）
            self._log_observation_info(observations)
        
        return fused_features
    
    def _log_observation_info(self, observations):
        """记录观察信息（避免频繁打印）"""
        if not hasattr(self, '_log_step_count'):
            self._log_step_count = 0
        
        self._log_step_count += 1
        
        # # 每1000步打印一次观察信息
        # if self._log_step_count % 1000 == 0:
        #     print(f"\n=== 观察信息 (Step {self._log_step_count}) ===")
        #     if observations is not None:
        #         print(f"观察形状: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in observations.items()]}")
    
    def log_training_metrics(self, observations, actions, rewards, losses=None):
        """记录训练指标"""
        if not hasattr(self, '_step_count'):
            self._step_count = 0
            self._episode_count = 0
            self._total_reward = 0
            self._action_counts = {}
        
        self._step_count += 1
        
        # 记录动作统计
        if actions is not None:
            action_str = str(actions.cpu().numpy() if hasattr(actions, 'cpu') else actions)
            self._action_counts[action_str] = self._action_counts.get(action_str, 0) + 1
        
        # 记录奖励
        if rewards is not None:
            self._total_reward += float(rewards.mean() if hasattr(rewards, 'mean') else rewards)
        
        # 每100步打印一次统计信息
        if self._step_count % 100 == 0:
            print(f"\n=== 训练统计 (Step {self._step_count}) ===")
            print(f"总奖励: {self._total_reward:.4f}")
            print(f"平均奖励: {self._total_reward / self._step_count:.4f}")
            print(f"动作分布: {dict(list(self._action_counts.items())[:5])}")  # 显示前5个动作
            
            if losses is not None:
                print(f"损失信息: {losses}")
            
            # 打印当前观察信息
            if observations is not None:
                print(f"观察形状: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in observations.items()]}")
    
    def log_episode_info(self, episode_id, success=False, episode_length=0):
        """记录episode信息"""
        if not hasattr(self, '_episode_count'):
            self._episode_count = 0
        
        self._episode_count += 1
        
        print(f"\n=== Episode {self._episode_count} 完成 ===")
        print(f"Episode ID: {episode_id}")
        print(f"成功: {success}")
        print(f"长度: {episode_length}")
        print(f"当前指令: {getattr(self, '_last_instruction', 'N/A')}")
        print(f"时间: {torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else 'CPU'}")

class DINOv3TextFusionEncoder(nn.Module):
    def __init__(self, observation_space: spaces.Dict, text_instruction_path: str = None):
        super().__init__()
        from dinov3.dinov3.eval.text.text_tower import TextTower
        from dinov3.dinov3.eval.text.vision_tower import VisionTower
        from dinov3.dinov3.eval.text.dinotxt_model import DINOTextModel

        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces
        self.text_instruction_path = text_instruction_path

        # 初始化视觉塔
        self.vision_tower = VisionTower(model_name="dinov3-vitb16", pretrained=True)

        # 初始化文本塔
        self.text_tower = TextTower(model_name="bert-base-uncased", pretrained=True)

        # 构建融合模型
        self.fusion_model = DINOTextModel(
            vision_tower=self.vision_tower,
            text_tower=self.text_tower,
            fusion_dim=768,  # 可调
            use_attention=True  # 启用注意力融合
        )

        self.output_shape = (768,)  # 输出特征维度

    def forward(self, observations: Dict[str, torch.Tensor], episode_ids: List[str] = None):
        # 提取图像和深度
        images = []
        if self.rgb:
            images.append(observations["overhead_front_rgb"])
        if self.depth:
            images.append(observations["overhead_front_depth"])
        image_input = torch.cat(images, dim=1) if len(images) > 1 else images[0]

        # 提取文本指令
        text = self._extract_text(observations.get("falcon_instruction", None))

        # 融合特征
        fused_features = self.fusion_model(image_input, text)
        return fused_features

    def _extract_text(self, instruction_data):
        if isinstance(instruction_data, torch.Tensor):
            try:
                # 处理不同形状的tensor
                instruction_np = instruction_data.detach().cpu().numpy()
                if len(instruction_np.shape) == 2:
                    instruction_np = instruction_np[0]
                elif len(instruction_np.shape) > 2:
                    instruction_np = instruction_np.flatten()
                
                # 转换为uint8类型
                instruction_np = instruction_np.astype(np.uint8)
                
                # 转换为字节并解码
                instruction_bytes = instruction_np.tobytes()
                null_pos = instruction_bytes.find(b'\x00')
                if null_pos >= 0:
                    instruction_bytes = instruction_bytes[:null_pos]
                
                try:
                    return instruction_bytes.decode('utf-8').strip()
                except UnicodeDecodeError:
                    # 如果UTF-8解码失败，使用latin-1并过滤
                    instruction_str = instruction_bytes.decode('latin-1').strip()
                    return ''.join(c for c in instruction_str if 32 <= ord(c) <= 126)
            except Exception:
                return "Navigate to the target location."
        elif isinstance(instruction_data, str):
            return instruction_data
        return "Navigate to the target location."
