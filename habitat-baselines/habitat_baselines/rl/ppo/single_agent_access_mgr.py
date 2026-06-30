from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import sys
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.rollout_storage import (  # noqa: F401.
    RolloutStorage,
)
from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ddppo.policy import (
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.ppo import PPO


def extract_state_dict(pretrained_state):
    """
    从不同格式的checkpoint中提取state_dict

    支持的格式:
    1. [{int: {"state_dict": ...}}, ...] - 列表包含整数key的dict（如DDPPO多agent格式）
    2. {"state_dict": ...} - 字典格式
    3. 直接的state_dict - 直接格式
    """
    if isinstance(pretrained_state, list):
        # 格式1: [{"state_dict": ...}] 或 [state_dict]
        first = pretrained_state[0] if len(pretrained_state) > 0 else {}
        if isinstance(first, dict) and "state_dict" in first:
            return first["state_dict"]
        return first

    if isinstance(pretrained_state, dict):
        # 格式2: {0: {"state_dict": ...}, 1: {"state_dict": ...}, ...}
        # 检测是否是整数key + state_dict结构（DDPPO多agent格式）
        int_keys = [k for k in pretrained_state.keys() if isinstance(k, int)]
        if int_keys:
            # 取第一个agent的state_dict
            first_int_key = min(int_keys)
            inner = pretrained_state[first_int_key]
            if isinstance(inner, dict) and "state_dict" in inner:
                return inner["state_dict"]
            if isinstance(inner, dict) and "state_dict" in inner:
                return inner["state_dict"]
            return inner

        # 常见格式: {"state_dict": ...}
        if "state_dict" in pretrained_state:
            return pretrained_state["state_dict"]
        # Habitat/自定义保存格式: {"actor_critic": ...}
        if "actor_critic" in pretrained_state:
            return pretrained_state["actor_critic"]
        # 其它格式: {"model": ...}
        if "model" in pretrained_state:
            return pretrained_state["model"]

    # 格式3: 直接是 state_dict
    return pretrained_state
from habitat_baselines.rl.ppo.updater import Updater

if TYPE_CHECKING:
    from omegaconf import DictConfig


def linear_lr_schedule(percent_done: float) -> float:
    return 1 - percent_done


@baseline_registry.register_agent_access_mgr
class SingleAgentAccessMgr(AgentAccessMgr):
    def __init__(
        self,
        config: "DictConfig",
        env_spec: EnvironmentSpec,
        is_distrib: bool,
        device,
        num_envs: int,
        percent_done_fn: Callable[[], float],
        resume_state: Optional[Dict[str, Any]] = None,
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
        agent_name=None,
    ):
        """
        :param percent_done_fn: Function that will return the percent of the
            way through training.
        :param lr_schedule_fn: For a learning rate schedule. ONLY used if
            specified in the config. Takes as input the current progress in
            training and returns the learning rate multiplier. The default behavior
            is to use `linear_lr_schedule`.
        :param agent_name: the name of the agent for which we set the singleagentaccessmanager
        """

        self._env_spec = env_spec
        self._config = config
        self._num_envs = num_envs
        self._device = device
        self._ppo_cfg = self._config.habitat_baselines.rl.ppo
        self._is_distributed = is_distrib
        self._is_static_encoder = (
            not config.habitat_baselines.rl.ddppo.train_encoder
        )

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you should specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        self.agent_name = agent_name
        self._nbuffers = 2 if self._ppo_cfg.use_double_buffered_sampler else 1
        self._percent_done_fn = percent_done_fn
        if lr_schedule_fn is None:
            lr_schedule_fn = linear_lr_schedule
        self._init_policy_and_updater(lr_schedule_fn, resume_state)

    def _init_policy_and_updater(self, lr_schedule_fn, resume_state):
        self._actor_critic = self._create_policy()
        self._updater = self._create_updater(self._actor_critic)

        if self._updater.optimizer is None:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = LambdaLR(
                optimizer=self._updater.optimizer,
                lr_lambda=lambda _: lr_schedule_fn(self._percent_done_fn()),
            )
        if resume_state is not None:
            resume_weights_only = bool(
                getattr(self._config.habitat_baselines, "resume_weights_only", False)
            ) or any(
                arg.strip() == "habitat_baselines.resume_weights_only=True"
                for arg in sys.argv
            )
            print(
                f"[SingleAgentAccessMgr] resume_state detected, resume_weights_only={resume_weights_only}"
            )
            if not resume_weights_only:
                self._updater.load_state_dict(resume_state)
            elif "state_dict" in resume_state:
                model_state = resume_state["state_dict"]
                if isinstance(model_state, dict):
                    self.load_state_dict({"state_dict": model_state})

    @property
    def masks_shape(self) -> Tuple:
        return (1,)

    @property
    def nbuffers(self):
        return self._nbuffers

    def _create_storage(
        self,
        num_envs: int,
        env_spec: EnvironmentSpec,
        actor_critic: NetPolicy,
        policy_action_space: spaces.Space,
        config: "DictConfig",
        device,
    ) -> Storage:
        """
        Default behavior for setting up and initializing the rollout storage.
        """

        obs_space = get_rollout_obs_space(
            env_spec.observation_space, actor_critic, config
        )
        ppo_cfg = config.habitat_baselines.rl.ppo
        rollouts = baseline_registry.get_storage(
            config.habitat_baselines.rollout_storage_name
        )(
            numsteps=ppo_cfg.num_steps,
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=policy_action_space,
            actor_critic=actor_critic,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
        )
        rollouts.to(device)
        return rollouts

    def post_init(self, create_rollouts_fn: Optional[Callable] = None) -> None:
        # Create the rollouts storage.
        if create_rollouts_fn is None:
            create_rollouts_fn = self._create_storage

        policy_action_space = self._actor_critic.policy_action_space
        self._rollouts = create_rollouts_fn(
            num_envs=self._num_envs,
            env_spec=self._env_spec,
            actor_critic=self._actor_critic,
            policy_action_space=policy_action_space,
            config=self._config,
            device=self._device,
        )

    def _create_updater(self, actor_critic) -> PPO:
        if self._is_distributed:
            updater_cls = baseline_registry.get_updater(
                self._config.habitat_baselines.distrib_updater_name
            )
        else:
            updater_cls = baseline_registry.get_updater(
                self._config.habitat_baselines.updater_name
            )

        updater = updater_cls.from_config(actor_critic, self._ppo_cfg)
        logger.info(
            "Agent number of parameters: {}".format(
                sum(param.numel() for param in updater.parameters())
            )
        )
        return updater

    def init_distributed(self, find_unused_params: bool = True) -> None:
        if len(list(self._updater.parameters())) > 0:
            self._updater.init_distributed(
                find_unused_params=find_unused_params
            )

    def _create_policy(self) -> NetPolicy:
        """
        Creates and initializes the policy. This should also load any model weights from checkpoints.
        """

        policy = baseline_registry.get_policy(
            self._config.habitat_baselines.rl.policy[self.agent_name].name
        )
        if policy is None:
            raise ValueError(
                f"Couldn't find policy {self._config.habitat_baselines.rl.policy[self.agent_name].name}"
            )
        actor_critic = policy.from_config(
            self._config,
            self._env_spec.observation_space,
            self._env_spec.action_space,
            orig_action_space=self._env_spec.orig_action_space,
            agent_name=self.agent_name,
        )
        if (
            self._config.habitat_baselines.rl.ddppo.pretrained_encoder
            or self._config.habitat_baselines.rl.ddppo.pretrained
        ):
            pretrained_state = torch.load(
                self._config.habitat_baselines.rl.ddppo.pretrained_weights,
                map_location="cpu",
                weights_only=False,
            )

        # adapt to multi-agent setup
        if self._config.habitat_baselines.rl.ddppo.pretrained and (self.agent_name == "agent_0" or self.agent_name == "main_agent") : 
            # 检查是否使用CLIP架构
            backbone = getattr(self._config.habitat_baselines.rl.ddppo, 'backbone', 'resnet50')
            is_clip_architecture = backbone in ['resnet50_clip_text', 'resnet50_clip_attnpool']
            
            if is_clip_architecture:
                # CLIP架构：选择性加载权重
                print("检测到CLIP架构，使用选择性权重加载...")
                print("  - 跳过ResNet视觉编码器权重（使用CLIP预训练权重）")
                print("  - 加载其他模块权重（LSTM、策略头等）")
                model_state_dict = actor_critic.state_dict()
                
                # 过滤权重：跳过ResNet视觉编码器权重，保留其他兼容权重
                filtered_pretrained_state_dict = {}
                skipped_resnet_weights = 0
                skipped_clip_weights = 0
                loaded_weights = 0
                skipped_shape_mismatch = 0
                
                # 提取state_dict（兼容不同格式）
                state_dict_to_load = extract_state_dict(pretrained_state)
                
                for k, v in state_dict_to_load.items():
                    if not isinstance(k, str):
                        continue
                    key = k[len("actor_critic."):] if k.startswith("actor_critic.") else k
                    
                    # 跳过ResNet视觉编码器权重（使用CLIP预训练权重）
                    if (key.startswith("net.visual_encoder.backbone.") or 
                        key.startswith("net.visual_encoder.compression.") or
                        key.startswith("net.visual_encoder.running_mean_and_var.")):
                        skipped_resnet_weights += 1
                        continue
                    
                    # 跳过CLIP相关的新增权重（这些权重在原始检查点中不存在）
                    if (key.startswith("net.visual_encoder.clip_model.") or 
                        key.startswith("net.visual_encoder.visual_encoder.") or
                        key.startswith("net.visual_encoder.text_projection.") or
                        key.startswith("net.visual_encoder.cross_modal_attention.") or
                        key.startswith("net.visual_encoder.visual_projection.") or
                        key.startswith("net.visual_encoder.output_projection.")):
                        skipped_clip_weights += 1
                        continue
                    
                    # 只加载形状匹配的权重
                    if key in model_state_dict and v.shape == model_state_dict[key].shape:
                        filtered_pretrained_state_dict[key] = v
                        loaded_weights += 1
                    else:
                        skipped_shape_mismatch += 1
                
                model_state_dict.update(filtered_pretrained_state_dict)
                missing_keys, unexpected_keys = actor_critic.load_state_dict(model_state_dict, strict=False)
                
                print(f"CLIP架构权重加载统计：")
                print(f"  - 跳过ResNet视觉编码器权重: {skipped_resnet_weights}")
                print(f"  - 跳过CLIP新增权重: {skipped_clip_weights}")
                print(f"  - 跳过形状不匹配: {skipped_shape_mismatch}")
                print(f"  - 成功加载权重: {loaded_weights}")
                # print(f"  - 总权重数: {len(pretrained_state['state_dict'])}")
                
                # 详细权重检查
                print(f"\n=== 详细权重检查 ===")
                print(f"Missing keys (模型需要但检查点没有): {len(missing_keys)}")
                if missing_keys:
                    print("Missing keys详情:")
                    for key in missing_keys[:10]:  # 只显示前10个
                        print(f"  - {key}")
                    if len(missing_keys) > 10:
                        print(f"  ... 还有 {len(missing_keys) - 10} 个")
                
                print(f"Unexpected keys (检查点有但模型不需要): {len(unexpected_keys)}")
                if unexpected_keys:
                    print("Unexpected keys详情:")
                    for key in unexpected_keys[:10]:  # 只显示前10个
                        print(f"  - {key}")
                    if len(unexpected_keys) > 10:
                        print(f"  ... 还有 {len(unexpected_keys) - 10} 个")
                
                # 检查可训练参数
                total_params = sum(p.numel() for p in actor_critic.parameters())
                trainable_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
                frozen_params = total_params - trainable_params
                
                print(f"\n=== 参数统计 ===")
                print(f"总参数数量: {total_params:,}")
                print(f"可训练参数: {trainable_params:,}")
                print(f"冻结参数: {frozen_params:,}")
                print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
                
                # 检查各模块参数
                print(f"\n=== 各模块参数统计 ===")
                for name, module in actor_critic.named_children():
                    module_params = sum(p.numel() for p in module.parameters())
                    module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    print(f"{name}: 总参数={module_params:,}, 可训练={module_trainable:,}")
                
                # 检查CLIP相关模块
                if hasattr(actor_critic, 'net') and hasattr(actor_critic.net, 'visual_encoder'):
                    visual_encoder = actor_critic.net.visual_encoder
                    if hasattr(visual_encoder, 'clip_model'):
                        clip_params = sum(p.numel() for p in visual_encoder.clip_model.parameters())
                        clip_trainable = sum(p.numel() for p in visual_encoder.clip_model.parameters() if p.requires_grad)
                        print(f"CLIP模型: 总参数={clip_params:,}, 可训练={clip_trainable:,}")
                    
                    if hasattr(visual_encoder, 'text_projection'):
                        text_proj_params = sum(p.numel() for p in visual_encoder.text_projection.parameters())
                        text_proj_trainable = sum(p.numel() for p in visual_encoder.text_projection.parameters() if p.requires_grad)
                        print(f"文本投影层: 总参数={text_proj_params:,}, 可训练={text_proj_trainable:,}")
                    
                    if hasattr(visual_encoder, 'cross_modal_attention'):
                        attn_params = sum(p.numel() for p in visual_encoder.cross_modal_attention.parameters())
                        attn_trainable = sum(p.numel() for p in visual_encoder.cross_modal_attention.parameters() if p.requires_grad)
                        print(f"跨模态注意力: 总参数={attn_params:,}, 可训练={attn_trainable:,}")
                
                print(f"=== 权重检查完成 ===\n")
                
            elif not is_clip_architecture:
                # ResNet架构：正常加载所有权重
                print("检测到ResNet架构，加载完整预训练权重...")
                print("  - 加载ResNet视觉编码器权重")
                print("  - 加载LSTM、策略头等其他模块权重")
                
                state_dict_to_load = extract_state_dict(pretrained_state)
                missing_keys, unexpected_keys = actor_critic.load_state_dict(
                    { 
                        k[len("actor_critic.") :]: v
                        for k, v in state_dict_to_load.items()
                    },
                    strict=False,
                )
                
                print(f"\n=== 权重加载检查 (ResNet架构) ===")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
                # 参数统计
                total_params = sum(p.numel() for p in actor_critic.parameters())
                trainable_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
                print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
                print(f"=== 权重检查完成 ===\n")
                
            elif "oracle_humanoid_future_trajectory" in self._env_spec.observation_space.spaces:
                model_state_dict = actor_critic.state_dict()
                state_dict_to_load = extract_state_dict(pretrained_state)
                filtered_pretrained_state_dict = {k[len("actor_critic.") :]: v for k, v in state_dict_to_load.items() if k[len("actor_critic.") :] in model_state_dict and v.shape == model_state_dict[k[len("actor_critic.") :]].shape}
                model_state_dict.update(filtered_pretrained_state_dict)
                missing_keys, unexpected_keys = actor_critic.load_state_dict(model_state_dict, strict=False)
                
                print(f"\n=== 权重加载检查 (Oracle Humanoid) ===")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
                # 参数统计
                total_params = sum(p.numel() for p in actor_critic.parameters())
                trainable_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
                print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
                print(f"=== 权重检查完成 ===\n")
            elif self._config.habitat_baselines.rl.auxiliary_losses:
                model_state_dict = actor_critic.state_dict()
                state_dict_to_load = extract_state_dict(pretrained_state)
                filtered_pretrained_state_dict = {k[len("actor_critic.") :]: v for k, v in state_dict_to_load.items() if k[len("actor_critic.") :] in model_state_dict and v.shape == model_state_dict[k[len("actor_critic.") :]].shape}
                model_state_dict.update(filtered_pretrained_state_dict)
                missing_keys, unexpected_keys = actor_critic.load_state_dict(model_state_dict, strict=False)
                
                print(f"\n=== 权重加载检查 (Auxiliary Losses) ===")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
                # 参数统计
                total_params = sum(p.numel() for p in actor_critic.parameters())
                trainable_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
                print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
                print(f"=== 权重检查完成 ===\n")
            else:
                state_dict_to_load = extract_state_dict(pretrained_state)
                missing_keys, unexpected_keys = actor_critic.load_state_dict(
                        { 
                            k[len("actor_critic.") :]: v
                            for k, v in state_dict_to_load.items()
                        },
                        strict=False,
                    )
                
                print(f"\n=== 权重加载检查 (标准模式) ===")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
                # 参数统计
                total_params = sum(p.numel() for p in actor_critic.parameters())
                trainable_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
                print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
                print(f"=== 权重检查完成 ===\n")
        elif self._config.habitat_baselines.rl.ddppo.pretrained_encoder and hasattr(actor_critic, 'net'):
            prefix = "actor_critic.net.visual_encoder."
            state_dict_to_load = extract_state_dict(pretrained_state)
            actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in state_dict_to_load.items()
                    if k.startswith(prefix)
                }
            )
        if self._is_static_encoder:
            # Handle both standard and CLIP architectures
            visual_encoder = None
            # Try standard architecture first
            if hasattr(actor_critic, 'visual_encoder'):
                visual_encoder = actor_critic.visual_encoder
            # Try CLIP architecture (visual_encoder is under net)
            elif hasattr(actor_critic, 'net') and hasattr(actor_critic.net, 'visual_encoder'):
                visual_encoder = actor_critic.net.visual_encoder
            
            if visual_encoder is not None:
                for param in visual_encoder.parameters():
                    param.requires_grad_(False)

        if self._config.habitat_baselines.rl.ddppo.reset_critic and hasattr(actor_critic,"critic"):
            nn.init.orthogonal_(actor_critic.critic.fc.weight)
            nn.init.constant_(actor_critic.critic.fc.bias, 0)

        actor_critic.to(self._device)
        return actor_critic

    @property
    def rollouts(self) -> Storage:
        return self._rollouts

    @property
    def actor_critic(self) -> NetPolicy:
        return self._actor_critic

    @property
    def updater(self) -> Updater:
        return self._updater

    def get_resume_state(self) -> Dict[str, Any]:
        # If there is nothing to load, then we return the empty dict
        if self._updater.optimizer is None:
            return {"state_dict": {}, "optim_state": {}}
        ret = {
            "state_dict": self._actor_critic.state_dict(),
            **self._updater.get_resume_state(),
        }
        if self._lr_scheduler is not None:
            ret["lr_sched_state"] = self._lr_scheduler.state_dict()
        return ret

    def get_save_state(self):
        return {"state_dict": self._actor_critic.state_dict()}

    def eval(self):
        self._actor_critic.eval()

    def train(self):
        self._actor_critic.train()
        self._updater.train()

    def load_ckpt_state_dict(self, ckpt: Dict) -> None:
        self._actor_critic.load_state_dict(ckpt["state_dict"])

    def load_state_dict(self, state: Dict, strict: bool = False):
        # 处理IL checkpoint（扁平结构）和RL checkpoint（包含'state_dict'键）
        # IL checkpoint: state 直接是模型参数字典
        # RL checkpoint: state["state_dict"] 是模型参数字典
        has_training_state = False
        if "state_dict" in state:
            # RL checkpoint格式
            model_state = state["state_dict"]
            has_training_state = any(
                k in state
                for k in (
                    "optim_state",
                    "optimizer_state_dict",
                    "lr_sched_state",
                    "reference_state_dict",
                )
            )
        else:
            # IL checkpoint格式（扁平结构）
            model_state = state
        missing_keys = []
        unexpected_keys = []
        
        # 检查是否使用CLIP架构
        backbone = getattr(self._config.habitat_baselines.rl.ddppo, 'backbone', 'resnet50')
        is_clip_architecture = backbone in ['resnet50_clip_text', 'resnet50_clip_attnpool']
        
        if is_clip_architecture:
            # 检查checkpoint是否包含CLIP权重
            has_clip_weights = any(k.startswith("net.visual_encoder.clip_model.") or
                                   k.startswith("net.visual_encoder.visual_encoder.") or
                                   k.startswith("net.visual_encoder.text_projection.") 
                                   for k in model_state.keys())
            
            if has_clip_weights:
                # 情况B：checkpoint已经是CLIP架构，直接加载所有权重
                print("CLIP架构检查点加载：检测到checkpoint包含CLIP权重，直接加载...")
                model_state_dict = self._actor_critic.state_dict()
                filtered_state_dict = {}
                loaded_weights = 0
                skipped_shape_mismatch = 0
                shape_mismatch_details = []
                
                for k, v in model_state.items():
                    if k in model_state_dict:
                        if v.shape == model_state_dict[k].shape:
                            filtered_state_dict[k] = v
                            loaded_weights += 1
                        else:
                            # 形状不匹配
                            skipped_shape_mismatch += 1
                            shape_mismatch_details.append(
                                (k, v.shape, model_state_dict[k].shape)
                            )
                    else:
                        # 键不存在于当前模型
                        skipped_shape_mismatch += 1
                        shape_mismatch_details.append(
                            (k, v.shape, "不存在")
                        )
                
                model_state_dict.update(filtered_state_dict)
                missing_keys, unexpected_keys = self._actor_critic.load_state_dict(
                    model_state_dict, strict=False
                )
                
                print(f"CLIP架构检查点加载统计：")
                print(f"  - 成功加载权重: {loaded_weights}")
                print(f"  - 跳过形状不匹配: {skipped_shape_mismatch}")
                
                # # 详细的形状不匹配信息（已注释，需要时可启用）
                # if skipped_shape_mismatch > 0:
                #     print(f"\n{'='*80}")
                #     print(f"形状不匹配详情 (共 {skipped_shape_mismatch} 个)：")
                #     print(f"{'='*80}")
                #     for i, (key, ckpt_shape, model_shape) in enumerate(shape_mismatch_details[:20], 1):
                #         if model_shape == "不存在":
                #             print(f"{i:2d}. {key}")
                #             print(f"    Checkpoint: {ckpt_shape} → Model: [不存在]")
                #         else:
                #             print(f"{i:2d}. {key}")
                #             print(f"    Checkpoint: {ckpt_shape} → Model: {model_shape}")
                #     
                #     if len(shape_mismatch_details) > 20:
                #         print(f"\n    ... 还有 {len(shape_mismatch_details) - 20} 个未显示")
                #     print(f"{'='*80}")
                
                # # Missing和Unexpected keys（已注释，需要时可启用）
                # print(f"\n{'='*80}")
                # print(f"详细权重检查：")
                # print(f"{'='*80}")
                # print(f"Missing keys (模型需要但checkpoint没有): {len(missing_keys)}")
                # if missing_keys and len(missing_keys) <= 30:
                #     print("Missing keys 详情:")
                #     for i, key in enumerate(missing_keys, 1):
                #         print(f"  {i:2d}. {key}")
                # elif len(missing_keys) > 30:
                #     print("Missing keys 详情 (前30个):")
                #     for i, key in enumerate(missing_keys[:30], 1):
                #         print(f"  {i:2d}. {key}")
                #     print(f"  ... 还有 {len(missing_keys) - 30} 个")
                # 
                # print(f"\nUnexpected keys (checkpoint有但模型不需要): {len(unexpected_keys)}")
                # if unexpected_keys and len(unexpected_keys) <= 30:
                #     print("Unexpected keys 详情:")
                #     for i, key in enumerate(unexpected_keys, 1):
                #         print(f"  {i:2d}. {key}")
                # elif len(unexpected_keys) > 30:
                #     print("Unexpected keys 详情 (前30个):")
                #     for i, key in enumerate(unexpected_keys[:30], 1):
                #         print(f"  {i:2d}. {key}")
                #     print(f"  ... 还有 {len(unexpected_keys) - 30} 个")
                # print(f"{'='*80}\n")
            else:
                # 情况A：checkpoint是ResNet架构，需要跳过ResNet权重
                print("CLIP架构检查点加载：从ResNet checkpoint迁移，跳过ResNet权重...")
            print("  - 跳过ResNet视觉编码器权重（使用CLIP预训练权重）")
            print("  - 加载其他模块权重（LSTM、策略头等）")
            model_state_dict = self._actor_critic.state_dict()
            
            # 过滤权重：跳过ResNet视觉编码器权重，保留其他兼容权重
            filtered_state_dict = {}
            skipped_resnet_weights = 0
            skipped_clip_weights = 0
            loaded_weights = 0
            skipped_shape_mismatch = 0
            
            for k, v in model_state.items():
                # 跳过ResNet视觉编码器权重（使用CLIP预训练权重）
                if (k.startswith("net.visual_encoder.backbone.") or 
                    k.startswith("net.visual_encoder.compression.") or
                    k.startswith("net.visual_encoder.running_mean_and_var.")):
                    skipped_resnet_weights += 1
                    continue
                
                # 跳过CLIP相关的新增权重（这些权重在原始检查点中不存在）
                if (k.startswith("net.visual_encoder.clip_model.") or 
                    k.startswith("net.visual_encoder.visual_encoder.") or
                    k.startswith("net.visual_encoder.text_projection.") or
                    k.startswith("net.visual_encoder.cross_modal_attention.") or
                    k.startswith("net.visual_encoder.visual_projection.") or
                    k.startswith("net.visual_encoder.output_projection.")):
                    skipped_clip_weights += 1
                    continue
                
                # 只加载形状匹配的权重
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                    loaded_weights += 1
                else:
                    skipped_shape_mismatch += 1
            
            model_state_dict.update(filtered_state_dict)
            missing_keys, unexpected_keys = self._actor_critic.load_state_dict(
                model_state_dict, strict=False
            )
            print(f"CLIP架构检查点加载统计：")
            print(f"  - 跳过ResNet视觉编码器权重: {skipped_resnet_weights}")
            print(f"  - 跳过CLIP新增权重: {skipped_clip_weights}")
            print(f"  - 跳过形状不匹配: {skipped_shape_mismatch}")
            print(f"  - 成功加载权重: {loaded_weights}")
        else:
            # ResNet架构：正常加载所有权重
            print("ResNet架构检查点加载：加载完整预训练权重...")
            missing_keys, unexpected_keys = self._actor_critic.load_state_dict(
                model_state, strict=strict
            )

        if hasattr(self._updater, "set_reference_policy"):
            self._updater.set_reference_policy(self._actor_critic)

        if (
            has_training_state
            and self._updater is not None
            and not getattr(
            self._config.habitat_baselines, "resume_weights_only", False
            )
        ):
            self._updater.load_state_dict(state)
            if "lr_sched_state" in state:
                self._lr_scheduler.load_state_dict(state["lr_sched_state"])

        return missing_keys, unexpected_keys

    def after_update(self):
        if (
            self._ppo_cfg.use_linear_lr_decay
            and self._lr_scheduler is not None
        ):
            self._lr_scheduler.step()  # type: ignore
        self._updater.after_update()

    def pre_rollout(self):
        if self._ppo_cfg.use_linear_clip_decay:
            self._updater.clip_param = self._ppo_cfg.clip_param * (
                1 - self._percent_done_fn()
            )


def get_rollout_obs_space(obs_space, actor_critic, config):
    """
    Helper to get the observation space for the rollout storage when using a
    frozen visual encoder.
    """

    if not config.habitat_baselines.rl.ddppo.train_encoder:
        encoder = actor_critic.visual_encoder
        obs_space = spaces.Dict(
            {
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY: spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=encoder.output_shape,
                    dtype=np.float32,
                ),
                **obs_space.spaces,
            }
        )
    return obs_space
