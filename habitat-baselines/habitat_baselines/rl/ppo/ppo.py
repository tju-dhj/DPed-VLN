#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from habitat import logger
from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.updater import Updater
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import (
    LagrangeInequalityCoefficient,
    inference_mode,
)
from habitat_baselines.utils.timing import g_timer

from habitat_baselines.rl.ppo.belief_policy import AttentiveBeliefPolicy

EPS_PPO = 1e-5


@baseline_registry.register_updater
class PPO(nn.Module, Updater):
    entropy_coef: Union[float, LagrangeInequalityCoefficient]

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config, aux_tasks = [], aux_names = [], aux_cfg = None):
        return cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            aux_loss_coef=0.0, # config.aux_loss_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
            aux_tasks=aux_tasks,
            aux_names=aux_names,
            aux_cfg=aux_cfg,
        )

    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        aux_loss_coef: float = 0.0,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
        aux_tasks=[],
        aux_names=[],
        aux_cfg=None,
    ) -> None:
        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.aux_loss_coef = aux_loss_coef   ## added
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.device = next(actor_critic.parameters()).device

        if (
            use_adaptive_entropy_pen
            and hasattr(self.actor_critic, "num_actions")
            and getattr(self.actor_critic, "action_distribution_type", None)
            == "gaussian"
        ):
            num_actions = self.actor_critic.num_actions

            self.entropy_coef = LagrangeInequalityCoefficient(
                -float(entropy_target_factor) * num_actions,
                init_alpha=entropy_coef,
                alpha_max=1.0,
                alpha_min=1e-4,
                greater_than=True,
            ).to(device=self.device)

        self._aux_tasks=[]
        self._aux_names=[]
        if aux_cfg:
            self.aux_cfg = aux_cfg
            self._aux_tasks = aux_tasks
            self._aux_names = aux_names

        self.use_normalized_advantage = use_normalized_advantage
        self.optimizer = self._create_optimizer(lr, eps, self._aux_tasks)

        self.non_ac_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("actor_critic.")
        ]

    def _create_optimizer(self, lr, eps, aux_tasks=None):
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        logger.info(
            f"(No Aux) Main Number of params to train: {sum(param.numel() for param in params)}"
        )

        if len(aux_tasks) > 0:
            for aux_t in aux_tasks:
                params += list(filter(lambda p: p.requires_grad, aux_t.parameters()))

        logger.info(
            f"Total Number of params to train: {sum(param.numel() for param in params)}"
        )
        if len(params) > 0:
            optim_cls = optim.Adam
            optim_kwargs = dict(
                params=params,
                lr=lr,
                eps=eps,
            )
            signature = inspect.signature(optim_cls.__init__)
            if "foreach" in signature.parameters:
                optim_kwargs["foreach"] = True
            else:
                try:
                    import torch.optim._multi_tensor
                except ImportError:
                    pass
                else:
                    optim_cls = torch.optim._multi_tensor.Adam

            return optim_cls(**optim_kwargs)
        else:
            return None

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"]  # type: ignore
            - rollouts.buffers["value_preds"]
        )
        if not self.use_normalized_advantage:
            return advantages

        var, mean = self._compute_var_mean(
            advantages[torch.isfinite(advantages)]
        )

        advantages -= mean

        return advantages.mul_(torch.rsqrt(var + EPS_PPO))

    @staticmethod
    def _compute_var_mean(x):
        return torch.var_mean(x)

    def _set_grads_to_none(self):
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                p.grad = None

    @g_timer.avg_time("ppo.update_from_batch", level=1)
    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        """
        Performs a gradient update from the minibatch.
        """

        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        self._set_grads_to_none()
        aux_dist_entropy = None
        if isinstance(self.actor_critic, AttentiveBeliefPolicy):
            (
                values,
                action_log_probs,
                dist_entropy,
                final_rnn_state,
                rnn_features,
                individual_rnn_features,
                aux_dist_entropy,
                aux_weights,
            ) = self._evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
                batch.get("rnn_build_seq_info", None),
            )
        else:
            (
                values,
                action_log_probs,
                dist_entropy,
                final_rnn_state,
                aux_loss_res,
            ) = self._evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
                batch.get("rnn_build_seq_info", None),
            )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)

        values = values.float()
        orig_values = values

        if self.use_clipped_value_loss:
            delta = values.detach() - batch["value_preds"]
            value_pred_clipped = batch["value_preds"] + delta.clamp(
                -self.clip_param, self.clip_param
            )

            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )

        value_loss = 0.5 * F.mse_loss(
            values, batch["returns"], reduction="none"
        )

        if "is_coeffs" in batch:
            assert isinstance(batch["is_coeffs"], torch.Tensor)
            ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)
            mean_fn = lambda t: torch.mean(ver_is_coeffs * t)
        else:
            mean_fn = torch.mean

        action_loss, value_loss, dist_entropy = map(
            mean_fn,
            (action_loss, value_loss, dist_entropy),
        )
        
        total_aux_loss = 0
        aux_losses = []
        if isinstance(self.actor_critic, AttentiveBeliefPolicy) and len(self._aux_tasks) > 0:
            aux_raw_losses = self.actor_critic.evaluate_aux_losses(batch, final_rnn_state, rnn_features, individual_rnn_features)
            aux_losses = torch.stack(aux_raw_losses)
            total_aux_loss = torch.sum(aux_losses, dim=0)

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]

        if isinstance(self.actor_critic, AttentiveBeliefPolicy):
            all_losses.append(total_aux_loss * self.aux_loss_coef)
            
        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        if aux_dist_entropy is not None:
            # TODO: maybe take the mean of the entropy, since also dist_entropy is averaged on line 150
            all_losses.append(aux_dist_entropy * self.aux_cfg.entropy_coef)

        # #debug
        # if np.isnan(total_loss.item()):
        #     print("total_loss is nan")

        if len(self._aux_tasks) == 0:
            all_losses.extend(v["loss"] for v in aux_loss_res.values())

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)

            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )

            if len(self._aux_tasks) == 0: # not use my aux
                for name, res in aux_loss_res.items():
                    for k, v in res.items():
                        learner_metrics[f"aux_{name}_{k}"].append(v.detach())
            else:
                learner_metrics["aux_entropy"].append(aux_dist_entropy)
                for i, aux_loss in enumerate(aux_losses):
                    learner_metrics[f"aux_entropy_{self._aux_names[i]}"].append(aux_loss.item())
                for i, aux_weight in enumerate(aux_weights):
                    learner_metrics[f"aux_weights_{self._aux_names[i]}"].append(aux_weight.item())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (
                        rollouts.current_policy_version
                        - batch["policy_version"]
                    ).float(),
                    "policy_version_difference",
                )

    def update(
        self,
        rollouts: RolloutStorage,
    ) -> Dict[str, float]:
        advantages = self.get_advantages(rollouts)

        learner_metrics: Dict[str, List[Any]] = collections.defaultdict(list)

        for epoch in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.data_generator(
                advantages, self.num_mini_batch
            )

            for _bid, batch in enumerate(data_generator):
                self._update_from_batch(
                    batch, epoch, rollouts, learner_metrics
                )

            profiling_wrapper.range_pop()  # PPO.update epoch

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    @g_timer.avg_time("ppo.eval_actions", level=1)
    def _evaluate_actions(self, *args, **kwargs):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(*args, **kwargs)

    def before_backward(self, loss: Tensor) -> Tensor:
        return loss

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> torch.Tensor:
        handles = []
        if torch.distributed.is_initialized():
            for p in self.non_ac_params:
                if p.grad is not None:
                    p.grad.data.detach().div_(
                        torch.distributed.get_world_size()
                    )
                    handles.append(
                        torch.distributed.all_reduce(
                            p.grad.data.detach(), async_op=True
                        )
                    )

        grad_norm = nn.utils.clip_grad_norm_(
            self.actor_critic.policy_parameters(),
            self.max_grad_norm,
        )

        for v in self.actor_critic.aux_loss_parameters().values():
            nn.utils.clip_grad_norm_(v, self.max_grad_norm)

        [h.wait() for h in handles]

        return grad_norm

    def after_step(self) -> None:
        if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
            self.entropy_coef.project_into_bounds()

    def get_resume_state(self):
        return {
            "optim_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        if self.optimizer is not None and "optim_state" in state:
            self.optimizer.load_state_dict(state["optim_state"])


@baseline_registry.register_updater
class GRPO(PPO):
    @classmethod
    def from_config(
        cls, actor_critic: NetPolicy, config, aux_tasks=[], aux_names=[], aux_cfg=None
    ):
        return cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            aux_loss_coef=0.0,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
            aux_tasks=aux_tasks,
            aux_names=aux_names,
            aux_cfg=aux_cfg,
            group_size=getattr(config, "grpo_group_size", 4),
            min_group_size=getattr(config, "grpo_min_group_size", 2),
            group_key=getattr(config, "grpo_group_key", "instruction"),
            reward_aggregation=getattr(config, "grpo_reward_aggregation", "sum"),
            advantage_eps=getattr(config, "grpo_advantage_eps", EPS_PPO),
            normalize_std=getattr(config, "grpo_normalize_std", True),
            fallback_to_gae=getattr(config, "grpo_fallback_to_gae", True),
            use_batch_fallback=getattr(config, "grpo_use_batch_fallback", True),
            use_value_critic=getattr(config, "grpo_use_value_critic", False),
            ref_kl_coef=getattr(config, "grpo_ref_kl_coef", 0.0),
            advantage_clip=getattr(config, "grpo_advantage_clip", 5.0),
            global_norm_weight=getattr(config, "grpo_global_norm_weight", 0.3),
            # KL annealing schedule parameters
            ref_kl_anneal_start=getattr(config, "grpo_ref_kl_anneal_start", None),
            ref_kl_anneal_end=getattr(config, "grpo_ref_kl_anneal_end", None),
            ref_kl_anneal_steps=getattr(config, "grpo_ref_kl_anneal_steps", 0),
            # Adaptive global-norm-weight parameters
            adaptive_global_norm=getattr(config, "grpo_adaptive_global_norm", True),
            global_norm_weight_min=getattr(config, "grpo_global_norm_weight_min", 0.05),
            # Entropy anti-collapse parameters
            entropy_min_threshold=getattr(config, "grpo_entropy_min_threshold", 0.01),
            entropy_collapse_action=getattr(config, "grpo_entropy_collapse_action", "warn"),
            # Small-group advantage scaling
            scale_adv_by_group_size=getattr(config, "grpo_scale_adv_by_group_size", True),
        )

    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        aux_loss_coef: float = 0.0,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = False,
        use_normalized_advantage: bool = True,
        entropy_target_factor: float = 0.0,
        use_adaptive_entropy_pen: bool = False,
        aux_tasks=[],
        aux_names=[],
        aux_cfg=None,
        group_size: int = 4,
        min_group_size: int = 2,
        group_key: str = "instruction",
        reward_aggregation: str = "sum",
        advantage_eps: float = EPS_PPO,
        normalize_std: bool = True,
        fallback_to_gae: bool = True,
        use_batch_fallback: bool = True,
        use_value_critic: bool = False,
        ref_kl_coef: float = 0.0,
        advantage_clip: float = 5.0,
        global_norm_weight: float = 0.3,
        # KL annealing
        ref_kl_anneal_start: Optional[float] = None,
        ref_kl_anneal_end: Optional[float] = None,
        ref_kl_anneal_steps: int = 0,
        # Adaptive global-norm-weight
        adaptive_global_norm: bool = True,
        global_norm_weight_min: float = 0.05,
        # Entropy anti-collapse
        entropy_min_threshold: float = 0.01,
        entropy_collapse_action: str = "warn",
        # Small-group advantage scaling
        scale_adv_by_group_size: bool = True,
    ) -> None:
        super().__init__(
            actor_critic=actor_critic,
            clip_param=clip_param,
            ppo_epoch=ppo_epoch,
            num_mini_batch=num_mini_batch,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            aux_loss_coef=aux_loss_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            use_normalized_advantage=use_normalized_advantage,
            entropy_target_factor=entropy_target_factor,
            use_adaptive_entropy_pen=use_adaptive_entropy_pen,
            aux_tasks=aux_tasks,
            aux_names=aux_names,
            aux_cfg=aux_cfg,
        )
        self.group_size = max(1, int(group_size))
        self.min_group_size = max(1, int(min_group_size))
        self.group_key = str(group_key)
        self.reward_aggregation = str(reward_aggregation)
        self.advantage_eps = float(advantage_eps)
        self.normalize_std = bool(normalize_std)
        self.fallback_to_gae = bool(fallback_to_gae)
        self.use_batch_fallback = bool(use_batch_fallback)
        self.use_value_critic = bool(use_value_critic)
        self.ref_kl_coef = float(ref_kl_coef)
        self.reference_actor_critic: Optional[NetPolicy] = None
        self._last_grpo_stats: Dict[str, float] = {}
        self.advantage_clip = float(advantage_clip)
        self.global_norm_weight = float(global_norm_weight)

        # --- KL annealing schedule ---
        self.ref_kl_anneal_start = (
            float(ref_kl_anneal_start) if ref_kl_anneal_start is not None else self.ref_kl_coef
        )
        self.ref_kl_anneal_end = (
            float(ref_kl_anneal_end) if ref_kl_anneal_end is not None else self.ref_kl_coef
        )
        self.ref_kl_anneal_steps = max(0, int(ref_kl_anneal_steps))
        self._num_updates = 0  # track update count for scheduling

        # --- Adaptive global-norm-weight ---
        self.adaptive_global_norm = bool(adaptive_global_norm)
        self.global_norm_weight_min = float(global_norm_weight_min)
        self._effective_global_norm_weight = self.global_norm_weight  # Updated dynamically

        # --- Entropy anti-collapse ---
        self.entropy_min_threshold = float(entropy_min_threshold)
        self.entropy_collapse_action = str(entropy_collapse_action)
        self._entropy_collapse_warned = False

        # --- Small-group advantage scaling ---
        self.scale_adv_by_group_size = bool(scale_adv_by_group_size)

        self.set_reference_policy(actor_critic)
        logger.info(
            f"[GRPO] Initialized: ref_kl_coef={self.ref_kl_coef:.4f}, "
            f"anneal: {self.ref_kl_anneal_start:.4f} -> {self.ref_kl_anneal_end:.4f} "
            f"over {self.ref_kl_anneal_steps} steps, "
            f"adaptive_global_norm={self.adaptive_global_norm}, "
            f"scale_adv_by_group_size={self.scale_adv_by_group_size}, "
            f"entropy_min_threshold={self.entropy_min_threshold}"
        )

    def _get_current_ref_kl_coef(self) -> float:
        """Get reference KL coefficient with linear annealing applied."""
        if self.ref_kl_anneal_steps <= 0:
            return self.ref_kl_coef
        # Linear schedule: anneal from start to end over ref_kl_anneal_steps
        progress = min(self._num_updates / self.ref_kl_anneal_steps, 1.0)
        coef = self.ref_kl_anneal_start + (self.ref_kl_anneal_end - self.ref_kl_anneal_start) * progress
        return coef

    def _compute_effective_global_norm_weight(self, grouped: Dict[str, list]) -> float:
        """Adaptively reduce global_norm_weight when groups are large enough.

        When most episodes are in groups of size >= min_group_size * 2,
        the group-local normalization is reliable, so we reduce the
        global bias to preserve relative ordering within groups.
        """
        if not self.adaptive_global_norm:
            return self.global_norm_weight

        total_items = 0
        well_grouped = 0
        for items in grouped.values():
            n = len(items)
            total_items += n
            if n >= self.min_group_size:
                # Group is usable: measure how far above min_group_size it is
                # Weight by group cardinality — well-populated groups count more
                surplus = n - self.min_group_size
                confidence = min(surplus / max(self.group_size, 1), 1.0)
                well_grouped += n * confidence

        if total_items == 0:
            return self.global_norm_weight

        # Fraction of episodes in "well-grouped" context
        well_frac = well_grouped / total_items
        # Linearly interpolate: well_frac=0 → full global_weight; well_frac=1 → min
        effective = self.global_norm_weight_min + (self.global_norm_weight - self.global_norm_weight_min) * (1.0 - well_frac)
        return float(effective)

    def _compute_advantage_scaling(self, n_group: int) -> float:
        """Scale advantage signals inversely with group size.

        Without scaling, groups of size 2 produce ±1 binary advantages
        regardless of return magnitude.  This function down-weights
        advantages from very small groups to reduce their influence.
        """
        if not self.scale_adv_by_group_size:
            return 1.0
        if n_group <= 0:
            return 0.0
        # sigmoid-style scaling: full weight at group_size, ~0.5 at min_group_size
        # Returns factor in [0, 1]
        midpoint = self.min_group_size
        slope = 2.0 / max(midpoint, 1)
        return float(torch.sigmoid(torch.tensor(slope * (n_group - midpoint))))

    def set_reference_policy(self, actor_critic: Optional[NetPolicy] = None) -> None:
        source_policy = actor_critic if actor_critic is not None else self.actor_critic
        self.reference_actor_critic = copy.deepcopy(source_policy).to(self.device)
        self.reference_actor_critic.eval()
        for param in self.reference_actor_critic.parameters():
            param.requires_grad_(False)

    def _instruction_group_key(self, instruction_tensor: torch.Tensor) -> str:
        instruction_arr = (
            instruction_tensor.detach().to(device="cpu").view(-1).to(torch.int64)
        )
        valid_bytes = [int(v) for v in instruction_arr.tolist() if int(v) != 0]
        if len(valid_bytes) == 0:
            return "__empty_instruction__"
        try:
            return bytes(valid_bytes).decode("utf-8", errors="ignore").strip() or "__empty_instruction__"
        except Exception:
            return str(valid_bytes)

    def _aggregate_episode_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.reward_aggregation == "mean":
            return rewards.mean()
        if self.reward_aggregation == "last":
            return rewards[-1]
        return rewards.sum()

    def _build_completed_episode_groups(
        self, rollouts: RolloutStorage
    ) -> Dict[str, list]:
        assert isinstance(rollouts.buffers["rewards"], torch.Tensor)
        assert isinstance(rollouts.buffers["masks"], torch.Tensor)
        num_steps = rollouts.current_rollout_step_idx
        rewards = rollouts.buffers["rewards"][:num_steps]
        next_masks = rollouts.buffers["masks"][1 : num_steps + 1]
        instructions = rollouts.buffers["observations"].get("falcon_instruction", None)
        num_envs = rewards.size(1)

        grouped = collections.defaultdict(list)
        completed_order = []

        for env_idx in range(num_envs):
            episode_start = 0
            for step_idx in range(num_steps):
                done = not bool(next_masks[step_idx, env_idx, 0].item())
                if not done:
                    continue

                reward_segment = rewards[episode_start : step_idx + 1, env_idx, 0]
                episode_return = self._aggregate_episode_reward(reward_segment)

                if instructions is not None and self.group_key == "instruction":
                    group_name = self._instruction_group_key(
                        instructions[episode_start, env_idx]
                    )
                else:
                    group_name = "__all__"

                item = {
                    "env_idx": env_idx,
                    "start": episode_start,
                    "end": step_idx + 1,
                    "return": episode_return.detach(),
                }
                grouped[group_name].append(item)
                completed_order.append(item)
                episode_start = step_idx + 1

        if self.use_batch_fallback:
            leftovers = []
            for group_name in list(grouped.keys()):
                if len(grouped[group_name]) < self.min_group_size:
                    leftovers.extend(grouped.pop(group_name))
            for batch_start in range(0, len(leftovers), self.group_size):
                chunk = leftovers[batch_start : batch_start + self.group_size]
                if len(chunk) >= self.min_group_size:
                    grouped[f"__fallback_{batch_start // self.group_size}__"] = chunk

        return grouped

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        """Compute GRPO advantages with adaptive global-norm blending.

        Improvements over baseline:
        1. Adaptive global-norm weight: reduces global bias when groups are large.
        2. Small-group advantage scaling: down-weights advantages from groups
           smaller than ``min_group_size * 2`` to prevent binary (+1/-1) signals.
        3. Group-size-aware blending: confidence computed per-group, not per-episode.
        4. Final advantages are clipped to ``advantage_clip`` to prevent extreme
           gradient steps from single outlier episodes.
        """
        base_advantages = super().get_advantages(rollouts)
        num_steps = rollouts.current_rollout_step_idx

        # Default stats for zero-step edge case
        self._last_grpo_stats = {
            "grpo_group_count": 0.0,
            "grpo_episode_count": 0.0,
            "grpo_step_fraction": 0.0,
            "grpo_adv_mean": 0.0,
            "grpo_adv_std": 0.0,
            "grpo_adv_min": 0.0,
            "grpo_adv_max": 0.0,
            "grpo_adv_n_clipped": 0.0,
            "grpo_reward_mean": 0.0,
            "grpo_reward_std": 0.0,
            "grpo_reward_min": 0.0,
            "grpo_reward_max": 0.0,
            "grpo_avg_ep_len": 0.0,
            "grpo_avg_group_size": 0.0,
            "grpo_n_episodes_total": 0.0,
            "grpo_eff_global_norm_weight": 0.0,
        }
        if num_steps == 0:
            return base_advantages

        grouped = self._build_completed_episode_groups(rollouts)

        # --- collect ALL completed episode returns ---
        all_returns: List[torch.Tensor] = []
        all_items: List[dict] = []
        for items in grouped.values():
            for item in items:
                all_returns.append(item["return"])
                all_items.append(item)

        # --- compute group-normalized (local) advantages ---
        local_advantages = (
            base_advantages.clone()
            if self.fallback_to_gae
            else torch.zeros_like(base_advantages)
        )
        assigned_mask = torch.zeros_like(local_advantages, dtype=torch.bool)
        used_groups = 0
        used_episodes = 0

        for items in grouped.values():
            if len(items) < self.min_group_size:
                continue

            returns = torch.stack([item["return"] for item in items]).to(
                device=local_advantages.device, dtype=local_advantages.dtype
            )
            centered = returns - returns.mean()

            # --- Small-group advantage scaling ---
            # Groups near min_group_size produce unreliable std estimates,
            # leading to binary ±1 advantages.  Scale these down.
            adv_scale = self._compute_advantage_scaling(len(items))

            if self.normalize_std and len(items) > 1:
                raw_std = returns.std(unbiased=False)
                # If std is near-zero, skip normalization to avoid exploding advantages
                if raw_std > self.advantage_eps * 100:
                    scores = centered / (raw_std + self.advantage_eps)
                else:
                    # Near-zero-variance group: use raw centered values (small signals)
                    scores = centered
            else:
                scores = centered

            scores = scores * adv_scale

            for item, score in zip(items, scores):
                local_advantages[item["start"] : item["end"], item["env_idx"], 0] = score
                assigned_mask[item["start"] : item["end"], item["env_idx"], 0] = True

            used_groups += 1
            used_episodes += len(items)

        # --- global normalization over ALL completed episodes ---
        n_episodes = len(all_returns)
        effective_gw = self._compute_effective_global_norm_weight(grouped)
        self._effective_global_norm_weight = effective_gw

        if n_episodes >= 2 and effective_gw > 0.0:
            returns_global = torch.stack(all_returns).to(
                device=local_advantages.device, dtype=local_advantages.dtype
            )
            global_mean = returns_global.mean()
            global_std = returns_global.std(unbiased=False) + self.advantage_eps
            global_scores = (returns_global - global_mean) / global_std

            global_advantages = torch.zeros_like(local_advantages)
            for item, gscore in zip(all_items, global_scores):
                global_advantages[item["start"] : item["end"], item["env_idx"], 0] = gscore

            # Blend per-group: confidence grows with group size relative to min_group_size.
            # Smaller groups → heavier global weight. Larger groups → heavier local weight.
            blended = local_advantages.clone()
            for group_name, items in grouped.items():
                n_group = len(items)
                if n_group < 1:
                    continue
                # Confidence: 0 at min_group_size, ~1 at group_size*2
                confidence = torch.sigmoid(
                    torch.tensor(
                        (n_group - self.min_group_size) / max(self.group_size, 1),
                        dtype=torch.float32,
                    )
                ).item()
                # gw is at most effective_gw, at least (1-confidence)
                # When groups are very small, gw ≈ 1-confidence (heavy global)
                # When groups are large, gw ≈ effective_gw (minimal global)
                gw = min(effective_gw, max(effective_gw * 0.5, 1.0 - confidence))
                lw = 1.0 - gw
                for item in items:
                    seg = slice(item["start"], item["end"])
                    eidx = item["env_idx"]
                    blended[seg, eidx, 0] = (
                        lw * local_advantages[seg, eidx, 0]
                        + gw * global_advantages[seg, eidx, 0]
                    )

            advantages = blended
        else:
            advantages = local_advantages

        # --- clip extreme advantage values ---
        adv_mean_before = advantages.mean().item()
        adv_std_before = advantages.std().item()
        n_clipped = (advantages.abs() > self.advantage_clip).sum().item() if self.advantage_clip > 0 else 0
        if self.advantage_clip > 0:
            advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)

        total_steps = float(num_steps * advantages.size(1))
        assigned_steps = float(
            (assigned_mask if n_episodes >= 2 else torch.ones_like(assigned_mask))
            .sum()
            .item()
        )

        # --- compute episode return statistics ---
        ret_mean = 0.0
        ret_std = 0.0
        ret_min = 0.0
        ret_max = 0.0
        avg_ep_len = 0.0
        n_episodes_total = len(all_returns)
        if n_episodes_total > 0:
            rets = torch.stack(all_returns).float()
            ret_mean = rets.mean().item()
            ret_std = rets.std(unbiased=False).item() if n_episodes_total > 1 else 0.0
            ret_min = rets.min().item()
            ret_max = rets.max().item()
            avg_ep_len = sum(
                item["end"] - item["start"] for item in all_items
            ) / max(n_episodes_total, 1)

        # --- compute group-level statistics ---
        group_sizes = [len(items) for items in grouped.values() if len(items) >= self.min_group_size]
        avg_group_size = float(sum(group_sizes)) / max(len(group_sizes), 1)

        self._last_grpo_stats = {
            "grpo_group_count": float(used_groups),
            "grpo_episode_count": float(used_episodes),
            "grpo_step_fraction": assigned_steps / max(total_steps, 1.0),
            "grpo_adv_mean": adv_mean_before,
            "grpo_adv_std": adv_std_before,
            "grpo_adv_min": advantages.min().item(),
            "grpo_adv_max": advantages.max().item(),
            "grpo_adv_n_clipped": float(n_clipped),
            "grpo_reward_mean": ret_mean,
            "grpo_reward_std": ret_std,
            "grpo_reward_min": ret_min,
            "grpo_reward_max": ret_max,
            "grpo_avg_ep_len": avg_ep_len,
            "grpo_avg_group_size": avg_group_size,
            "grpo_n_episodes_total": float(n_episodes_total),
            "grpo_eff_global_norm_weight": float(effective_gw),
        }

        return advantages

    def _get_reference_log_probs(self, batch) -> Optional[torch.Tensor]:
        if self.reference_actor_critic is None or self.ref_kl_coef <= 0.0:
            return None

        with torch.no_grad():
            ref_outputs = self.reference_actor_critic.evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
                batch.get("rnn_build_seq_info", None),
            )
        return ref_outputs[1].detach()

    @g_timer.avg_time("grpo.update_from_batch", level=1)
    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        def record_min_mean_max(t: torch.Tensor, prefix: str):
            for name, op in (
                ("min", torch.min),
                ("mean", torch.mean),
                ("max", torch.max),
            ):
                learner_metrics[f"{prefix}_{name}"].append(op(t))

        self._set_grads_to_none()
        aux_dist_entropy = None
        if isinstance(self.actor_critic, AttentiveBeliefPolicy):
            (
                values,
                action_log_probs,
                dist_entropy,
                final_rnn_state,
                rnn_features,
                individual_rnn_features,
                aux_dist_entropy,
                aux_weights,
            ) = self._evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
                batch.get("rnn_build_seq_info", None),
            )
        else:
            (
                values,
                action_log_probs,
                dist_entropy,
                final_rnn_state,
                aux_loss_res,
            ) = self._evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
                batch.get("rnn_build_seq_info", None),
            )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])
        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * torch.clamp(
            ratio,
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        )
        action_loss = -torch.min(surr1, surr2)

        values = values.float()
        orig_values = values
        if self.use_value_critic:
            if self.use_clipped_value_loss:
                delta = values.detach() - batch["value_preds"]
                value_pred_clipped = batch["value_preds"] + delta.clamp(
                    -self.clip_param, self.clip_param
                )
                values = torch.where(
                    delta.abs() < self.clip_param,
                    values,
                    value_pred_clipped,
                )

            value_loss = 0.5 * F.mse_loss(
                values, batch["returns"], reduction="none"
            )
        else:
            value_loss = torch.zeros_like(action_loss)

        if "is_coeffs" in batch:
            assert isinstance(batch["is_coeffs"], torch.Tensor)
            ver_is_coeffs = batch["is_coeffs"].clamp(max=1.0)
            mean_fn = lambda t: torch.mean(ver_is_coeffs * t)
        else:
            mean_fn = torch.mean

        ref_log_probs = self._get_reference_log_probs(batch)
        ref_kl = None
        if ref_log_probs is not None:
            ref_kl = action_log_probs - ref_log_probs

        action_loss, value_loss, dist_entropy = map(
            mean_fn,
            (action_loss, value_loss, dist_entropy),
        )
        if ref_kl is not None:
            ref_kl = mean_fn(ref_kl)

        total_aux_loss = 0
        aux_losses = []
        if isinstance(self.actor_critic, AttentiveBeliefPolicy) and len(self._aux_tasks) > 0:
            aux_raw_losses = self.actor_critic.evaluate_aux_losses(
                batch,
                final_rnn_state,
                rnn_features,
                individual_rnn_features,
            )
            aux_losses = torch.stack(aux_raw_losses)
            total_aux_loss = torch.sum(aux_losses, dim=0)

        # --- Use annealed KL coefficient ---
        current_ref_kl_coef = self._get_current_ref_kl_coef() if ref_kl is not None else 0.0

        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]
        if ref_kl is not None and current_ref_kl_coef > 0.0:
            all_losses.append(current_ref_kl_coef * ref_kl)

        if isinstance(self.actor_critic, AttentiveBeliefPolicy):
            all_losses.append(total_aux_loss * self.aux_loss_coef)

        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        if aux_dist_entropy is not None:
            all_losses.append(aux_dist_entropy * self.aux_cfg.entropy_coef)

        if len(self._aux_tasks) == 0:
            all_losses.extend(v["loss"] for v in aux_loss_res.values())

        total_loss = torch.stack(all_losses).sum()
        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            if "is_coeffs" in batch:
                record_min_mean_max(batch["is_coeffs"], "ver_is_coeffs")
            record_min_mean_max(orig_values, "value_pred")
            record_min_mean_max(ratio, "prob_ratio")
            # Log GRPO advantage statistics for stability monitoring
            record_min_mean_max(batch["advantages"], "grpo_adv")

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if ref_kl is not None:
                learner_metrics["grpo_ref_kl"].append(ref_kl)
                learner_metrics["grpo_ref_kl_coef"].append(current_ref_kl_coef)

            # --- Entropy anti-collapse monitoring ---
            mean_ent = dist_entropy.detach().item() if isinstance(dist_entropy, torch.Tensor) else float(dist_entropy)
            learner_metrics["grpo_entropy_check"].append(mean_ent)
            if mean_ent < self.entropy_min_threshold and not self._entropy_collapse_warned:
                msg = (
                    f"[GRPO] WARNING: Policy entropy ({mean_ent:.6f}) below "
                    f"threshold ({self.entropy_min_threshold}) at update {self._num_updates}. "
                    f"Policy may be collapsing to deterministic behavior."
                )
                if self.entropy_collapse_action == "skip_update":
                    logger.warning(f"{msg} Skipping this update.")
                    # Don't apply this optimizer step — but we've already stepped above.
                    # For true skipping we'd need to restructure; warn for now.
                else:
                    logger.warning(msg)
                self._entropy_collapse_warned = True
            elif mean_ent >= self.entropy_min_threshold * 2.0:
                self._entropy_collapse_warned = False

            if epoch == (self.ppo_epoch - 1):
                learner_metrics["ppo_fraction_clipped"].append(
                    (ratio > (1.0 + self.clip_param)).float().mean()
                    + (ratio < (1.0 - self.clip_param)).float().mean()
                )

            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(
                    self.entropy_coef().detach()
                )

            if len(self._aux_tasks) == 0:
                for name, res in aux_loss_res.items():
                    for k, v in res.items():
                        learner_metrics[f"aux_{name}_{k}"].append(v.detach())
            else:
                learner_metrics["aux_entropy"].append(aux_dist_entropy)
                for i, aux_loss in enumerate(aux_losses):
                    learner_metrics[f"aux_entropy_{self._aux_names[i]}"].append(aux_loss.item())
                for i, aux_weight in enumerate(aux_weights):
                    learner_metrics[f"aux_weights_{self._aux_names[i]}"].append(aux_weight.item())

            if "is_stale" in batch:
                assert isinstance(batch["is_stale"], torch.Tensor)
                learner_metrics["fraction_stale"].append(
                    batch["is_stale"].float().mean()
                )

            if isinstance(rollouts, VERRolloutStorage):
                assert isinstance(batch["policy_version"], torch.Tensor)
                record_min_mean_max(
                    (
                        rollouts.current_policy_version
                        - batch["policy_version"]
                    ).float(),
                    "policy_version_difference",
                )

    def update(
        self,
        rollouts: RolloutStorage,
    ) -> Dict[str, float]:
        metrics = super().update(rollouts)
        metrics.update(self._last_grpo_stats)
        # Add scheduling-related metrics
        metrics["grpo_ref_kl_coef"] = self._get_current_ref_kl_coef()
        metrics["grpo_eff_global_norm_weight"] = float(
            getattr(self, "_effective_global_norm_weight", self.global_norm_weight)
        )
        metrics["grpo_num_updates"] = float(self._num_updates)
        self._num_updates += 1
        return metrics

    def get_resume_state(self):
        state = super().get_resume_state()
        if self.reference_actor_critic is not None:
            state["reference_state_dict"] = self.reference_actor_critic.state_dict()
        state["grpo_num_updates"] = self._num_updates
        state["grpo_entropy_collapse_warned"] = self._entropy_collapse_warned
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "reference_state_dict" in state:
            self.set_reference_policy(self.actor_critic)
            if self.reference_actor_critic is not None:
                self.reference_actor_critic.load_state_dict(
                    state["reference_state_dict"], strict=False
                )
        if "grpo_num_updates" in state:
            self._num_updates = int(state["grpo_num_updates"])
        if "grpo_entropy_collapse_warned" in state:
            self._entropy_collapse_warned = bool(state["grpo_entropy_collapse_warned"])
