"""
辅助损失管理模块 - 参考VLN-CE的实现
用于在DAgger训练中管理和计算辅助损失
"""

import torch


class _AuxLosses:
    """辅助损失管理器，采用单例模式"""
    
    def __init__(self) -> None:
        self._losses = {}
        self._loss_alphas = {}
        self._is_active = False

    def clear(self):
        """清空所有已注册的损失"""
        self._losses.clear()
        self._loss_alphas.clear()

    def register_loss(self, name, loss, alpha=1.0):
        """注册一个辅助损失
        
        Args:
            name: 损失名称
            loss: 损失张量（通常是每个样本的损失，形状为[batch_size]或[T, N]）
            alpha: 损失的权重系数
        """
        assert self.is_active(), "AuxLosses must be activated before registering losses"
        assert name not in self._losses, f"Loss {name} already registered"

        self._losses[name] = loss
        self._loss_alphas[name] = alpha

    def get_loss(self, name):
        """获取指定名称的损失"""
        return self._losses[name]

    def has_losses(self):
        """检查是否有已注册的损失"""
        return len(self._losses) > 0

    def reduce(self, mask):
        """计算所有辅助损失的加权和
        
        Args:
            mask: 布尔掩码，形状应与损失张量匹配，用于选择有效样本
        
        Returns:
            加权后的总辅助损失
        """
        assert self.is_active(), "AuxLosses must be activated before reducing losses"
        if len(self._losses) == 0:
            # 如果没有注册任何损失，返回0
            return torch.tensor(0.0, device=mask.device if isinstance(mask, torch.Tensor) else "cpu")
        
        total = 0.0

        for k in self._losses.keys():
            k_loss = torch.masked_select(self._losses[k], mask).mean()
            total = total + self._loss_alphas[k] * k_loss

        return total

    def is_active(self):
        """检查辅助损失是否激活"""
        return self._is_active

    def activate(self) -> None:
        """激活辅助损失管理器"""
        self._is_active = True

    def deactivate(self):
        """停用辅助损失管理器"""
        self._is_active = False


# 全局单例实例
AuxLosses = _AuxLosses()

