"""Core module for TRL compatibility."""
import torch
import random
import numpy as np
from functools import wraps
from typing import Any, Dict, Optional

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    return torch.randn(shape, generator=generator, device=device, dtype=dtype, layout=layout)

class PPODecorators:
    @staticmethod
    def empty_device_cache():
        """Decorator that clears CUDA cache before and after function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = func(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return result
            return wrapper
        return decorator

# Wandb padding constant
WANDB_PADDING = -1

import torch as _torch
def clip_by_value(x, min_val, max_val):
    return _torch.clamp(x, min_val, max_val)

import torch
from typing import Any, Dict

def convert_to_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)

def flatten(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(flatten(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {flatten(v)}" for k, v in value.items())
    return str(value)

import torch.nn.functional as F
def entropy_from_logits(logits):
    pd = F.softmax(logits, dim=-1)
    entropy = -(pd * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    return entropy

def logprobs_from_logits(logits, labels, gather=True):
    """Compute log probabilities from logits. Standard TRL function."""
    logp = F.log_softmax(logits, dim=-1)
    if gather:
        return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logp

def masked_mean(values, mask, axis=None):
    """Compute mean of values with mask."""
    values = torch.as_tensor(values, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    return (values * mask).sum() / mask.sum()

def masked_var(values, mask, unbiased=True):
    """Compute variance of values with mask."""
    values = torch.as_tensor(values, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum > 1:
            variance = variance * mask_sum / (mask_sum - 1)
    return variance

def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with mask."""
    values = torch.as_tensor(values, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    mean = masked_mean(values, mask) if shift_mean else 0.0
    var = masked_var(values, mask)
    whitened = (values - mean) * mask / (torch.sqrt(var) + 1e-8)
    return whitened

def stack_dicts(stats_dicts):
    """Stack list of dicts into a single dict."""
    results = {}
    if not stats_dicts:
        return results
    for k in stats_dicts[0].keys():
        values = [d[k] for d in stats_dicts if k in d]
        if len(values) > 0 and isinstance(values[0], torch.Tensor):
            results[k] = torch.stack(values)
        else:
            results[k] = values
    return results

def stats_to_np(stats):
    """Convert stats dict of tensors to numpy."""
    result = {}
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            result[k] = v
        else:
            result[k] = np.asarray(v)
    return result
