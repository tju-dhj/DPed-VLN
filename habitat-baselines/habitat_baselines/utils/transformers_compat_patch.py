"""
Transformers兼容性补丁
为新版transformers添加缺失的函数
"""

import torch


def apply_chunking_to_forward(forward, chunk_size, chunk_dim, tensor):
    """apply_chunking_to_forward兼容函数"""
    return forward(tensor)


def find_pruneable_heads_and_indices(heads, num_attention_heads, attention_head_size, pruned_heads):
    """
    找到可剪枝的注意力头及其索引
    
    来自旧版transformers的find_pruneable_heads_and_indices函数
    """
    # 转换pruned_heads为集合
    pruned_heads = set(pruned_heads) if pruned_heads else set()
    
    # 过滤掉pruned的头
    indices = []
    new_heads = []
    for i, head in enumerate(heads):
        if head not in pruned_heads:
            new_heads.append(head)
            indices.append(i)
    
    return new_heads, indices


def prune_linear_layer(layer, index, dim=0):
    """
    剪枝线性层
    """
    if dim == 0:
        # 返回新的层，只保留指定的索引
        new_weight = layer.weight.index_select(dim, index)
        if hasattr(layer, 'bias') and layer.bias is not None:
            new_bias = layer.bias.index_select(dim, index)
            return type(layer)(new_weight, new_bias)
        return type(layer)(new_weight)
    else:
        new_weight = layer.weight.index_select(dim, index)
        if hasattr(layer, 'bias') and layer.bias is not None:
            return type(layer)(new_weight, layer.bias)
        return type(layer)(new_weight)


# 应用补丁到 transformers.modeling_utils
try:
    import transformers.modeling_utils as modeling_utils
    if not hasattr(modeling_utils, 'apply_chunking_to_forward'):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

    # find_pruneable_heads_and_indices已移到transformers.utils
    try:
        import transformers.utils as transformers_utils
        if not hasattr(modeling_utils, 'find_pruneable_heads_and_indices'):
            modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

        if not hasattr(modeling_utils, 'prune_linear_layer'):
            modeling_utils.prune_linear_layer = prune_linear_layer
    except ImportError:
        pass

except ImportError as e:
    pass

