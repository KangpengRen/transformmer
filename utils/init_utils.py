import torch.nn as nn
from .device import device


def init_weights(module: nn.Module):
    """
    原论文标准权重初始化规则：
    — 线性层：Xavier均匀初始化权重，偏置置0
    — 嵌入层：Xavier均匀初始化权重
    :param module: 模型/组件实例 (通过model.apply(init_weights)调用)
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
    # 层归一化保持默认初始化即可
