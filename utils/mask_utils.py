import torch
from .device import device


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    创建解码器因果掩码（下三角掩码），防止解码器看未来token
    :param seq_len: 目标序列长度
    :return: 掩码张量 (1, 1, seq_len, seq_len) - 适配多头注意力的维度广播
    """
    # 下三角矩阵：1表示可见，0表示不可见
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # 不可见设置为-∞，softmax后权重为0；可见位置设置为0，不影响softmax
    return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
