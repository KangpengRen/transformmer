import torch
import torch.nn as nn
from utils import device
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .residual_norm import ResidualNorm


class TransformerEncoderLayer(nn.Module):
    """
    单编码器层：：自注意力层 + 前馈网络层，各一个残差连接
    编码器层输入维度完全一致，可无限堆叠
    所有编码器层参数独立（原论文设计）
    """

    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()  # 自注意力层
        self.ffn = FeedForward()  # 前馈网络层
        self.res_norm1 = ResidualNorm()  # 自注意的残差连接
        self.res_norm2 = ResidualNorm()  # 前馈网络的残差连接

    def forward(
            self,
            x: torch.Tensor,
            src_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: 编码器层输入：(batch_size, src_seq_len, d_model)
        :param src_mask: 源序列padding掩码
        :return: 编码器层输出 (batch_size, src_seq_len, d_model)，自注意力权重
        """
        # 1. 自注意力 + 残差连接
        attn_out, attn_weights = self.self_attn(x, x, x, src_mask)
        x = self.res_norm1(x, attn_out)

        # 2. 前馈网络 + 残差连接
        ffn_out = self.ffn(x)
        x = self.res_norm2(x, ffn_out)

        return x, attn_weights