import torch
import torch.nn as nn
from utils import device
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward
from .residual_norm import ResidualNorm


class TransformerDecoderLayer(nn.Module):
    """
    单解码器层：掩码自注意力 + 编码器-解码器注意力 + 前馈网络，各接一个残差连接
    解码器层输入输出维度完全一致，可无限堆叠
    所有解码器层参数独立（原论文设计）
    """

    def __init__(self):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention()  # 掩码自注意力（防止看到未来）
        self.enc_dec_attn = MultiHeadAttention()  # 编码器-解码器注意力（跨模态注意力）
        self.ffn = FeedForward()  # 前馈网络层
        self.res_norm1 = ResidualNorm()  # 掩码自注意力的残差连接
        self.res_norm2 = ResidualNorm()  # 编解码器的残差连接
        self.res_norm3 = ResidualNorm()  # 前馈网络的残差连接

    def forward(
            self,
            x: torch.Tensor,
            enc_out: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            src_tgt_mask: torch.Tensor = None
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: 解码器层输入 (batch_size, tgt_seq_len, d_model)
        :param enc_out: 编码器最终输出 (batch_size, src_seq_len, d_model)
        :param tgt_mask: 解码器因果掩码 + padding掩码
        :param src_tgt_mask: 编码器-解码器的padding掩码
        :return: 解码器层输出，焱玛琪自注意力权重，编解码器注意力权重
        """
        # 1. 掩码自注意力 + 残差连接
        masked_attn_out, masked_attn_w = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.res_norm1(x, masked_attn_out)
        # 2. 编码器-解码器注意力 + 残差连接 （Q=解码器，K/V=编码器）
        enc_dec_attn_out, enc_dec_attn_w = self.enc_dec_attn(x, enc_out, enc_out, src_tgt_mask)
        x = self.res_norm2(x, enc_dec_attn_out)
        # 3. 前馈网络 + 残差连接
        ffn_out = self.ffn(x)
        x = self.res_norm3(x, ffn_out)
        return x, masked_attn_w, enc_dec_attn_w