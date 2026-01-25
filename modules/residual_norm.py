import torch
import torch.nn as nn
from utils import device, d_model, dropout


class ResidualNorm(nn.Module):
    """
    Pre_LN版本残差设置（相对原文Post-LN训练更稳定）
    - Post-LN：norm(x + dropout(sublayer(x)))
    - Pre-LN：x + dropout(sublayer(norm(x)))
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)  # 层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 原始输入张量 (batch_size, seq_len, d_model)
        :param sublayer_x: 子组件（注意力/前馈）的输出张量 (batch_size, seq_len, d_model)
        :return: 残差连接后的张量 (batch_size, seq_len, d_model)
        """
        return x + self.dropout(self.norm(sublayer_x))
