import torch
import torch.nn as nn
import math

from utils.device import dropout, d_model, device


class PositionalEncoding(nn.Module):
    """
    原论文正余弦绝对位置编码：为词嵌入添加位置，支持任意长度序列
    预计算位置编码矩阵，注册为非参数张量，不参与梯度更新
    """

    def __init__(self, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 预计算位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model, device=device)
        # 位置索引：(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        # 分母项：10000^(2i/d_model)，通过指数+对数简化计算
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model)
        )
        # 偶数位用正弦，奇数位用预先
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度，(1, max_len, d_model)，适配广播维度
        pe = pe.unsqueeze(0)
        # 注册为非参数张量，不参与训练，仅随模型保存/加载
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 词嵌入张量 (batch_size, seq_len, d_model)
        :return: 叠加位置编码后的张量 (batch_size, seq_len, d_model)
        """
        # 截取与输入序列长度匹配的位置编码，避免维度不匹配
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
