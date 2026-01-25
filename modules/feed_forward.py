import torch
import torch.nn as nn
from utils import device, d_model, d_ff, dropout


class FeedForward(nn.Module):
    """
    原论文前馈网络层：d_model -> d_ff -> d_model
    核心逻辑：线性层 -> ReLU -> Dropout -> 线性层
    所有层共享一个前馈网络（原论文设计）
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, device=device) # 升维
        self.fc2 = nn.Linear(d_ff, d_model, device=device) # 降维
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        return self.fc2(self.dropout(self.relu(self.fc1(x))))