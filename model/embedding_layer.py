import math
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        # Embedding层，将词汇表的大小映射为d_model维的向量
        self.lut = nn.Embedding(vocab, d_model)
        # 存储模型的维度 d_model
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要除以 sqrt(d_model)）
        # 使得嵌入向量的量级与后续残差/位置编码在同意尺度，稳定训练、加速收敛
        return self.lut(x) * math.sqrt(self.d_model)
