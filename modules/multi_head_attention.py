import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device, d_model, n_heads, dropout


class MultiHeadAttention(nn.Module):
    """
    原论文多头自注意力层：
    核心逻辑：先行投影 -> 拆分为多头 -> 计算注意力 -> 合并多头 -> 输出投影
    支持三种注意力模式：
    - 自注意力（编码器）：Q = K = V
    - 掩码子注意力（解码器）：Q = K = V + 因果掩码
    - 编码器-解码器注意力（解码器）：Q= 解码器输出，K = V = 编码器输出
    """

    def __init__(self):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model}必须能被n_heads={n_heads}整除"
        self.d_k = d_model // n_heads  # 每个注意力头的维度
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(device)  # 缩放因子，防止内积过大

        # 线性投影层，Q/K/V各一个，输出层一个（原论文无偏置，保留偏置以提升训练稳定性）
        self.w_q = nn.Linear(d_model, d_model, device=device)
        self.w_k = nn.Linear(d_model, d_model, device=device)
        self.w_v = nn.Linear(d_model, d_model, device=device)
        self.w_o = nn.Linear(d_model, d_model, device=device)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将特征拆分为多头，提升自注意力细粒度
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, n_head, seq_len, d_k)
        """
        batch_size = x.size(0)
        return x.view(batch_size, -1, n_heads, self.d_k).permute(0, 2, 1, 3)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头特征合并
        @:param x: (batch_size, n_heads, seq_len, d_k)
        @:return: (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, d_model)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        :param q: 查询张量 (batch_size, seq_len_q, d_model)
        :param k: 键张量 (batch_size, seq_len_k, d_model)
        :param v: 值张量 (batch_size, seq_len_v, d_model) （seq_len_k = seq_len_v）
        :param mask: 掩码张量 (1, 1, seq_len_q, seq_len_k) 或 (batch_size, 1, seq_len_q, seq_len_k)
        :return: 注意力输出 (batch_size, seq_len_q, d_model)，注意力权重 (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        # 1. 线性投影得到 Q/K/V
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # 2. 拆分为多头
        q_head = self.split_heads(q_proj)
        k_head = self.split_heads(k_proj)
        v_head = self.split_heads(v_proj)

        # 3. 计算注意力分数：Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(q_head, k_head.permute(0, 1, 3, 2)) / self.scale

        # 4. 应用掩码，不可见位置设为-∞，softmax后权重为0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == float('-inf'), float('-inf'))

        # 5. Softmax得到注意力权重，添加Dropout防止过拟合
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)

        # 6. 注意力加权和 + 合并多头 + 输出投影
        attn_out = torch.matmul(attn_weights, v_head)
        attn_out = self.combine_heads(attn_out)
        output = self.w_o(attn_out)

        return output, attn_weights
