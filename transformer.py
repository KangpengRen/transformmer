import torch
import torch.nn as nn
from utils import (
    device, d_model, n_layers, init_weights,
    PositionalEncoding, create_causal_mask
)
from modules import TransformerEncoderLayer, TransformerDecoderLayer


class Transformer(nn.Module):
    """
    完整Transformer模型
    核心组成：词嵌入 -> 位置编码 -> 编码器堆叠 -> 解码器堆叠 -> 输出层
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int):
        """
        :param src_vocab_size: 源语言词汇表大小
        :param tgt_vocab_size: 目标语言词汇表大小
        """
        super().__init__()
        # 1. 词嵌入层，将token Id映射为d_model维向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model, device=device)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, device=device)
        # 2. 位置编码层，为词嵌入添加位置信息
        self.pos_enc = PositionalEncoding()
        # 3. 编码器堆叠：n_layers层独立的编码器
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(n_layers)])
        # 4. 解码器堆叠：n_layers层独立的解码器
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer() for _ in range(n_layers)])
        # 5. 输出层，将d_model维向量映射到词汇表，得到logits
        self.fc_out = nn.Linear(d_model, tgt_vocab_size, device=device)
        # 6. 最终层归一化，提升训练稳定性
        self.norm = nn.LayerNorm(d_model, device=device)
        # 7. 应用源论文标准权重初始化
        self.apply(init_weights)

    def encode(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        编码器整体前向传播
        :param src: 源序列token Id (batch_size, src_seq_len)
        :param src_mask: 源序列padding掩码
        :return: 编码器最终输出 (batch_size, src_seq_len, d_model)，各层注意力权重列表
        """
        attn_weights_list = []
        # 词嵌入 + 放缩（原论文关键：emb * sqrt(d_model)，防止嵌入值过小）
        x = self.src_emb(src) * torch.sqrt(torch.FloatTensor([d_model])).to(device)
        # 叠加位置编码
        x = self.pos_enc(x)
        # 堆叠编码器层
        for layer in self.encoder_layers:
            x, attn_w = layer(x, src_mask)
            attn_weights_list.append(attn_w)
        # 最终层归一化
        return self.norm(x), attn_weights_list

    def decode(
            self,
            tgt: torch.Tensor,
            enc_out: torch.Tensor,
            tgt_mask: torch.Tensor = None,
            src_tgt_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        解码器整体前向传播
        :param tgt: 目标序列token Id (batch_size, tgt_seq_len)
        :param enc_out: 编码器最终输出 (batch_size, src_seq_len, d_model)
        :param tgt_mask: 解码器因果掩码 + padding掩码
        :param src_tgt_mask: 编码器-解码器padding掩码
        :return: 解码器最终输出，各层掩码注意力权重，各层编解码器注意力权重
        """
        masked_attn_list, enc_dec_attn_list = [], []
        x = self.tgt_emb(tgt) * torch.sqrt(torch.FloatTensor([d_model]).to(device))
        x = self.pos_enc(x)
        # 堆叠解码器
        for layer in self.decoder_layers:
            x, masked_w, enc_dec_w = layer(x, enc_out, tgt_mask, src_tgt_mask)
            masked_attn_list.append(masked_w)
            enc_dec_attn_list.append(enc_dec_w)
        # 最终层归一化
        return self.norm(x), masked_attn_list, enc_dec_attn_list

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: torch.Tensor = None,
            tgt_mask: torch.Tensor = None,
            src_tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Transformer整体前向传播：编码 -> 解码 -> 输出层
        :return: 目标词汇表logits (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 1. 编码器前向
        enc_out, _ = self.encode(src, src_mask)
        # 2. 解码器前向
        dec_out, _, _ = self.decode(tgt, enc_out, tgt_mask, src_tgt_mask)
        # 3. 输出层映射到目标词汇表
        logits = self.fc_out(dec_out)
        return logits

