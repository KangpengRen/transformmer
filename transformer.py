import torch
import torch.nn as nn
import math


# 定义自注意力
class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 对10%的神经元随机失活，防止过拟合
        self.softmax = nn.Softmax(dim=-1)  # 将得分转化成概率分布，对最后一个维度进行softmax

    def forward(self, Q, K, V, mask=None):
        """
        X: batch, seq_len, d_model
            batch：一次送到模型的句子个数
            seq_len：一个句子中的token数量
            d_model：embedding中向量的维度
        :param self:
        :param Q: query向量 维度：batch, heads, seq_len_q, d_k
        :param K: key向量 维度：batch, heads, seq_len_k, d_k
        :param V: value向量 维度：batch, heads, seq_len_v, d_v
        :param mask: 掩码，哪些位置需要关注，哪些位置需要忽略
        :return: 返回输出和注意力权重
        """
        d_k = Q.size(-1)  # q的最后一个维度是每个query向量的维度，代表我们对每个query进行缩放
        # batch, heads, seq_len_q, d_k      batch, heads, d_k, seq_len_k
        # batch, heads, seq_len_q, seq_len_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # 进行缩放，让模型训练的梯度更加稳定

        # 如果提供了mask，通过mask==0找到需要屏蔽的位置，masked_fill函数会将该位置的值修改成负无穷（-inf)，softmax后会趋近零，即被忽略
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # batch, heads, seq_len_q, seq_len_k，对最后一个维度进行softmax，得到注意力权重矩阵，对每个query的key权重之和为1
        attn = self.softmax(scores)
        attn = self.dropout(attn)  # 对注意力权重进行dropout，防止过拟合

        # batch, heads, seq_len_q, seq_len_k        batch, heads, seq_len_v, d_v
        # batch, heads, seq_len_q, d_v
        out = torch.matmul(attn, V)  # V的加权和即为输出

        return out, attn


# 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # d_model需要被n_heads整除，结果为64
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads  # 每个头的维度
        self.n_heads = n_heads

        # 将输入映射到Q、K、V三个向量，通过线性映射让模型具有学习能力
        self.W_q = nn.Linear(d_model, d_model)  # query线性映射，维度不需要改变，方便后续的多头拆分
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)  # 多头拼接后再映射会原始的d_model，让模型融合不同头的信息

        self.attention = SelfAttention(dropout)  # 使用定义好的自注意力
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.norm = nn.LayerNorm(d_model)  # 用于残差后的归一化

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)  # 获取batch大小

        # q的维度：batch, seq_len, d_model -> batch, -1(seq_len), n_heads, d_k -> batch, n_heads, seq_len, d_k
        # 让每个注意力头处理每个序列，方便后续计算注意力权重
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力 attn为注意力权重，out为加权后的值
        out, attn = self.attention(Q, K, V, mask)
        # batch, heads, seq_len_q, d_v -> batch, seq_len_q, heads, d_v -> batch, seq_len, d_model
        # contiguous()目的：让tensor在内存中连续存储，避免view()时产生报错
        # 多头拼接
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        out = self.fc(out)  # 让输入和输出一致，方便残差连接
        out = self.dropout(out)  # 在训练阶段随机丢弃10%神经元，防止过拟合

        # 返回输出和注意力权重
        return self.norm(out + q), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 输入维度：d_model=512，输出维度：d_ff=2048，让模型学习到更丰富的特征
        self.fc2 = nn.Linear(d_ff, d_model)  # 输入维度：d_ff=1024，输出维度：f_model=512，让模型便于进行后续残差连接
        self.dropout = nn.Dropout(dropout)  # 随机丢弃，防止过拟合
        self.norm = nn.LayerNorm(d_model)  # 对最后一个维度进行归一化

    def forward(self, x):
        # x: batch, seq_len, d_model
        out = self.fc2(self.dropout(torch.relu(self.fc1(x))))  # 第一个线性层 -> relu -> dropout -> 第二个线性层
        return self.norm(out + x)  # 残差连接，再进行归一化（让模型更稳定，并且加快收敛）


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 多头注意力机制
        self.ffn = FeedForward(d_model, d_ff, dropout)  # 对每个位置的向量独立进行非线性变换，提升模型表达能力

    def forward(self, src, src_mask=None):
        # src 输入序列张量，形状：batch, seq_len, d_model
        # src_mask 屏蔽padding的位置，避免模型关注无效token
        out, _ = self.self_attn(src, src, src, src_mask)  # 输入为原序列，实现序列内部的信息交互，每个token都可以看到其他token，从而可以学习到上下文依赖
        # 经过前馈神经网络，每个位置的token都会单独通过两层映射和激活函数，提升模型表达能力
        out = self.ffn(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Mask Multi-Head Attention
        # 输入tgt，在翻译任务中已经生成的前几个单词，计算序列内部的注意力，通过mask遮挡未来的token，避免信息泄露
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 交叉注意力，和encoder做交互，输入 Q=当前解码器输出，K=V来自编码器的memory（原序列上下文信息）
        # 为了将目标序列和原序列对齐
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 为了提升模型的表达能力
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        :param tgt: 目标序列
        :param memory: 编码器输出（原序列的表示）
        :param tgt_mask: 屏蔽未来的token
        :param memory_mask: PAD做掩码
        :return:
        """
        # 目标序列的自注意力，未来位置被mask
        out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        # 将目标序列和原序列进行交互，Q解码器当前的输出out，K=V=memory（编码器的输出）
        out, _ = self.cross_attn(out, memory, memory, memory_mask)
        out = self.ffn(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 初始化位置编码矩阵，形状为max_len, d_model
        pe = torch.zeros(max_len, d_model)  # d_model：每个词向量的维度，max_len：句子的最大长度

        # 定义记录每个位置的索引，0 ~ max_len-1     [max_len, 1]，方便后续与缩放因子相乘
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 每个维度的缩放因子，torch.arange(0, d_model, 2)生成偶数维度的索引，对应公式中的2i
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 每个token的位置索引pos * 每个维度的缩放因子，再套上sin得到偶数维数的位置编码值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加batch维度，1, max_len, d_model   方便后续与embedding进行相加
        self.register_buffer('pe', pe)  # 注册为buffer，把位置编码pe存在模型中，不参与训练，但是随着模型的保存/加载

    def forward(self, x):
        seq_len = x.size(1)  # x：输入的embedding形状batch, seq_len, d_model
        return x + self.pe[:, :seq_len, :]  # 每个token的embedding加上对应位置的编码


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        # 词嵌入层  vocab_size：词表大小，包含了token总数
        # 将输入的token Id（对原始文本进行分词，不同词对应不同Id）转换成连续向量，维度为d_module
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码，假如序列中token的位置信息
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 构建编码器的堆叠结构，堆叠num_layers个encoder
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        # 将输入token Id转化成embedding向量，输出shape batch，seq_len，d_model
        out = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        out = self.pos_encoding(out)  # 经过位置编码
        for layer in self.layers:
            out = layer(out, src_mask)

        return out  # 返回编码后的输出 batch, seq_len, d_model


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        # 将目标序列的token Id转化为向量，维度为d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)  # 经过位置编码
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        # 输出投影层，将decoder的输出映射回原词汇表的大小，从而得到每个token的预测分布
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        # 添加位置编码信息
        out = self.pos_encoding(out)
        # 逐层经过decoder_layer
        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask)

        # 将解码器最后一层输出的隐藏向量映射回原词汇表的维度，从而得到token的预测向量
        return self.fc_out(out)


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,  # 原语言词表大小
                 tgt_vocab,  # 目标语言词表大小
                 d_model=512,  # embedding向量大小
                 n_heads=8,  # 多头注意力头数
                 num_encoder_layers=6,  # 编码器层数
                 num_decoder_layers=6,  # 解码器层数
                 d_ff=2048,  # ffn隐藏层维度
                 dropout=0.1,
                 max_len=5000):
        super().__init__()

        # 编码器：源语言token编码为上下文表示
        self.encoder = Encoder(
            src_vocab, d_model, n_heads, num_encoder_layers, d_ff, dropout, max_len
        )

        # 解码器：根据编码器的输出和目标语言输入生成预测
        self.decoder = Decoder(
            tgt_vocab, d_model, n_heads, num_encoder_layers, d_ff, dropout, max_len
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)  # 编码器前向传播
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)  # 解码器前向传播

        return out


def generate_mask(size):
    # 生成上三角，不包含对角线
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask == 0  # True为可见

# ------------------ 验证 ------------------
src_vocab = 10000  # 源语言词表大小
tgt_vocab = 10000
# 初始化模型
model = Transformer(src_vocab, tgt_vocab)
src = torch.randint(0, src_vocab, (32, 10))  # 原序列 batch=32，src_len=10，元素为token Id
tgt = torch.randint(0, tgt_vocab, (32, 20))

tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)

out = model(src, tgt, tgt_mask=tgt_mask)  # 前向传播

print(out.shape)  # batch, tgt_len, tgt_vocab
