import torch


# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 原始论文核心参数
d_model = 512   # 模型维度
n_heads = 8     # 多头注意力头数（必须被d_model整除）
d_ff = 2048     # 前馈神经网络隐藏维度
n_layers = 6    # 编码器/解码器堆叠层数
dropout = 0.1   # Dropout概率

if __name__ == '__main__':
    print("==============================")
    print(f"Transformer device: {device}")
    print(f"d_model = {d_model}\nn_heads = {n_heads}\nd_ff = {d_ff}\nn_layers = {n_layers}\ndropout = {dropout}")
    print("==============================")