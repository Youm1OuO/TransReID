import torch
import torch.nn as nn

class GenericMLP(nn.Module):
    """通用的多层感知机模块，支持自定义隐藏层维度、输出维度和激活函数"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ParameterFreeCrossAttention(nn.Module):
    """纯粹的矩阵点积 Cross-Attention，不引入任何可学习的 Wq, Wk, Wv"""
    def __init__(self, dim=768):
        super().__init__()
        self.scale = dim ** -0.5  # 缩放因子，防止 Softmax 梯度饱和

    def forward(self, query, keys):
        # query: [B, 1, 768]
        # keys:  [B, 196, 768]
        attn = (query @ keys.transpose(-2, -1)) * self.scale  # [B, 1, 196]
        attn = attn.softmax(dim=-1)
        out = attn @ keys  # [B, 1, 768]
        return out