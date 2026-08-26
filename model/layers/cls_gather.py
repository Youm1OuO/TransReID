import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicCLSGenerator(nn.Module):
    """使用 MLP 根据当前图像特征动态生成 12 层所需的初始 CLS Queries"""
    def __init__(self, embed_dim=768, depth=12):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        # 一开始的 MLP：压缩输入信息并展开为 12 层的 Queries
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, depth * embed_dim)
        )

    def forward(self, x_patches):
        # x_patches: [Batch, 196, 768]
        # 对 Patch 进行全局平均池化，提取整张图的总体表征
        x_pooled = x_patches.mean(dim=1)  # [Batch, 768]
        cls_queries = self.mlp(x_pooled)  # [Batch, 12 * 768]
        # 调整形状为 [Batch, 12, 768]
        return cls_queries.view(-1, self.depth, self.embed_dim)


class StaticCLSGenerator(nn.Module):
    """对比实验组：不使用 MLP，而是使用全局可学习的静态 Parameter"""
    def __init__(self, embed_dim=768, depth=12):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.cls_queries = nn.Parameter(torch.randn(1, depth, embed_dim))

    def forward(self, x_patches):
        B = x_patches.shape[0]
        # 扩展到当前 Batch 大小
        return self.cls_queries.expand(B, -1, -1)


class ParameterFreeCrossAttention(nn.Module):
    """纯粹的矩阵点积 Cross-Attention，不引入任何可学习的 Wq, Wk, Wv"""
    def __init__(self, dim=768):
        super().__init__()
        self.scale = dim ** -0.5  # 必须缩放，防止 Softmax 梯度饱和

    def forward(self, query, keys):
        # query (cls_i): [B, 1, 768]
        # keys (patches): [B, 196, 768]
        
        # 计算 Attention Map
        attn = (query @ keys.transpose(-2, -1)) * self.scale  # [B, 1, 196]
        attn = attn.softmax(dim=-1)
        
        # 加权提取 Patch 特征
        out = attn @ keys  # [B, 1, 768]
        return out


class CLSAggregator(nn.Module):
    """最终的聚合器：将 12 层的 CLS 融合为一个 768 维特征，送给最后的 Loss"""
    def __init__(self, depth=12, embed_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(depth * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, cls_list):
        # cls_list 是包含 12 个张量的列表，每个张量形状为 [B, 768]
        stacked_cls = torch.cat(cls_list, dim=-1)  # 拼接为 [B, 12 * 768]
        global_feat = self.mlp(stacked_cls)        # 映射回 [B, 768]
        return global_feat