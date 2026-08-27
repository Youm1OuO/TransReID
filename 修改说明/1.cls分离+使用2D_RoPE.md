# TransReID CLS解耦分离模块 (CLS Separation) 修改说明

## 1. 核心思想
彻底剥离传统 ViT 中参与 Self-Attention 的 `[CLS]` token，让 Transformer 内部变为纯粹的 Patch 空间结构交互。改为采用“小队长”机制，逐层生成 Query，通过 **Pre-LN 规范化 + 无参 Cross-Attention**，单向提取各层 Patch 的宏观语义，实现全局特征与微观纹理的解耦。该机制不仅提升了空间表征的纯度，还完美契合 De-ReID 中对非学习目标环境变异 (Variation-guided) 的特征捕获需求。

## 2. 新增配置参数 (YAML)
在配置文件中新增以下控制开关：
```yaml
MODEL:
  CLS_SEP: True              # CLS 解耦模块总开关 (开启即启用分离逻辑)
  CLS_GEN_TYPE: 'dynamic'    # Query 生成方式 (static: 静态参数 / dynamic: MLP基于输入动态生成)
  CLS_MLP_RATIO: 4.0         # 动态 Query 生成器与最终聚合器的隐藏层膨胀倍率
  USE_ROPE: True             # 是否启用 2D-RoPE 旋转位置编码
```

## 3. 核心修改实现细节

### 3.1 剥离全局 Token 与纯净 Patch 流
网络前向传播时不再向输入序列前置拼接 `[CLS]`，主干的 Transformer Block (`blk(x)`) 仅在 196 个 Patch 之间执行 Self-Attention。这防止了全局 Token 对底层图像结构的注意力绑架。

### 3.2 逐层 Query (小队长) 生成
根据参数选择两种模式为所有 Transformer 层提供 Query：
*   **Static (静态)**：初始化全局可学习的 Tensor `[1, depth, embed_dim]`。
*   **Dynamic (动态)**：将空间 Patch 特征转置后喂入 `self.cls_generator` (MLP)，基于当前图像自适应生成维度为 `[B, depth, embed_dim]` 的各层专属 Query。

### 3.3 无参 Cross-Attention 与 Pre-LN 保护
为了防止残差网络深度增加导致的特征方差爆炸与 Softmax 梯度消失：
*   计算注意力前，对当层 Query 和 Patch 特征 (`x`) 分别应用独立的 `LayerNorm`。
*   计算标准的无参缩放点积注意力：`attn = softmax( (Q_norm @ K_norm^T) * scale )`。
*   提取特征时，使用**未归一化**的原始 `x` 进行加权聚合 (`attn @ x`)，以完整保留流向下一层的残差幅值与语义。

### 3.4 最终特征聚合 (Aggregator)
收集所有层的交叉注意力输出形成 `[B, depth, embed_dim]`，将其转置并输入最终的 `self.cls_aggregator` (MLP)，映射融合为一维的 `global_feat [B, embed_dim]`。该输出与原版 TransReID 接口完全一致，可无缝对接 De-ReID 损失计算。

### 3.5 高级特性兼容 (2D-RoPE & JPM)
*   **兼容 2D-RoPE**：在主干 Self-Attention 阶段保留坐标旋转，但为抽象的全局 Query 向量强行补齐“零旋转 (`cos=1, sin=0`)”，使相对坐标系统互不干扰。
*   **兼容 JPM (局部重组)**：复用最后一层的 `query_12_norm` 向打乱后的局部分支 (`local_feats`) 索取特征，并通过统一的 `local_aggregator` 输出子区域全局表征。