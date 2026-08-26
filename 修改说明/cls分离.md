# TransReID CLS解耦分离模块 (CLS Separation) 修改说明

## 1. 核心思想
彻底剥离传统 ViT 中参与 Self-Attention 的 `[CLS]` token，避免全局令牌与局部 Patch 的特征相互污染。改为**动态生成 Query**，并通过**无参 Cross-Attention** 逐层、单向地向纯净的 Patch 流索取特征，实现宏观语义与微观纹理的完美解耦。

## 2. 新增配置参数 (YAML)
在配置文件中新增以下控制开关：
```yaml
MODEL:
  CLS_SEP: True              # CLS 解耦模块总开关 (开启即启用分离逻辑)
  CLS_GEN_TYPE: 'dynamic'    # Query 生成方式 (static: 静态参数 / dynamic: MLP动态生成)
  CLS_MLP_RATIO: 4.0         # 动态 Query 生成器与最终聚合器的隐藏层膨胀倍率