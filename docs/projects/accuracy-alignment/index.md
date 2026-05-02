---
title: Accuracy Alignment
---

# Accuracy Alignment

跨框架精度对齐标准化 workflow：左右两侧各跑一个普通 Falcon dump Exp，argus 完成**捕获 → 比对 → 定位 → 收敛**；reference 收敛到 HF transformers；mapping / 阈值 / 结论沉淀到 alignment report，后续可复用。

**常见对齐对**：

- sgl-jax ↔ HF transformers — sgl-jax 接入新模型 / 新硬件 / 周期审计
- MaxText ↔ Megatron-LM — MaxText 升级关键路径前的回归
- sgl-jax ↔ MaxText — RLHF 后 serving / training 一致性
- 同框架历史版本 / 同框架不同精度（FP32 vs BF16/TF32）

## 文档

- [设计方案](./design.md) — 两个 dump Exp + alignment report / 整体架构（argus + Falcon）/ 5 步流程
