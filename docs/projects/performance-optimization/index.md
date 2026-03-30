# 性能优化

ALModel 17B XL 在 TPU v7x 上的训练性能分析与优化。

## 背景

- **模型**：17.43B 参数，混合 GLA/MLA 注意力，256 experts top-8 MoE
- **硬件**：TPU v7x 64 chips (128 TensorCores)，Peak bf16 2,307 TFLOPS/chip
- **基线 MFU**：11.89%（DP=4, FSDP=32, save_out_proj）

## 文档索引

| 文档 | 说明 |
|------|------|
| [性能理论分析框架](./analysis-framework) | 两级上界分析框架（Roofline Bound → System Bound），覆盖算子 roofline、显存模型、通信模型、overlap 模型 |
| [理论分析结果](./analysis-results) | 128 TPU v7x 上所有可行并行/重计算配置的 MFU 预测，最优配置 DP=4/FSDP=32 理论 MFU 31.5% |
| [性能优化工作拆解](./work-breakdown) | 优化工作全局视图：理论分析、Profiling sweep、Kernel profiling、优化项优先级排序 |
