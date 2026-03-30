# 性能理论分析框架

## 目标

建立系统化的性能分析框架，量化 ALModel 训练的理论 MFU 上界，并指导并行策略、重计算策略的选择。

## 核心思路

性能分析分为两级上界，逐级引入约束：

*   **Roofline Bound (物理上限)**: 假设无限显存单卡，无并行、无重计算。逐算子做 roofline 分析，得到硬件物理极限下的 MFU 上限。
    
*   **System Bound (工程上限)**: 引入真实显存约束。显存不够导致两个后果——需要多卡并行（引入通信开销）和需要重计算（引入额外计算开销）。对每种可行的配置组合计算 MFU，选最优配置。
    

两级之间的 gap 是优化空间，gap 的组成是优化方向。最终通过 profiling 拿到实测 MFU，与 System Bound 对比定位工程实现层面的损失。

```plaintext
MFU
 ▲
 │  ┌──────────────────────────────────────┐
 │  │  Roofline Bound                      │
 │  │  物理极限, 不可超越                    │
 │  └───────────────┬──────────────────────┘
 │                  │ gap = 并行通信 + 重计算 + bubble + 调度损失
 │  ┌───────────────▼──────────────────────┐
 │  │  System Bound                        │
 │  │  每种 (并行, remat, batch) 一个值     │
 │  └───────────────┬──────────────────────┘
 │                  │ gap = 算子实现效率 + XLA 调度 + 数据加载等
 │  ┌───────────────▼──────────────────────┐
 │  │  实测 MFU (Profiling)                │
 │  └──────────────────────────────────────┘
 └──────────────────────────────────────────►

```

## MFU 定义

```plaintext
MFU = F_useful / (Peak_FLOPS × T_step)

F_useful = 模型训练一个 step 的最少计算量 (forward + backward, 不含 remat/通信)
         = 3 × F_forward

F_useful 是固定的，不随策略变化。策略只影响分母中的 T_step。

```

## Roofline Bound

### 思路

假设模型运行在一张无限显存的单卡上，不需要并行（无通信），不需要重计算（激活全部保存）。唯一的约束来自硬件本身的算力和访存带宽——即 roofline。

### 需要做的事情

**Step 1: 收集硬件 spec**

确定目标硬件的两个关键参数：

*   峰值算力 Peak\_FLOPS (对应训练 dtype，通常 bf16)
    
*   HBM 峰值带宽 Peak\_BW
    

由此算出硬件拐点 `Ridge Point = Peak_FLOPS / Peak_BW`，算术强度高于拐点的算子是 compute-bound，低于拐点的是 memory-bound。

**Step 2: 建立算子清单**

遍历模型 forward + backward 的所有算子，对每个算子理论计算：

*   FLOPs: 计算量
    
*   Bytes: 总访存量（输入 + 输出 + 权重的读写）
    
*   AI = FLOPs / Bytes: 算术强度
    

算子需要覆盖完整的计算图，包括但不限于：

| 类别 | 算子 | 需要确认的参数 |
| --- | --- | --- |
| 线性投影 | Attention QKV/Output, MLA low-rank projections, FFN/Expert wi/wo | shape, dtype |
| Attention 计算 | MLA 的 QK^T / Softmax×V, GLA 的 chunk kernel | seq\_len, heads, head\_dim, chunk\_size |
| 激活函数 | SiLU, Softmax, Sigmoid | tensor shape |
| 归一化 | RMSNorm (每层多处) | hidden\_dim |
| 路由 | Router score, top-k selection | num\_experts, tokens |
| Embedding | 前向 lookup, 反向 gradient scatter | vocab\_size, d\_model |

> 示例: 对一个 matmul `[M, K] × [K, N]`，FLOPs = 2MKN，Bytes = (MK + KN + MN) × dtype\_size，AI = 2MKN / Bytes。

**Step 3: 逐算子计算 roofline 耗时**

对每个算子取 compute 和 memory 的较大值：

```plaintext
T_op = max(FLOPs / Peak_FLOPS, Bytes / Peak_BW)

```

**Step 4: 求和得到理论 step time 和 MFU 上界**

```plaintext
T_step_ideal = Σ T_op   (所有算子, forward + backward)

MFU_roofline = F_useful / (Peak_FLOPS × T_step_ideal)

```

### 注意事项

*   Backward 的算子列表与 forward 不完全对称，需要单独分析（如 backward 中有额外的转置 matmul，gradient 累加等）。
    
*   对于 XLA 融合后的算子（如 flash attention），应按融合后的整体分析，而非拆开算。
    
*   对于自定义 Pallas kernel（如 GLA chunk kernel），需要根据 kernel 实现单独计算 FLOPs 和访存模式。
    
*   MoE 路由引入的不规则计算（token dispatch/combine、expert load imbalance）在 roofline 分析中按平均值计算。
    

## System Bound

### 思路

引入真实硬件的显存约束。对每一种可行的配置组合 `(DP, FSDP, TP, EP, PP, CP, remat_policy, batch_size)` 回答三个问题：

```plaintext
Q1: 能不能跑？    → Memory Model
Q2: 要算多久？    → Compute Model + Communication Model
Q3: 能不能重叠？  → Overlap Model

```

### 需要做的事情

**Step 1: Memory Model — 判断配置是否可行**

计算给定配置下每张卡的显存占用，确认不超过 HBM 容量。

```plaintext
M_total = M_params + M_optimizer + M_activations + M_buffers  ≤  HBM_per_device

```

各项的关键影响因素：

| 显存组成 | 公式要点 | 受哪些策略影响 |
| --- | --- | --- |
| 参数 | attention / TP, experts / EP, 总和 / FSDP | TP, EP, FSDP |
| 优化器状态 | 参数量 × 12B (AdamW fp32) / FSDP | FSDP |
| 激活 | ∝ batch × seq / (TP × CP), remat 减少保存量 | batch, remat, TP, CP |
| 通信缓冲 | All-to-All buffer, gradient buffer 等 | EP, DP |

> 示例: 如果 M\_total = 70 GB，TPU v4 HBM = 32 GB → 不可行，需要引入 FSDP 切优化器状态，或减小 batch。

**Step 2: Compute Model — 计算含重计算的总计算量**

```plaintext
F_total = F_useful + F_remat

T_compute = F_total / (N_devices × Peak_per_device)

```

F\_remat 取决于重计算策略：

| 策略 | F\_remat | 说明 |
| --- | --- | --- |
| none | 0 | 不重计算，激活全保存 |
| save\_dot\_except\_mlp | F\_mlp\_forward | MLP/FFN forward 重算一遍 |
| full | F\_forward | 整个 forward 重算一遍 |

注意 remat 策略不是自由选择的——它和 Memory Model 耦合。选择更轻的 remat 意味着更多激活需要保存，可能导致显存不足。

**Step 3: Communication Model — 计算各并行策略的通信时间**

对每种并行维度分别建模通信量和通信时间：

| 并行策略 | 通信原语 | 通信量 | 频率 |
| --- | --- | --- | --- |
| EP | All-to-All | tokens × top\_k × d\_model × dtype | 每 MoE 层 2 次 (dispatch + combine) |
| FSDP | AllGather + ReduceScatter | 每层参数量 / FSDP | 每层 forward 1 次 AG, backward 1 次 RS |
| TP | AllReduce | tokens × d\_model × dtype | 每层 2 次 (attention 后 + FFN 后) |
| DP | AllReduce | 总梯度量 | 每 step 1 次 |
| PP | P2P + Bubble | 激活量 (microbatch) | 每 microbatch |

通信时间需要区分网络层级：

*   ICI（芯片间互联，同 pod）：高带宽、低延迟，通常用于 TP、EP
    
*   DCN（数据中心网络，跨 pod）：低带宽、高延迟，通常用于 DP、FSDP
    

```plaintext
T_collective = latency + data_size / effective_bandwidth

```

**Step 4: Overlap Model — 评估通信掩盖程度**

判断哪些通信可以和计算并行执行（无数据依赖）：

| 通信 | 可掩盖方式 | 条件 |
| --- | --- | --- |
| EP All-to-All dispatch | 分块 pipeline: dispatch/expert compute/combine 流水线执行 | 通过分块在 MoE block 内部实现通信与计算的 pipeline |
| FSDP AllGather | 第 n+1 层 prefetch 与第 n 层计算重叠 | FSDP prefetch |
| DP AllReduce | 和 backward 最后几层计算重叠 | gradient bucketing |
| PP P2P | 和计算 pipeline 重叠 | 1F1B schedule |

```plaintext
T_comm_exposed = T_comm_total - T_overlapped

```

overlap\_ratio 的精确值需要从 profiling trace 中测量，理论上可以根据计算图的依赖关系估算上界。

**Step 5: 计算最终 Step Time 和 MFU**

```plaintext
T_step = max(T_compute, T_comm_exposed) + T_bubble + T_overhead

MFU_system = F_useful / (N_devices × Peak_per_device × T_step)

```

### 配置空间搜索

对所有合法的并行分解（乘积 = N\_devices）：

```plaintext
for each (DP, FSDP, TP, EP, PP, CP):
  for each remat_policy:
    for each batch_size:
      if Memory Model feasible:
        compute T_step → MFU
        record
select configuration with max MFU

```
> 示例: 对 8 卡的合法分解包括 EP=8、EP=4×FSDP=2、TP=2×EP=4、TP=8 等。每种分解对应不同的通信模式、显存分布和重计算需求。

## 分析流程总结

```plaintext
Phase 1: 计算 Roofline Bound
  输入: 模型架构参数 + 硬件 spec
  输出: MFU_roofline (物理上限)
  方法: 逐算子 roofline 求和

Phase 2: 计算 System Bound
  输入: 硬件数量 + 显存容量
  输出: 各可行配置的 MFU_system, 最优配置
  方法: Memory 可行性筛选 → Compute/Comm/Overlap 建模 → 枚举比较

Phase 3: Profiling 验证
  输入: 选定配置的实际运行 trace
  输出: 实测 MFU, 各算子/通信的实际耗时
  方法: JAX profiler / tensorboard / perfetto

Phase 4: Gap 分析与优化
  Roofline Bound → System Bound:  gap 来自并行通信 + 重计算
  System Bound   → 实测 MFU:      gap 来自算子实现效率 + XLA 调度 + 数据加载等
  定位最大 gap 项, 针对性优化

```