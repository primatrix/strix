# Fused EP-MoE 系统性理论性能分析

> 参数化通用分析框架 — 覆盖 DP/EP/TP sharding、四阶段计算/访存/通信公式、全局调度 DAG 与 overlap tradeoff、prefill/decode regime crossover
>
> 主要硬件参考: TPU v7x; 可扩展至 v6e

---

## 1. Prologue: 参数空间与 Sharding 模型

### 1.1 符号表

#### 模型参数

| 符号 | 含义 | 说明 |
|------|------|------|
| $N_E$ | 路由专家总数 | 例: 128, 256, 384 |
| $H$ | 隐藏层维度 (`hidden_size`) | 例: 2048, 4096, 8192 |
| $I$ | 专家 FFN 中间维度 (`intermediate_size`) | 例: 512, 768, 2048 |
| $I_{SE}$ | 共享专家 (SE) 中间维度 | 通常 = $I$ |
| $k$ | Top-K 路由 (每 token 激活专家数) | 例: 2, 4, 8 |
| $T$ | 全局 token 数 (`batch_size` × `seq_len` 或 decode batch) | |
| $B_w$ | 权重元素字节数 | BF16: 2, FP8: 1 |
| $B_a$ | 激活元素字节数 | BF16: 2 |

#### 硬件参数 (TPU v7x)

| 符号 | 含义 | v7x 值 |
|------|------|--------|
| $BW_{HBM}$ | HBM 带宽 | 3,690 GB/s |
| $F_{MXU}$ | MXU 峰值算力 (BF16, per TensorCore) | 1,154 TFLOPS |
| $BW_{ICI}$ | ICI 链路带宽 (per link) | ~200 GB/s (双向) |
| $\text{VMEM}$ | VMEM 容量 | 64 MiB |
| $\text{HBM}_{total}$ | HBM 总容量 | 96 GB |
| $\text{Ridge}$ | Roofline 拐点 $= F_{MXU} / BW_{HBM}$ | 313 FLOPs/byte |
| $t_{packing}$ | BF16 打包因子 = `MXU_dim` / 128 | 2 (for BF16) |

#### Block Config 参数

| 符号 | 含义 | 约束 |
|------|------|------|
| $bt$ | 外层 token tile (routing/output) | $bt \leq T_L$, $bt \mid T_L$ |
| $bts$ | Expert FFN 内层 token staging tile | $\leq bt \times P_{EP}$ |
| $btc$ | MXU GEMM M 维度计算 tile | $btc \mid bts$ |
| $bf$ | $I$ 维分块 | $bf \mid I$ |
| $bfc$ | $I$ 维计算分块 | $bfc \mid bf$ |
| $bd1$ | $H$ 维分块 (FFN1 K 维) | $bd1 \mid H$ |
| $bd1c$ | $H$ 维计算分块 (FFN1) | $bd1c \mid bd1$ |
| $bd2$ | $H$ 维分块 (FFN2 N 维) | $bd2 \mid H$ |
| $bd2c$ | $H$ 维计算分块 (FFN2) | $bd2c \mid bd2$ |
| $bse$ | SE $I$ 维分块 | $bse \mid I_{SE}$ |

#### 派生循环计数

| 符号 | 公式 |
|------|------|
| $n_{bt}$ | $T_L / bt$ |
| $n_{bf}$ | $I / bf$ |
| $n_{bd1}$ | $H / bd1$ |
| $n_{bd2}$ | $H / bd2$ |
| $n_{bse}$ | $\lceil I_{SE} / bse \rceil$ |

### 1.2 Mesh 与 EP/DP/TP 分解

Kernel 运行在 2D JAX mesh `(data, tensor)` 上:

- $P_{DP}$ = `data` axis size (DP 并行度)
- $P_{TP}$ = `tensor` axis size (TP 并行度)
- $P_{EP} = P_{DP} \times P_{TP}$ (EP = 全 mesh 乘积)
- 设备 ID: $d = d_{DP} \times P_{TP} + d_{TP}$

**`shard_map` 分片规则**:

| 张量 | Partition Spec | 本地 Shape |
|------|---------------|-----------|
| `tokens` | `P((dp, tp))` 第 0 轴 | $(T_L, H)$ |
| `w1, w2, w3` | `P((dp, tp))` 第 0 轴 | $(E_L, H, I)$ 或 $(E_L, I, H)$ |
| `topk_weights, topk_ids` | `P((dp, tp))` 第 0 轴 | $(T_L, k)$ |
| A2A scratch buffers | `P()` (全复制) | 本地分配 |
| SE 权重 | `P()` (全复制) | 完整 SE 权重 |
| `output` | `P((dp, tp))` 第 0 轴 | $(T_L, H)$ |

### 1.3 每设备派生量

| 符号 | 公式 | 含义 |
|------|------|------|
| $E_L$ | $N_E / P_{EP}$ | 本地专家数 |
| $T_L$ | $T / P_{EP}$ | 本地 token 数 (routing 前) |
| $T_{a2a}$ | $\text{align}(bt \times P_{EP}, bts)$ | 每专家 scatter buffer 最大 token 数 |
| $\bar{n}_e$ | $T_L \times k / E_L$ | 每专家期望 token 数 (均匀路由) |
| $T_L^{max}$ | $T \times k$ (当 $E_L \geq k$) | 最坏情况 token-expert pair 数 |
| $h_{pt}$ | $H / t_{packing}$ | 每 packing 维的 hidden 元素数 |
| $bd1_{pt}$ | $bd1 / t_{packing}$ | 每 packing 维的 bd1 元素数 |
| $bd2_{pt}$ | $bd2 / t_{packing}$ | 每 packing 维的 bd2 元素数 |

### 1.4 Expert Buffer Count 与路径选择

A2A scatter 需要 HBM scratch buffer 存放中间数据。Buffer 数决定了两种执行路径:

$$N_{buf} = \min\left(E_L, \max\left(2, \left\lfloor \frac{0.03 \times \text{HBM}_{total}}{2 \times T_{a2a} \times H \times B_a} \right\rfloor\right)\right)$$

| 条件 | 路径 | 特征 |
|------|------|------|
| $N_{buf} \geq E_L$ | **Batch Scatter** | 一次性发射所有 scatter DMA，无 buffer 复用 barrier |
| $N_{buf} < E_L$ | **Pipelined** | 逐专家 scatter，buffer 复用需 `sync_barrier` |

> **Decode 场景**: 通常 $N_{buf} = E_L$（token 少，buffer 需求小），走 Batch Scatter 路径。
>
> **Prefill 场景**: 可能 $N_{buf} < E_L$（大 batch 使 buffer 需求超出 3% HBM），走 Pipelined 路径。

### 1.5 负载均衡对 $T_L$ 的影响

在 EP 模式下，all-to-all scatter 将 token 发送到路由专家所在设备。路由不均会导致某些设备接收远多于 $T_L$ 的 token：

- 设备 $d$ 的有效 token 数: $T_L^{(d)} = \sum_{e \in \text{experts}(d)} n_e$
- 均匀路由: $T_L^{(d)} = T_L \times k \times E_L / N_E = T_L \times k$（因为每 token 路由到 $k$ 个专家）
- 实际: $T_L^{(d)}$ 取决于路由分布，上界为 $T \times k$

对于方案 A (weight-streaming): $n_{bt}^{(d)} = T_L^{(d)} / bt$，权重加载次数与 $n_{bt}$ 成正比。

**整个 kernel 延迟由最慢设备决定** ($\max_d T_L^{(d)}$)，因为设备间有 `sync_barrier`。

---

## 2. 阶段分析

### 2.1 Stage 1: Gate/TopK 路由 + AllReduce Metadata + Permute

#### 2.1.1 操作描述

1. 从 HBM 读取预计算的 `topk_weights` (F32) 和 `topk_ids` (S32)
2. 生成路由矩阵 `t2e_routing`: 展开为 $(bt, k, N_E^{padded})$ mask
3. 按专家聚合 token 计数 `expert_sizes`: $(1, N_E^{padded})$
4. `all_reduce_metadata()`: 跨 $P_{EP}$ 设备交换路由元数据

#### 2.1.2 计算公式

| 指标 | 公式 | 说明 |
|------|------|------|
| Routing mask 生成 | $bt \times k \times N_E^{padded}$ 次比较 | VPU (broadcast + compare) |
| Expert sizes reduce | $bt \times k$ 次累加 | VPU sum |
| Prefix sum | $P_{EP} \times N_E$ 次加法 | VPU，allgather 后本地计算 |
| **总 VPU ops** | $O(bt \times k \times N_E)$ | |

**MXU FLOPs**: 0 (本阶段无矩阵乘法)

#### 2.1.3 访存公式

| 指标 | 公式 | 单位 |
|------|------|------|
| HBM 读取 | $bt \times k \times (4 + 4)$ | bytes (F32 weights + S32 ids) |
| HBM 写入 | 0 | 结果在 VMEM/SMEM |
| **总 HBM** | $8 \times bt \times k$ | bytes |

#### 2.1.4 通信公式

AllReduce metadata 使用递归倍增 (power-of-2) 或逐步轮转 (non-power-of-2) allgather:

| 指标 | 公式 (power-of-2) | 说明 |
|------|-------------------|------|
| 轮数 | $\lceil \log_2 P_{EP} \rceil$ | 递归倍增 |
| 每轮数据量 | $\text{chunk} \times N_E^{padded} \times 4$ bytes | chunk = $2^{round}$ |
| 每轮操作 | 1× `remote_copy` send + 1× recv wait + 1× send wait | |
| 同步 barrier | 每轮 1 次 + 最终 1 次 | $\lceil \log_2 P_{EP} \rceil + 1$ 次 barrier |
| **总 ICI 通信** | $P_{EP} \times N_E^{padded} \times 4$ bytes | 最终每设备拥有完整路由表 |

**时间估算**:

$$T_{S1} = \lceil \log_2 P_{EP} \rceil \times (t_{barrier} + t_{ICI\_latency}) + \frac{P_{EP} \times N_E \times 4}{BW_{ICI}} + \frac{bt \times k \times N_E}{V_{VPU}}$$

其中 $t_{barrier} \approx 3\text{-}5 \mu s$, $t_{ICI\_latency} \approx 1\text{-}2 \mu s$, $V_{VPU}$ 为 VPU 吞吐。

#### 2.1.5 VMEM 占用

| 分配 | Shape | 大小 | 用途 |
|------|-------|------|------|
| `topk_ids_x2_vmem` | $(2, bt, k)$ S32 | $8 \times bt \times k$ | 双缓冲 `topk_ids` |
| `topk_weights_x2_vmem` | $(2, bt, k)$ F32 | $8 \times bt \times k$ | 双缓冲 `topk_weights` |
| `d2e_count_vmem` | $(P_{EP}, 1, N_E^{padded})$ S32 | $4 \times P_{EP} \times N_E^{padded}$ | allgather 工作区 |
| `t2e_routing_smem` | $(bt, k)$ | 已在 SMEM | 路由矩阵 |

$$\text{VMEM}_{S1} = 16 \times bt \times k + 4 \times P_{EP} \times N_E^{padded} + \text{routing temporaries}$$

#### 2.1.6 LLO Overhead 观测

最新 LLO dump (bt=32, bf=512, bd1=1024, bd2=1024, bse=512):

- Stage 1 (routing) 代码已内联到主 expert loop 中，不再有独立的 routing 区域
- Routing 逻辑（`vrot.slane`, `vbcast.lane`, `vpop`）与 scatter DMA 交叠执行
- Stage 1 在总 kernel 时间中占比极小 (<1%)，但其 barrier latency 是不可压缩的固定开销
- 主 critical path 位于 FFN1 matmul 内层循环 (748 bundles, region4355/4377 vld→vmul→vst 链)，其次是 scatter 循环入口 (450 bundles, region4322)

---

### 2.2 Stage 2: AllToAll Dispatch + Expert Compute + AllToAll Combine

这是 kernel 的**绝对瓶颈阶段**。

#### 2.2.1 Sub-stage 2a: AllToAll Scatter (Dispatch)

**操作**: 遍历 $bt$ 个 token，对每个 token 的 $k$ 个路由专家，将 token 数据 DMA 发送至目标设备的 scatter buffer。

##### 计算

无计算 (纯数据搬运)。

##### 访存

| 指标 | 公式 | 说明 |
|------|------|------|
| HBM 读取 (source) | $bt \times H \times B_a$ | 源 token 从 HBM 读出 |
| HBM 写入 (dest) | $bt \times k \times H \times B_a$ | 散布到 $k$ 个 buffer slot |
| **总 HBM 上界** | $bt \times (1 + k) \times H \times B_a$ | |

##### 通信

| 指标 | 公式 | 说明 |
|------|------|------|
| 本地 DMA 占比 | $E_L / N_E$ | 目标专家在本设备的概率 |
| 远程 ICI 量 | $bt \times k \times (1 - E_L / N_E) \times H \times B_a$ | |
| DMA 发起次数 | $bt \times k$ | 每 token-expert pair 一次 DMA |

##### 时间下界

$$T_{2a} = \max\left(\frac{bt \times k \times H \times B_a}{BW_{HBM}}, \frac{bt \times k \times (1 - E_L/N_E) \times H \times B_a}{BW_{ICI}}, bt \times k \times t_{DMA\_setup}\right)$$

> **Decode 场景**: 数据量极小 (例: $64 \times 8 \times 8192 \times 2 = 8$ MB)，瓶颈为 DMA setup latency 和 ICI latency，而非 BW。

#### 2.2.2 Sub-stage 2b: Expert FFN Compute (核心瓶颈)

对 $E_L$ 个本地专家逐一执行 FFN: Gate(W1) + Up(W3) → Activation → Down(W2)。

##### 每专家预期 Token 数

均匀路由下:

$$\bar{n}_e = \frac{T \times k}{N_E} = \frac{T_L \times P_{EP} \times k}{N_E} = \frac{T_L \times k}{E_L}$$

零 token 专家概率 (Poisson 近似, $\lambda = \bar{n}_e$):

$$P(n_e = 0) = e^{-\bar{n}_e}$$

> 当 $\bar{n}_e = 8$: $P(0) = 0.034\%$，几乎所有本地专家都收到 token。

##### FFN1: Gate + Up (per expert)

遍历 $(bf\_id, bd1\_id)$，对每个组合执行 Gate 和 Up 两个 matmul:

```
token[btc, bd1c/t_p] × W1[bd1c/t_p, bfc] → gate_acc[btc, bfc]   (MXU)
token[btc, bd1c/t_p] × W3[bd1c/t_p, bfc] → up_acc[btc, bfc]     (MXU)
```

**权重 HBM 读取 (per expert)**:

| 权重 | Shape | 每 tile 大小 | Tile 数 | 总量 |
|------|-------|-------------|---------|------|
| W1 | $(t_p, bd1_{pt}, bf)$ | $t_p \times bd1_{pt} \times bf \times B_w$ | $n_{bf} \times n_{bd1}$ | $H \times I \times B_w$ |
| W3 | 同 W1 | 同上 | $n_{bf} \times n_{bd1}$ | $H \times I \times B_w$ |

$$\text{HBM}_{FFN1}^{weight} = 2 \times H \times I \times B_w$$

**Token HBM 读取 (per expert, per `bf_id`)**:

$$\text{HBM}_{FFN1}^{token} = n_{bf} \times bts \times H \times B_a$$

> Token 读取在 $(bf\_id, bd1\_id=0)$ 时 staging 一次，跨 $n_{bd1}$ 复用。

**MXU FLOPs (per expert, executed)**:

$$\text{FLOPs}_{FFN1}^{exec} = n_{bf} \times n_{bd1} \times t_p \times 2 \times (2 \times btc \times bd1c_{pt} \times bfc)$$

简化为: $2 \times 2 \times bts \times H \times I = 4 \times bts \times H \times I$ (Gate + Up 两个 matmul)

**MXU FLOPs (per expert, useful)**:

$$\text{FLOPs}_{FFN1}^{useful} = 4 \times \bar{n}_e \times H \times I$$

> Useful FLOPs 取决于实际 token 数 $\bar{n}_e$ 而非 tile 大小 $bts$。

##### Activation (per expert)

SiLU/GeLU/SwiGLU 在 VPU 上执行:

$$\text{VPU}_{act} = O(bts \times bf \times n_{bf}) = O(bts \times I)$$

相比 MXU FLOPs 可忽略 ($< 0.01\%$)。

##### FFN2: Down (per expert)

遍历 $(bf\_id, bd2\_id)$:

```
act[btc, bfc] × W2[bfc, bd2c/t_p] → output[btc, bd2c/t_p]   (MXU)
```

**权重 HBM**:

$$\text{HBM}_{FFN2}^{weight} = I \times H \times B_w$$

**Result HBM 写入 (per expert)**:

$$\text{HBM}_{FFN2}^{write} = n_{bf} \times bts \times H \times B_a$$

**MXU FLOPs (per expert, executed)**:

$$\text{FLOPs}_{FFN2}^{exec} = 2 \times bts \times I \times H$$

##### 单专家汇总

| 指标 | 公式 |
|------|------|
| 权重 HBM 读取 | $3 \times H \times I \times B_w$ |
| Token HBM 读取 | $n_{bf} \times bts \times H \times B_a$ |
| Result HBM 写入 | $n_{bf} \times bts \times H \times B_a$ |
| **MXU FLOPs (exec)** | $6 \times bts \times H \times I$ |
| **MXU FLOPs (useful)** | $6 \times \bar{n}_e \times H \times I$ |
| **AI (exec)** | $6 \times bts \times H \times I / (3HIB_w + 2 \times n_{bf} \times bts \times H \times B_a)$ |

当 $3HIB_w \gg 2 \times n_{bf} \times bts \times H \times B_a$ (decode 下成立):

$$\text{AI}_{exec} \approx \frac{6 \times bts \times H \times I}{3 \times H \times I \times B_w} = \frac{2 \times bts}{B_w}$$

$$\text{AI}_{useful} \approx \frac{2 \times \bar{n}_e}{B_w}$$

> **BF16** ($B_w = 2$): $\text{AI}_{useful} = \bar{n}_e$。当 $\bar{n}_e = 8$: AI = 8 FLOPs/byte，远低于 ridge point 313。

##### 全部 $E_L$ 专家汇总

| 指标 | 公式 |
|------|------|
| **权重 HBM** | $E_L \times 3 \times H \times I \times B_w$ |
| Token + Result HBM | $E_L \times 2 \times n_{bf} \times bts \times H \times B_a$ |
| **MXU FLOPs** | $E_L \times 6 \times bts \times H \times I$ |
| **$T_{HBM}$ (时间下界)** | $E_L \times 3HIB_w / BW_{HBM}$ |
| **$T_{compute}$** | $E_L \times 6 \times bts \times HI / F_{MXU}$ |
| **瓶颈判定** | $T_{HBM} / T_{compute} = 3HIB_w \times F_{MXU} / (6 \times bts \times HI \times BW_{HBM}) = B_w \times F_{MXU} / (2 \times bts \times BW_{HBM})$ |

> 当此比值 > 1 时为 HBM-bound。对 v7x BF16: $2 \times 1154T / (2 \times bts \times 3690G) = 313 / bts$。当 $bts < 313$ (decode 常态) 时**深度 HBM-bound**。

##### DMA:Compute 比 (双缓冲分析)

权重双缓冲 ping-pong (bd1/bd2 维度):

| 指标 | 公式 |
|------|------|
| 每 tile DMA 时间 | $t_p \times bd1_{pt} \times bf \times B_w / BW_{HBM}$ |
| 每 tile Compute 时间 | $t_p \times 2 \times btc \times bd1c_{pt} \times bfc / F_{MXU}$ |
| **DMA:Compute 比** | $(bd1_{pt} \times bf \times B_w \times F_{MXU}) / (2 \times btc \times bd1c_{pt} \times bfc \times BW_{HBM})$ |

> 当此比 > 1 时 DMA 无法被 compute 完全覆盖。双缓冲有效但不能消除等待。

#### 2.2.3 Sub-stage 2c: AllToAll Gather (Combine)

**操作**: 每个专家 FFN 完成后启动 gather，将结果发送回 token 的原始设备。

##### 通信

| 指标 | 公式 | 说明 |
|------|------|------|
| 每专家 gather 量 | $\bar{n}_e \times H \times B_a$ | |
| 全部专家 gather | $E_L \times \bar{n}_e \times H \times B_a = T_L \times k \times H \times B_a$ | 对称于 scatter |
| 远程 ICI 占比 | $1 - E_L / N_E$ | |
| DMA 发起次数 | $E_L \times P_{EP}$ | 每专家向每个设备发一次 |

##### Pipeline 特性

Gather 与后续专家的 FFN 计算 pipeline 化:
- `start_a2a_gather(expert_i)` 在 `expert_ffn(expert_i)` 结束后立即启动
- `expert_ffn(expert_{i+1})` 与 gather DMA 并行执行
- 最终 `wait_a2a_gather_recv_all()` 在所有专家完成后等待

$$T_{2c}^{visible} = \max(0, T_{2c}^{total} - (E_L - 1) \times T_{expert})$$

> 当 $T_{expert} \gg T_{gather}^{per\_expert}$ (decode 常态) 时，gather 完全被 compute 覆盖，$T_{2c}^{visible} \approx 0$。

#### 2.2.4 Stage 2 VMEM 占用公式

这是 VMEM 压力最大的阶段。同时驻留:

| 分配 | 公式 | 说明 |
|------|------|------|
| W1 双缓冲 | $2 \times t_p \times bd1_{pt} \times bf \times B_w$ | Gate 权重 ping-pong |
| W3 双缓冲 | $2 \times t_p \times bd1_{pt} \times bf \times B_w$ | Up 权重 ping-pong |
| W2 双缓冲 | $2 \times t_p \times bf \times bd2_{pt} \times B_w$ | Down 权重 ping-pong |
| Token staging 双缓冲 | $2 \times bts \times t_p \times bd1_{pt} \times B_a$ | HBM→VMEM 搬运 |
| FFN1 累加器 | $2 \times bts \times bf \times 4$ | Gate + Up, F32 |
| FFN2 result 三缓冲 | $3 \times bts \times t_p \times bd2_{pt} \times B_a$ | Load/Compute/Store |
| **SE 权重双缓冲** | $3 \times 2 \times t_p \times bd1_{pt} \times bse \times B_w$ | 如果 SE 与 expert loop 交错 |
| **SE 累加器** | $2 \times bt \times H \times 4$ | F32, 2 个 buf (`bt_id` 双缓冲) |

$$\boxed{\text{VMEM}_{S2} = 6 \times t_p \times bd_{w\_max} \times bf \times B_w + 2 \times bts \times t_p \times bd1_{pt} \times B_a + 8 \times bts \times bf + 3 \times bts \times t_p \times bd2_{pt} \times B_a + 6 \times t_p \times bd1_{pt} \times bse \times B_w + 8 \times bt \times H}$$

> **约束**: $\text{VMEM}_{S2} \leq \text{VMEM}$。这限制了 $bts, bf, bd1, bd2, bse$ 的可选范围。

---

### 2.3 Stage 3: MoE 输出聚合 + SE 输出合并

#### 2.3.1 操作描述

`acc_and_store_output()`: 双缓冲 pipeline 遍历 $bt$ 个 token:
1. 从 `a2a_g_hbm` 加载每 token 的 $k$ 个专家输出
2. 使用 `topk_weights` 加权求和 (F32 精度)
3. 加上 SE 输出 (从 `b_se_acc_vmem` 读取)
4. 转 BF16 写入 `output_hbm`

#### 2.3.2 访存公式

| 指标 | 公式 | 说明 |
|------|------|------|
| HBM 读取 (expert outputs) | $bt \times k \times H \times B_a$ | 从 gather buffer |
| HBM 读取 (`topk_weights`) | 0 | 已在 VMEM |
| VMEM 读取 (SE output) | $bt \times H \times 4$ | F32 累加器，已在 VMEM |
| HBM 写入 (final output) | $bt \times H \times B_a$ | 最终输出 |
| **总 HBM** | $bt \times (k + 1) \times H \times B_a + bt \times H \times 4$ | 读+写 |

#### 2.3.3 计算公式

| 指标 | 公式 | 说明 |
|------|------|------|
| 加权求和 | $bt \times k \times H$ | VPU multiply + add |
| SE 加法 | $bt \times H$ | VPU add |
| 精度转换 | $bt \times H$ | F32 → BF16 |
| **总 VPU ops** | $bt \times (k + 2) \times H$ | |

**MXU FLOPs**: 0

#### 2.3.4 时间估算

$$T_{S3} = \frac{bt \times k \times H \times B_a}{BW_{HBM}} + t_{VPU} + t_{DMA\_output}$$

> Decode 下数据量小 (例 8 MB)，VPU 计算极快，总耗时 ~5-10 μs，由 DMA latency 主导。

---

### 2.4 Stage 4: Shared Expert Compute (数据独立)

#### 2.4.1 数据依赖特性

**Stage 4 与 Stage 1-2 完全独立:**
- 上游依赖: 仅 `tokens_input` (原始 token 数据)
- 下游依赖: Stage 3 需要 SE 输出
- 不依赖 routing 结果、scatter/gather 状态、或任何路由专家的中间数据

这意味着 SE 计算可以在 token 数据可用后的**任何时刻**开始。

#### 2.4.2 操作描述

$n_{bse} = \lceil I_{SE} / bse \rceil$ 个 SE block，每个 block 执行完整的 SwiGLU FFN:
- FFN1: Gate(`W1_SE`) + Up(`W3_SE`)，遍历 $(n_{bd1}, t_p)$
- Activation: SiLU(gate) × up
- FFN2: Down(`W2_SE`)，遍历 $(n_{bd2}, t_p)$

#### 2.4.3 每 SE Block 访存

| 指标 | 公式 |
|------|------|
| `W1_SE` 读取 | $n_{bd1} \times t_p \times bd1_{pt} \times bse \times B_w$ |
| `W3_SE` 读取 | 同上 |
| `W2_SE` 读取 | $n_{bd2} \times t_p \times bse \times bd2_{pt} \times B_w$ |
| Token 读取 | $n_{bd1} \times bt \times H \times B_a$ (跨 bd1 复用) |
| **Per block 权重** | $(2 \times H + H) \times bse \times B_w = 3 \times H \times bse \times B_w$ |

#### 2.4.4 全部 SE 访存

$$\text{HBM}_{SE}^{weight} = n_{bse} \times 3 \times H \times bse \times B_w = 3 \times H \times I_{SE} \times B_w$$

$$\text{HBM}_{SE}^{token} = n_{bse} \times n_{bd1} \times bt \times H \times B_a$$

> Token 读取量取决于 SE 权重切分方式。当前每个 SE block 需要重新 staging token。

#### 2.4.5 每 SE Block FLOPs

| 操作 | 公式 |
|------|------|
| FFN1 (Gate + Up) | $2 \times n_{bd1} \times t_p \times (2 \times bt \times bd1c_{pt} \times bse) = 4 \times bt \times H \times bse$ |
| FFN2 (Down) | $n_{bd2} \times t_p \times (2 \times bt \times bse \times bd2c_{pt}) = 2 \times bt \times bse \times H$ |
| **Per block** | $6 \times bt \times H \times bse$ |

#### 2.4.6 全部 SE FLOPs

$$\text{FLOPs}_{SE} = n_{bse} \times 6 \times bt \times H \times bse = 6 \times bt \times H \times I_{SE}$$

#### 2.4.7 SE 时间估算

$$T_{SE}^{HBM} = \frac{3 \times H \times I_{SE} \times B_w}{BW_{HBM}}$$

$$T_{SE}^{compute} = \frac{6 \times bt \times H \times I_{SE}}{F_{MXU}}$$

$$T_{SE} = \max(T_{SE}^{HBM}, T_{SE}^{compute})$$

> 对于 $I_{SE} = I$: SE 权重量 = 单个路由专家权重量。SE 的 AI 与路由专家相同 ($\approx \bar{n}_e$ for useful)，但 SE 的有效 token 数是 $bt$ (全部本地 token)，因此 $\text{AI}_{SE}^{useful} = 2 \times bt / B_w$。

#### 2.4.8 SE VMEM 占用

| 分配 | 公式 |
|------|------|
| `W1_SE` 双缓冲 | $2 \times t_p \times bd1_{pt} \times bse \times B_w$ |
| `W3_SE` 双缓冲 | 同上 |
| `W2_SE` 双缓冲 | $2 \times t_p \times bse \times bd2_{pt} \times B_w$ |
| SE 累加器 (F32) | $2 \times bt \times H \times 4$ |

$$\text{VMEM}_{SE} = 6 \times t_p \times bd_{pt} \times bse \times B_w + 8 \times bt \times H$$

> SE 与 expert FFN 共享 VMEM 空间（交错执行时），限制了 $bse$ 的大小。

#### 2.4.9 当前交错策略

当前实现将 SE 交错在 expert loop 内:

```
se_per_expert = max(2, ⌈n_bse / E_L⌉)
se_before = se_per_expert // 2
se_after = se_per_expert - se_before

for local_e_id in 0..E_L-1:
    for _ in range(se_before):
        run_shared_expert_slice(...)   // SE block
    wait_scatter_recv(expert_id)
    expert_ffn(expert_id)              // 路由专家 FFN
    start_gather(expert_id)
    for _ in range(se_after):
        run_shared_expert_slice(...)   // SE block
```

SE 在前 $\lceil E_L / (n_{bse} / se\_per\_expert) \rceil$ 个专家迭代中完成。

---

## 3. 全局调度与 Overlap 分析

### 3.1 依赖 DAG

```
                    tokens_input (HBM)
                   /                  \
                  v                    v
          [Stage 1]              [Stage 4: SE Compute]
          Gate/TopK/                    |
          AllReduce                     | (完全独立: 仅依赖 tokens)
              |                         |
              v                         |
          [Stage 2a]                    |
          A2A Scatter                   |
              |                         |
              v                         |
          [Stage 2b]                    |
          Expert FFN ──── (交错机会) ───┤
              |                         |
              v                         |
          [Stage 2c]                    |
          A2A Gather                    |
              |                         |
              v                         v
          [Stage 3: Output Accumulation]
          weighted_sum(expert_outs) + SE_output
              |
              v
            output (HBM)
```

**关键依赖链 (Critical Path)**:

$$\text{tokens} \to S1 \to S2a \to S2b \to S2c \to S3 \to \text{output}$$

**SE 可在 $\text{tokens}$ 可用后的任何位置执行**, 只需在 $S3$ 前完成。

### 3.2 合法 Overlap 枚举

| # | Overlap 对 | 数据依赖 | 合法性 | 资源竞争 |
|---|-----------|---------|--------|---------|
| 1 | SE ↔ S1 (routing) | 无 | **合法** | HBM BW (轻), VPU (routing vs SE act) |
| 2 | SE ↔ S2a (scatter) | 无 | **合法** | HBM BW, DMA engine |
| 3 | SE ↔ S2b (expert FFN) | 无 | **合法** | HBM BW (SE权重 vs expert权重), MXU, VMEM |
| 4 | SE ↔ S2c (gather) | 无 | **合法** | DMA engine, ICI BW |
| 5 | S2c[$e_i$] ↔ S2a[$e_{i+1}$] | scatter 不依赖前一 gather | **合法** | DMA engine |
| 6 | S2b[$e_{i+1}$] ↔ S2c[$e_i$] | expert FFN 仅依赖自身 scatter recv | **合法** | DMA engine |
| 7 | S3 ↔ S2b/S2c | S3 需要全部 gather 和 SE 完成 | **不合法** | — |

### 3.3 Tradeoff 1: SE Compute 与 Stage 1 的 Overlap

#### 当前状态

SE 交错在 S2b 的 expert loop 间隙——S1 期间 MXU 和 HBM BW 基本空闲。

#### 替代方案: SE 提前到 S1

在 routing 计算和 allreduce metadata 执行的同时，开始 SE FFN 计算。

**资源竞争分析**:

| 资源 | S1 消耗 | SE 消耗 | 竞争程度 |
|------|--------|--------|---------|
| MXU | 0 | 高 (matmul) | **无竞争** |
| VPU | 中 (routing mask) | 低 (activation) | 轻微 |
| HBM BW | 极低 (4-8 KB topk data) | 高 (SE 权重) | **无竞争** |
| ICI | 中 (allgather) | 0 | **无竞争** |
| VMEM | routing temporaries | SE 权重 + acc | **需检查容量** |

**结论**: S1 和 SE 消耗几乎完全不同的功能单元，**天然适合 overlap**。

**收益量化**:

$$\Delta T_{\text{saved}} = \min(T_{S1}, T_{SE}) - T_{\text{overhead}}^{\text{VMEM}}$$

$$T_{S1} \approx \lceil \log_2 P_{EP} \rceil \times (t_{barrier} + t_{ICI}) \approx 15\text{-}30 \mu s$$

$$T_{SE} = 3 \times H \times I_{SE} \times B_w / BW_{HBM} \approx 26 \mu s \quad \text{(Ling 2.6)}$$

可 overlap 量: ~15-26 μs。相比 expert loop ~1.8 ms 的关键路径，收益约 1-2%。

**核心价值**: 不在于绝对时间节省，而在于**将 SE 完全移出 expert loop**，使 expert loop 无间断执行——消除 SE 与 expert FFN 交错带来的 HBM BW 竞争 (当前前 4 个专家迭代增加约 25% BW 压力)。

**VMEM 约束**:

$$\text{VMEM}_{S1+SE} = \text{VMEM}_{S1} + \text{VMEM}_{SE} \leq \text{VMEM}$$

需验证 routing temporaries + SE 权重双缓冲 + SE 累加器不超过 64 MiB。

### 3.4 Tradeoff 2: AllToAll Combine + 下次 Dispatch 连续 DMA 发射

#### 问题建模

当前 expert loop 中，每个 expert FFN 完成后:
1. `start_a2a_gather(expert_i)` — 发射 gather DMA
2. (SE slice, 如有)
3. `wait_a2a_scatter_recv(expert_{i+1})` — 等待下一个 expert 的 scatter DMA

如果 gather DMA 和下一个 expert 的 scatter recv DMA 密集发射，DMA engine 可能出现 descriptor queue 拥塞。

#### DMA 管线模型

TPU DMA engine 参数:
- Descriptor queue depth: $D_{queue}$ (硬件限制)
- 每 DMA 启动开销: $t_{setup} \approx 200\text{-}500$ ns (descriptor 构建 + semaphore 操作)
- 并发 DMA 共享 BW

**Inflight DMA 分析**:

在 expert $e_i$ 到 $e_{i+1}$ 的转换点:

| DMA 类型 | 来源 | 方向 | 数量 |
|---------|------|------|------|
| Gather sends for $e_i$ | 本设备 → 各设备 | 发出 | ≤ $P_{EP}$ |
| Scatter recvs for $e_{i+1}$ | 各设备 → 本设备 | 接收 | ≤ $P_{EP}$ |
| (之前 expert 的残留 gather) | — | — | 可能 |

$$N_{inflight} = N_{gather\_sends} + N_{scatter\_recvs} \leq 2 \times P_{EP}$$

#### 延迟增加公式

$$\Delta T_{DMA} = \begin{cases}
0 & \text{if } N_{inflight} < D_{queue} \\
(N_{inflight} - D_{queue}) \times t_{setup} & \text{if } N_{inflight} \geq D_{queue} \\
\end{cases}$$

**Batch Scatter 路径的特殊性**: 所有 scatter DMA 在 Stage 2a 一次性发射完毕，expert loop 内不再有新的 scatter DMA。只有 gather DMA 在 expert loop 内发射。

$$\Delta T_{DMA}^{batch} \approx 0 \quad \text{(scatter 与 gather 不竞争 DMA engine)}$$

**Pipelined 路径**: Scatter 和 gather 交错发射，竞争更激烈。

$$\Delta T_{DMA}^{pipelined} \approx P_{EP} \times t_{setup} \quad \text{(每 expert 切换处)}$$

#### 量化 (Ling 2.6 decode)

- Batch scatter 路径，$P_{EP} = 4$
- 每 expert gather 量: $8 \times 16$ KB = 128 KB，分发到 4 设备
- 单次 DMA: ~128 KB / 4 = 32 KB per target，传输时间 $\approx 32$ KB / $200$ GB/s < 1 μs
- DMA setup: ~500 ns
- **净延迟增加**: $4 \times 500$ ns = **2 μs/expert**，总计 $64 \times 2 = 128$ μs

相比理论下界 1.8 ms，DMA chaining overhead 约 7%。

### 3.5 Tradeoff 3: 延迟 Combine (Deferred AllToAll Combine) + SE Overlap

#### 方案描述

不在每个 expert FFN 完成后立即发射 gather，而是等所有 $E_L$ 个 expert 计算完毕后**批量发射 gather**。

#### 收益

| 指标 | Pipelined Gather | Deferred Gather |
|------|-----------------|-----------------|
| Gather DMA 发起次数 | $E_L \times P_{EP}$ | $E_L \times P_{EP}$ (次数不变) |
| DMA setup 开销 | 分散在 expert loop 中 | 集中在 loop 后 |
| Expert loop 中的 DMA 干扰 | 有 (gather 与 expert 权重 DMA 竞争 HBM BW) | **无** |
| 额外 HBM scratch | 0 (buffer 可复用) | $E_L \times T_{a2a} \times H \times B_a$ |
| Gather 是否被 compute 覆盖 | 部分覆盖 (与下一 expert FFN overlap) | **不被覆盖** (全部暴露) |

**关键 tradeoff**:
- Pipelined: gather DMA 时间被后续 expert FFN 覆盖，但与 expert 权重 DMA 竞争 HBM BW
- Deferred: expert loop 无 gather 干扰，但 gather 时间完全暴露在关键路径上

#### Deferred 方案时间对比

$$T_{pipelined} = E_L \times (T_{SE\_slice} + T_{expert} + T_{gather\_start}) \approx E_L \times T_{expert}$$

(gather 大部分被 overlap 覆盖)

$$T_{deferred} = E_L \times T_{expert}' + T_{batch\_gather}$$

其中 $T_{expert}' < T_{expert}$（因为消除了 gather DMA 的 HBM BW 竞争）。

**净收益条件**: $E_L \times (T_{expert} - T_{expert}') > T_{batch\_gather}$

#### SE Overlap 窗口

如果 SE 未在 expert loop 中完成（例如 SE 提前到 S1 只完成一部分），deferred gather 期间是理想的 SE 完成窗口:

| 资源 | Batch Gather 消耗 | SE 消耗 | 竞争 |
|------|-------------------|---------|------|
| DMA engine | 高 (大量 gather DMA) | 低 (权重 DMA) | 中等 |
| ICI | 高 (跨设备传输) | 0 | **无** |
| MXU | 0 | 高 (matmul) | **无** |
| HBM BW | 中 (gather dest 写入) | 高 (SE 权重读取) | **有竞争** |

$$T_{deferred+SE} = E_L \times T_{expert}' + \max(T_{batch\_gather}, T_{SE\_remaining})$$

> 当 $T_{SE\_remaining} \leq T_{batch\_gather}$ 时，SE 完全被 gather 覆盖，实现零开销 SE。

### 3.6 Tradeoff 4: Shared Expert TP 切分

#### 当前状态

SE 权重在所有 $P_{EP}$ 设备上**完全复制**。每设备独立计算完整 SE FFN。

#### TP 切分方案

将 SE 的 $I_{SE}$ 维按 $P_{TP}$ 切分:
- 每设备 SE 权重: $3 \times H \times (I_{SE} / P_{TP}) \times B_w$
- FFN1: 每设备计算部分 gate/up，输出 shape $(bt, I_{SE}/P_{TP})$
- FFN2: down projection 后需要 **AllReduce** 聚合部分和

#### 收益

$$\Delta \text{HBM}_{save} = 3 \times H \times I_{SE} \times B_w \times (1 - 1/P_{TP})$$

$$\Delta T_{save}^{HBM} = \frac{3 \times H \times I_{SE} \times B_w \times (1 - 1/P_{TP})}{BW_{HBM}}$$

#### 代价

AllReduce 通信 (ring AllReduce over TP axis):

$$\Delta T_{comm} = \frac{2 \times (P_{TP} - 1)}{P_{TP}} \times \frac{bt \times H \times B_a}{BW_{ICI}^{ring}}$$

#### 收益条件

$$\Delta T_{save}^{HBM} > \Delta T_{comm}$$

$$\frac{3 \times H \times I_{SE} \times B_w \times (1 - 1/P_{TP})}{BW_{HBM}} > \frac{2 \times (P_{TP} - 1)}{P_{TP}} \times \frac{bt \times H \times B_a}{BW_{ICI}^{ring}}$$

简化 ($P_{TP} \gg 1$):

$$\frac{3 \times I_{SE} \times B_w}{BW_{HBM}} > \frac{2 \times bt \times B_a}{BW_{ICI}^{ring}}$$

$$\boxed{I_{SE} > \frac{2 \times bt \times B_a \times BW_{HBM}}{3 \times B_w \times BW_{ICI}^{ring}}}$$

代入 v7x 参数 (BF16, $BW_{ICI}^{ring} \approx 200$ GB/s):

$$I_{SE} > \frac{2 \times bt \times 2 \times 3690}{3 \times 2 \times 200} = \frac{14760 \times bt}{1200} = 12.3 \times bt$$

- $bt = 64$: 需 $I_{SE} > 787$ — **Ling 2.6 ($I_{SE} = 2048$) 满足**
- $bt = 32$: 需 $I_{SE} > 394$ — 大部分配置满足

> **结论**: 对于 $I_{SE} \geq 1024$ 的模型，SE TP 切分几乎总是有收益。但需权衡:
> 1. AllReduce 增加的延迟是否在关键路径上（如果 SE 与其他阶段 overlap，AllReduce 可能不在关键路径）
> 2. 实现复杂度增加

### 3.7 Tradeoff 5: `o_proj` 与 DP Attention 的 Overlap

#### 上下文

MoE block 输出后的数据流:

```
MoE output (EP sharded) → o_proj → 重新组合为 DP sharding → Attention
```

`o_proj` 是 output projection: $[T_L, H] \times [H, H_{out}]$

#### 依赖分析

| 阶段 | 输入 sharding | 输出 sharding | 操作 |
|------|-------------|-------------|------|
| MoE output | EP (=DP×TP) | EP | 本文分析的 kernel |
| `o_proj` | EP → TP | TP | matmul, 可能需要 TP reduce |
| DP 重组 | TP → DP | DP | AllGather 或 ReduceScatter |
| Attention | DP | DP | 后续 layer |

#### Overlap 机会

**如果 `o_proj` 按 TP 切分** ($H_{out}$ 维分片):
1. 每设备计算部分 `o_proj`: $[T_L, H] \times [H, H_{out}/P_{TP}]$
2. ReduceScatter over TP: 聚合部分和并分布到 DP 维度
3. AllGather over DP: 收集完整 DP batch

**Pipeline**: `o_proj` 的 tile 级别 compute 可与 ReduceScatter 的 DMA overlap:

$$T_{o\_proj+reorg} = \max(T_{o\_proj}^{compute}, T_{ReduceScatter}, T_{AllGather})$$

而非串行: $T_{o\_proj} + T_{RS} + T_{AG}$

#### 量化

$$T_{o\_proj}^{compute} = \frac{2 \times T_L \times H \times H_{out}}{F_{MXU}}$$

$$T_{RS} = \frac{(P_{TP} - 1)}{P_{TP}} \times \frac{T_L \times H_{out} \times B_a}{BW_{ICI}^{ring}}$$

> **注意**: 这超出了 fused MoE kernel 的边界。分析给出接口约束: kernel 输出 shape 和 sharding 需与 `o_proj` 的 pipeline 策略对齐。

### 3.8 Tradeoff 6: Prefill vs Decode 最优 Expert 写法

#### 关键变量

per-expert 的 Arithmetic Intensity 决定了 regime:

$$\text{AI}_{useful} = \frac{6 \times n_e \times H \times I}{3 \times H \times I \times B_w + 2 \times n_e \times H \times B_a} \approx \frac{2 \times n_e}{B_w} \quad \text{(when weights dominate)}$$

#### Crossover 分析

令 $T_{HBM}(n_e) = T_{compute}(n_e)$:

$$\frac{3HIB_w + 2 \times n_e \times H \times B_a}{BW_{HBM}} = \frac{6 \times n_e \times HI}{F_{MXU}}$$

$$3HIB_w \times F_{MXU} + 2 \times n_e \times H \times B_a \times F_{MXU} = 6 \times n_e \times HI \times BW_{HBM}$$

$$3IB_w \times F_{MXU} = n_e \times (6I \times BW_{HBM} - 2B_a \times F_{MXU})$$

$$\boxed{n_e^{*} = \frac{3 \times I \times B_w \times F_{MXU}}{6 \times I \times BW_{HBM} - 2 \times B_a \times F_{MXU}}}$$

> 当 $2 \times B_a \times F_{MXU} \ll 6 \times I \times BW_{HBM}$ (通常成立):
> $$n_e^{*} \approx \frac{B_w \times F_{MXU}}{2 \times BW_{HBM}} = \frac{B_w \times \text{Ridge}}{2}$$

#### 各配置 Crossover 值

| 配置 | $B_w$ | $n_e^{*}$ | 对应全局 $T$ (EP=4, $E_L$=64) |
|------|--------|---------|------|
| v7x BF16 | 2 | $2 \times 313 / 2 = 313$ | $313 \times 64 / 8 \times 4 = 10,016$ |
| v7x FP8 | 1 | $1 \times 313 / 2 = 157$ | $157 \times 64 / 8 \times 4 = 5,024$ |
| v6e BF16 | 2 | $B_w \times 918T / (2 \times 1600G) = 574$ | ~18,400 |

#### 最优策略矩阵

| 维度 | Decode ($n_e \ll n_e^*$) | Prefill ($n_e \gg n_e^*$) |
|------|--------------------------|---------------------------|
| **Bottleneck** | HBM BW | Compute (MXU) |
| **优化目标** | 减少权重加载 (减小 $3HIB_w$) | 增大 GEMM M 维 (增大 $n_e$) |
| **方案选择** | A/B 等价 ($n_{bt}=1$) | B (token-stationary) 更优 |
| **量化收益** | 显著 (HBM 减半 → 时间减半) | 有限 (非瓶颈) |
| **btc 选择** | 小 btc 减少 padding waste | 大 btc 最大化 MXU 利用 |
| **SE TP 切分** | 可能有收益 | 收益更大 (SE 也 compute-bound) |
| **Expert 循环优化** | 减少 DMA setup/barrier overhead | 增大 GEMM tile, 优化 MXU pipeline |
| **Weight-streaming vs Tiling** | Weight-streaming (逐 expert 流式权重) | Tile both dims (token + weight 均切分) |

---

## 4. Regime Crossover 深度分析

### 4.1 Per-Expert 时间曲线

$$T_{expert}(n_e) = \max\left(\underbrace{\frac{3HIB_w}{BW_{HBM}}}_{\text{权重 DMA (常数)}}, \underbrace{\frac{6 \times n_e \times HI}{F_{MXU}}}_{\text{计算 (线性增长)}}\right) + \underbrace{\frac{2 \times n_e \times H \times B_a}{BW_{HBM}}}_{\text{token DMA (线性, 通常可覆盖)}}$$

```
时间
  │
  │                          ╱ T_compute (线性增长)
  │                        ╱
  │                      ╱
  │ ─────────────────╱──── T_HBM (权重, 常数)
  │                ╱ │
  │              ╱   │
  │            ╱     │
  │          ╱       │
  │        ╱         │
  │      ╱           │
  │    ╱             │
  │  ╱               │
  │╱                 │
  ├──────────────────┼─────────────→ n_e
  0               n_e* ≈ 313 (BF16, v7x)
      ← HBM-bound →│← Compute-bound →
```

### 4.2 方案 A vs B 在不同 $n_e$ 下的对比

#### 方案 A (Weight-streaming) 全 bt-tile 时间

$$T_A = n_{bt} \times \left(\sum_{e=0}^{E_L-1} T_{expert}^A(n_e) + T_{barrier}\right)$$

其中 $T_{expert}^A$ 包含权重 DMA + compute + gather start。

当 $n_{bt} > 1$ 时，权重重复加载:

$$\text{HBM}_A^{weight} = n_{bt} \times E_{active} \times 3HIB_w$$

#### 方案 B (Token-stationary) 全 bt-tile 时间

$$T_B = \sum_{e=0}^{E_L-1} T_{expert}^B(n_e) + T_{permute}$$

其中 $T_{expert}^B$ 的权重 DMA 只发生一次，但 token 需要 $n_e$ 的 scatter/permute:

$$\text{HBM}_B^{weight} = E_{active} \times 3HIB_w \quad \text{(与 } n_{bt} \text{ 无关)}$$

#### 交叉条件

$$T_A > T_B \iff n_{bt} \times E_{active} \times 3HIB_w > E_{active} \times 3HIB_w + T_{permute\_overhead}$$

$$(n_{bt} - 1) \times E_{active} \times 3HIB_w > T_{permute\_overhead}$$

当 $n_{bt} = 1$ (当前 Ling 2.6 decode): $T_A = T_B$（方案 A 更优因控制流更简单）。

当 $n_{bt} > 1$: 方案 B 优势 = $(n_{bt} - 1) \times E_{active} \times 3HIB_w / BW_{HBM}$ 减去 permute overhead。

### 4.3 VMEM 对 Token-Stationary 的限制

方案 B 需要将所有 token 常驻 VMEM:

$$T_{max}^B = \left\lfloor \frac{\text{VMEM} - W_{buf}}{H \times B_a + A_{pt}} \right\rfloor$$

其中:
- $W_{buf}$: 权重双缓冲大小 (FFN1 或 FFN2 中较大者)
- $A_{pt}$: 每 token 累加器大小 (F32)

FFN1 约束 (binding constraint):

$$T_{max}^{FFN1} = \left\lfloor \frac{\text{VMEM} - 2 \times (2 \times t_p \times bd1_{pt} \times bf \times B_w)}{H \times B_a + 2 \times bf \times 4} \right\rfloor$$

> 对 Ling 2.6: $T_{max} = (64M - 16M) / (16K + 8K) = 2,048$ tokens。

### 4.4 Chunked Prefill 考量

当 prefill 的 $T_L > T_{max}^B$ 时，需要分 chunk 处理:

$$n_{chunks} = \lceil T_L / T_{max}^B \rceil$$

每 chunk 权重加载一次:

$$\text{HBM}_{chunked}^{weight} = n_{chunks} \times E_{active} \times 3HIB_w$$

对比方案 A ($n_{bt}$ 由 $bt$ 决定) 和方案 B (chunk 由 VMEM 决定):

- 方案 A: $n_{bt} = T_L / bt$, 每轮加载 $E_{active}$ 个专家权重
- 方案 B: $n_{chunks} = T_L / T_{max}^B$, 每轮加载 $E_{active}$ 个专家权重
- 方案 B 优势: $T_{max}^B \gg bt$（VMEM 限制远大于 tile 大小），所以 $n_{chunks} \ll n_{bt}$

---

## 5. 数值验证

### 5.1 Ling 2.6 Decode 实例化

#### bt=64 Config (旧分析参考)

| 参数 | 值 |
|------|------|
| $N_E = 256$, $H = 8192$, $I = 2048$, $I_{SE} = 2048$ | |
| $k = 8$, $T = 256$, $B_w = 2$ (BF16), $B_a = 2$ | |
| $P_{EP} = 4$ (2×2×1 mesh), $E_L = 64$, $T_L = 64$ | |
| $bt = 64$, $bts = 64$, $btc = 64$, $bf = 1024$ | (bt=64 config) |
| $bd1 = 2048$, $bd2 = 2048$, $bse = 256$ | |
| $n_{bt} = 1$, $n_{bf} = 2$, $n_{bd1} = 4$, $n_{bd2} = 4$, $n_{bse} = 8$ | |
| $\bar{n}_e = 64 \times 8 / 64 = 8$ tokens/expert | |
| $t_p = 2$, $h_{pt} = 4096$, $bd1_{pt} = 1024$, $bd2_{pt} = 1024$ | |

#### bt=32 Config (最新 LLO dump 验证)

| 参数 | 值 |
|------|------|
| $N_E = 256$, $H = 8192$, $I = 2048$, $I_{SE} = 2048$ | |
| $k = 8$, $T = 256$, $B_w = 2$ (BF16), $B_a = 2$ | |
| $P_{EP} = 4$ (2×2×1 mesh), $E_L = 64$, $T_L = 64$ | |
| $bt = 32$, $bts = 32$, $btc = 32$, $bf = 512$ | |
| $bd1 = 1024$, $bd2 = 1024$, $bse = 512$ | |
| $n_{bt} = 2$, $n_{bf} = 4$, $n_{bd1} = 8$, $n_{bd2} = 8$, $n_{bse} = 4$ | |
| $\bar{n}_e = 64 \times 8 / 64 = 8$ tokens/expert | |
| $t_p = 2$, $h_{pt} = 4096$, $bd1_{pt} = 512$, $bd2_{pt} = 512$ | |

> **LLO 验证**: bt=32 config 的 LLO 产出 24,106 post-delay bundles (详见附录 B)，相比 bt=64 config 有更小的 tile size 和更多的循环轮数 ($n_{bt}=2$ vs $1$, $n_{bf}=4$ vs $2$)，通过更小的 tile 换取更高的 MXU 与 DMA 并行度。

#### Stage-by-Stage 代入

| 阶段 | HBM Read | HBM Write | MXU FLOPS | ICI Comm | 时间估算 |
|------|----------|-----------|-----------|----------|---------|
| S1: Routing + AllReduce | 4 KB | 0 | 0 | ~4 KB | ~15 μs |
| S2a: A2A Scatter | 1 MB | 8 MB | 0 | 6 MB | ~40 μs |
| S2b: Expert FFN (×64) | 6,144 MB | 256 MB | 412 GFLOPS | 0 | **1,800 μs** |
| S2c: A2A Gather | — | 8 MB | 0 | 6 MB | (overlapped) |
| S3: Output Acc | 8 MB | 1 MB | 0 | 0 | ~10 μs |
| S4: SE FFN (×8 blocks) | 96 MB | — | 6.4 GFLOPS | 0 | (overlapped) |
| **Total** | **~6,249 MB** | **~273 MB** | **~418 GFLOPS** | **~12 MB** | **~1,865 μs** |

**验证**: 与现有分析文档 Section 4.1 的 ~1,910 μs 和理论下界 ~1.8 ms 一致 (±5%)。

#### 关键指标验证

| 指标 | 公式计算 | 现有分析值 | 匹配 |
|------|---------|-----------|------|
| 单专家权重 HBM | $3 \times 8192 \times 2048 \times 2 = 96$ MB | 96 MB | ✓ |
| 单专家 MXU FLOPs (exec) | $6 \times 64 \times 8192 \times 2048 = 6.44$ G | 6.44 GFLOPS | ✓ |
| AI (exec) | $6.44G / 96M = 67.1$ | 67.1 | ✓ |
| AI (useful) | $6.44G \times 8/64 / 96M = 8.4$ | 8.4 | ✓ |
| Crossover $n_e^*$ | $2 \times 313 / 2 = 313$ | — (新分析) | — |
| $T_{max}^B$ | $(64M - 16M) / 24K = 2,048$ | 2,048 | ✓ |

### 5.2 参数敏感性

#### 变化 $P_{EP}$ (固定 $T = 256$, 其他参数不变)

| $P_{EP}$ | $E_L$ | $T_L$ | $\bar{n}_e$ | 权重 HBM (total) | $T_{HBM}$ | AI (useful) |
|----------|-------|-------|-------------|-----------------|-----------|-------------|
| 1 | 256 | 256 | 8 | 24,576 MB | 6.7 ms | 8 |
| 2 | 128 | 128 | 8 | 12,288 MB | 3.3 ms | 8 |
| 4 | 64 | 64 | 8 | 6,144 MB | 1.7 ms | 8 |
| 8 | 32 | 32 | 8 | 3,072 MB | 0.8 ms | 8 |
| 16 | 16 | 16 | 8 | 1,536 MB | 0.4 ms | 8 |
| 32 | 8 | 8 | 8 | 768 MB | 0.2 ms | 8 |
| 64 | 4 | 4 | 8 | 384 MB | 0.1 ms | 8 |

> $\bar{n}_e$ 不随 $P_{EP}$ 变化 ($= T \times k / N_E = 8$, 与 EP 无关)。但每设备 HBM 线性减少，理论下界线性降低。

#### 变化 $T$ (固定 $P_{EP} = 4$, 其他参数不变)

| $T$ | $T_L$ | $\bar{n}_e$ | $n_{bt}$ (bt=64) | 权重 HBM | AI (useful) | Regime |
|-----|-------|-------------|-----------------|----------|-------------|--------|
| 32 | 8 | 1 | 1 | 6,144 MB | 1 | 极度 HBM-bound |
| 256 | 64 | 8 | 1 | 6,144 MB | 8 | 深度 HBM-bound |
| 2,048 | 512 | 64 | 8 | 49,152 MB | 64 | HBM-bound |
| 10,016 | 2,504 | 313 | 40 | 245,760 MB | 313 | **Crossover** |
| 16,384 | 4,096 | 512 | 64 | 393,216 MB | 512 | Compute-bound |
| 65,536 | 16,384 | 2,048 | 256 | — | 2,048 | Compute-bound |

> 注意: $T = 2,048$ 时使用方案 A 的 $n_{bt} = 8$，权重被加载 8 遍。方案 B 最多只需 $\lceil 512/2048 \rceil = 1$ chunk。

### 5.3 理论 vs 实测 Gap 归因

| 指标 | 值 |
|------|------|
| 理论下界 | ~1.8 ms |
| 实测时间 | 149 ms (`ep_size`=8) / 150 ms (`ep_size`=8, benchmark) |
| **Gap** | **~83x** |
| 有效 HBM BW | $6,249 \text{ MB} / 149 \text{ ms} = 42$ GB/s (峰值 1.1%) |

**Gap 归因 (来自 LLO 分析, bt=32 config)**:

1. **Per-expert DMA setup overhead** (~50-60% of gap)
   - 每专家多次 DMA 发起 × 专家循环迭代
   - 每次 ~500 ns setup → 固定开销累积
   - DMA engine 无法充分流水线化小块传输

2. **SALU 循环控制 + MEM/VPU 混合 overhead** (~25-35% of gap)
   - 27.4% VLIW bundle 为 SALU-only (6,607/24,106)
   - 32.8% 为 MEM+VPU 混合 bundle (7,900/24,106)
   - `lax.fori_loop` 的循环控制、地址计算、条件分支
   - 24,106 post-delay bundles (pre-delay: 31,876, delay converter 消除 7,770 空 bundle)

3. **Sync barrier** (~5-10% of gap)
   - 每 bt-tile 至少 $\lceil \log_2 P_{EP} \rceil + 2$ 次 barrier
   - 每次 ~3-5 μs
   - Pipelined 路径更多 barrier

4. **小 DMA 传输效率低** (~5-10% of gap)
   - Token tile 较小, routing data 数 KB
   - DMA engine 对小块传输的效率远低于大块

---

## 附录 A: 硬件参数

| 参数 | TPU v7x | TPU v6e |
|------|---------|---------|
| HBM 容量 | 96 GB | 32 GB |
| HBM 带宽 | 3,690 GB/s | 1,600 GB/s |
| MXU 峰值 (BF16, per TensorCore) | 1,154 TFLOPS | 918 TFLOPS |
| MXU 数量 | 2 (dual) | 1 |
| MXU 维度 | 256×256 | 256×256 |
| VMEM | 64 MiB | 64 MiB |
| Ridge Point (BF16) | 313 FLOPs/byte | 574 FLOPs/byte |
| ICI 带宽 (per link) | ~200 GB/s | ~100 GB/s |

## 附录 B: LLO Overhead 观测汇总

> 最新 dump: `fused-moe-k_8-bt_32_32_32-bf_512_512-bd1_1024_1024-bd2_1024_1024-shared_expert_bse_512`, 2026-05

| 观测 | 数值 | 来源 | 备注 |
|------|------|------|------|
| 总 VLIW bundle (post-delay) | 24,106 | LLO schedule analysis final | pre-delay: 31,876, delay converter 消除 7,770 |
| 空 bundle | 868 (3.6%) | schedule analysis | |
| SALU-only bundle | 6,607 (27.4%) | per-bundle utilization | 含 `sst`/`sld` 标量存储/加载 |
| MXU active bundle | 3,005 (12.5%) | per-bundle utilization | 含 co-issue 与纯 MXU |
| Dual MXU bundle | 2,659 (11.0%) | per-bundle utilization | 两路 MXU 均活跃 |
| Single MXU bundle | 346 (1.4%) | per-bundle utilization | 仅一路 MXU 活跃 |
| MXU + MEM co-issue | 2,245 (9.3%) | per-bundle utilization | 权重 DMA 与 matmul 并行 |
| MXU + VPU co-issue | 1,897 (7.9%) | per-bundle utilization | activation 与 matmul 并行 |
| MEM-only bundle | 2,779 (11.5%) | per-bundle utilization | 纯数据搬运 |
| MEM+VPU mixed bundle | 7,900 (32.8%) | per-bundle utilization | DMA 与向量运算并行 |
| MXU 槽位利用率 | 5,664 / 48,212 (11.7%) | per-bundle utilization | |
| VLOAD:FILL (weight staging) | 2,576 slots | per-bundle utilization | 权重从 HBM staging 到 VMEM |
| VSTORE:SPILL (result writeback) | 2,458 slots | per-bundle utilization | 结果从 VMEM 写回 |
| VMEM 分配总量 | 20.7 MB (32.3%) | allocation analysis | allocation8: 4 MB scatter buffer; allocation12-21: 2 MB×7 weight tile buffers |
| Critical path (FFN1 matmul) | 748 | region4355/4377 | vld→vmul→vst 链， members=2,848~2,976 |
| Critical path (scatter) | 450 | region4322 | scatter DMA + barrier 链, members=854 |
| Physical VPR 峰值 (post-RA) | 372 | register pressure dump | 物理寄存器并发数，predicate regs=17 |
| VPR spill (physical) | 0 | register pressure analysis | 无物理寄存器溢出 |
| TLP wrapper bundles | 356 (fused-moe=7) | TLP schedule analysis | TLP wrapper overhead 极小 |
