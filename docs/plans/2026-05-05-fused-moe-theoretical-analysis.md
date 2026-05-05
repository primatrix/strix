# Fused MoE Kernel — 系统性理论分析

> 面向 TPU v7x 的 fused expert-parallel MoE 性能分析的通用参数化框架。
> 覆盖 decode 和 prefill 场景，支持任意 DP/EP/TP 配置。
> 基于并泛化了 `primatrix/wiki/` 中 Ling 2.6 decode 的案例分析。

---

## 第一部分：系统模型

### 1.1 硬件参数 (TPU v7x)

| 符号 | 值 | 说明 |
|--------|-------|-------------|
| $\Phi_{\text{MXU}}$ | 2,307 TFLOPS | BF16 MXU 峰值吞吐 (双 MXU) |
| $\Phi_{\text{VPU}}$ | — | VPU 吞吐 (标量，对本 kernel 不构成瓶颈) |
| $\Theta_{\text{HBM}}$ | 3,690 GB/s | HBM 峰值带宽 |
| $\Theta_{\text{ICI}}$ | ~300 GB/s (每链路) | 芯片间互联带宽 |
| $M_{\text{VMEM}}$ | 64 MiB | 每个 TensorCore 的 VMEM 容量 |
| $R_{\text{VPR}}$ | ~32K | VPR 寄存器堆容量 |
| v7x 拓扑 | 1 chip = 2 TensorCores | 每芯片 `tpuv7x=2` 个 device |

### 1.2 模型与数据参数

| 符号 | 说明 | Ling 2.6 Decode |
|--------|-------------|-----------------|
| $B$ | 全局 batch size (tokens) | 256 |
| $E$ | 路由专家总数 | 256 |
| $k$ | Top-K 路由 | 8 |
| $H$ | 隐藏层维度 | 8192 |
| $I$ | 专家 FFN 中间维度 | 2048 |
| $I_{\text{se}}$ | 共享专家中间维度 | 2048 |
| $\tau$ | 数据类型字节数 (BF16=2, FP8=1) | 2 |
| $\tau_{\text{acc}}$ | 累加器类型字节数 (F32=4) | 4 |

### 1.3 并行参数

Kernel 使用二维 mesh `("data", "tensor")`：

```
Mesh: devices.reshape(1, ep_size) → axis_names=("data", "tensor")
```

| 符号 | 公式 | 说明 |
|--------|---------|-------------|
| $DP$ | — | 数据并行副本数 (沿 "data" 轴) |
| $EP$ | $\text{ep\_size}$ | 专家并行分片数 (沿 "tensor" 轴) |
| $TP$ | — | 专家内张量并行 (当前 kernel 未使用) |
| $N_{\text{dev}}$ | $DP \times EP$ | Mesh 中总设备数 |

当前 kernel：$DP=1$，$EP=8$（部分配置 $EP=4$），mesh = `(1, EP)`。

**核心洞察**：DP 和 EP 作用于正交维度：
- **DP**：沿 *token batch* 维度切分。每个 DP 副本看到 $B/DP$ 个 token，独立路由，MoE 部分无跨副本通信。
- **EP**：沿 *expert* 维度切分。Token 需通过 all-to-all 发送到持有目标专家的设备。

### 1.4 派生量（每设备）

| 符号 | 公式 | Ling 2.6 (EP=8) | Ling 2.6 (EP=4) |
|--------|---------|-----------------|-----------------|
| $T_{\text{local}}$ | $\lceil B / DP \rceil$ | 256 | 256 |
| $E_{\text{local}}$ | $E / EP$ | 32 | 64 |
| $Q_{\text{pairs}}$ | $T_{\text{local}} \times k$ | 2048 | 2048 |
| $\bar{n}_e$ | $Q_{\text{pairs}} / E_{\text{local}}$ | 64 | 32 |
| $\bar{n}_e^{\text{global}}$ | $(B \times k) / E$ | 8 | 8 |
| $P_{\text{local}}$ | $E_{\text{local}} / E$ | $1/EP$ | $1/EP$ |

**关键关系**：$\bar{n}_e$（每设备每个本地专家的平均 token 数）= $B \cdot k \cdot EP / (DP \cdot E)$。

对 decode 场景 $B=256, k=8, E=256$：$\bar{n}_e = 256 \times 8 / 256 = 8$（全局平均，与 EP/DP 无关）。但 $E_{\text{local}}$ 随 $E/EP$ 缩放，因此每设备的专家处理总量 $\propto 1/EP$，而每专家 token 数保持为 8。

### 1.5 Block Config 参数

| 符号 | 含义 | 典型范围 |
|--------|---------|---------------|
| `bt` | 外层 token tile（路由/输出维度） | $\min(\text{bt\_cfg}, T_{\text{local}})$，整除 $T_{\text{local}}$ |
| `bts` | `expert_ffn` 内的 token staging tile | $\leq bt \times EP$ |
| `btc` | 计算 tile M 维度 | $\leq bts$，整除 $bts$ |
| `bf` | 中间维度 tile (I/O dim) | 整除 $I$ |
| `bfc` | 中间维度计算 tile | $\leq bf$ |
| `bd1` | FFN1 K 维度 tile | 整除 $H$ |
| `bd1c` | FFN1 计算 tile | $\leq bd1$ |
| `bd2` | FFN2 N 维度 tile | 整除 $H$ |
| `bd2c` | FFN2 计算 tile | $\leq bd2$ |
| `bse` | 共享专家中间维度 tile | 整除 $I_{\text{se}}$ |
| $p$ | `t_packing` = dtype 打包因子 | 2 (BF16), 4 (FP8) |

### 1.6 阶段划分与 Kernel 执行映射

用户提出的四阶段划分在 kernel 执行中的对应关系：

```
Stage 1: gate, topk, allreduce metadata, permute
         → `wait_fetch_topk` + `t2e_routing` 计算 + `all_reduce_metadata`

Stage 2: alltoall dispatch, expert compute, alltoall combine
         → `start_a2a_scatter` + `expert_ffn` 循环 (`wait_scatter` + ffn + `start_gather`)

Stage 3: 共享专家与 MoE 输出聚合
         → `run_shared_expert_slice` (与 Stage 2 交错) + `acc_and_store_output`

Stage 4: 共享专家计算 (与 Stage 1-2 数据独立)
         → 共享专家 FFN (gate+up+激活+down)，当前实现与 Stage 2 交错
```

**数据依赖 DAG**：

```
  Tokens (HBM)
      │
      ├──→ [S1] Gate + TopK ──→ allreduce metadata ──→ permute/路由表
      │                                                        │
      │                                                        ▼
      │                                                   [S2] A2A Scatter
      │                                                        │
      │                                                        ▼
      │                                              [S2] Expert FFN × E_local
      │                                                        │
      │                                                        ▼
      │                                              [S2] A2A Combine
      │                                                        │
      │                                                        ▼
      │                                              [S3] 输出聚合
      │
      └──→ [S4] 共享专家 FFN ──────────────────────────────→ [S3] (加 SE 输出)
                                                                   │
                                                                   ▼
                                                              Output (HBM)

S4 与 S1-S2 数据无关。S3 同时依赖 S2 和 S4。
```

---

## 第二部分：逐阶段理论公式

### 2.1 Stage 1: Gate、TopK、AllReduce Metadata、Permute

#### 2.1.1 Gate Logits（如非预计算）

若需在设备上计算 gate logits（通常在上游独立算子完成）：

$$\text{FLOPs}_{\text{gate}} = T_{\text{local}} \times H \times E$$

实际中 gate logits 由上游计算；fused MoE kernel 接收已计算好的 `topk_ids` 和 `topk_weights`。

#### 2.1.2 TopK 数据读取

**HBM 读取**：

$$HBM_{\text{read}}^{S1} = bt \times k \times (\tau_{\text{f32}} + \tau_{\text{s32}}) = bt \times k \times (4 + 4) = 8 \cdot bt \cdot k \ \text{bytes}$$

每个 bt tile：`topk_weights` (F32) + `topk_ids` (S32)，各 shape `[bt, k]`。

#### 2.1.3 路由表构造

计算 `t2e_routing` 和 `expert_sizes`：

```
For each token t in [0, bt):
  For each expert rank r in [0, k):
    e_id = topk_ids[t, r]
    t2e_routing[t, r] = e_id
    expert_sizes[0, e_id] += 1
```

**VPU FLOPs**：$\approx bt \times k \times E$ 次比较操作（广播等值检查），加 $bt \times k$ 次归约。与 MXU 操作相比可忽略（~$10^5$ ops）。

**LLO 观测**：`vrot.slane`、`vbcast.lane`、`vpop`、`vselect` 链 — 标量/向量操作，不使用 MXU。

#### 2.1.4 AllReduce Metadata

跨所有 EP 设备 allgather 各设备的 `expert_sizes`（shape `[1, E]` S32）。

**每设备通信量**：

$$V_{\text{comm}}^{S1} = E \times \tau_{\text{s32}} = E \times 4 \ \text{bytes}$$

使用递归倍增（2 的幂次 EP）：$\lceil \log_2 EP \rceil$ 轮。

对 $E=256, EP=8$：$V_{\text{comm}} = 1 \text{ KB}$，3 轮。

**延迟模型**：

$$T_{S1}^{\text{comm}} \approx \lceil \log_2 EP \rceil \times (t_{\text{latency}} + V_{\text{comm}} / \Theta_{\text{ICI}}) + t_{\text{barrier}}$$

其中 $t_{\text{latency}} \approx 2\text{-}5\ \mu\text{s}$（ICI 链路建立），$t_{\text{barrier}} \approx 5\text{-}10\ \mu\text{s}$。

当前配置：$T_{S1}^{\text{comm}} \approx 10\text{-}20\ \mu\text{s}$。

#### 2.1.5 Permute

收到全局 `expert_sizes` 后，通过前缀和计算 `expert_starts`——纯 VMEM/SMEM 标量操作，耗时可忽略。

#### 2.1.6 Stage 1 汇总

| 指标 | 公式 | 值 (Ling 2.6, EP=8) |
|--------|---------|------------------------|
| HBM 读取 | $8 \cdot bt \cdot k$ | 4 KB |
| HBM 写入 | 0 | 0 |
| VPU FLOPs | $\sim bt \cdot k \cdot E$ | ~131K |
| ICI 通信 | $E \cdot 4 \cdot \lceil \log_2 EP \rceil$ (总量) | ~3 KB |
| 瓶颈 | 通信延迟 | ~10-20 μs |

### 2.2 Stage 2: All-to-All Dispatch + Expert Compute + All-to-All Combine

#### 2.2.1 All-to-All Dispatch (Scatter)

每个 token 被复制到其 top-k 专家所在设备。每设备 scatter 量：

$$V_{\text{scatter}} = T_{\text{local}} \times k \times H \times \tau$$

本地与远程流量比例：

$$f_{\text{local}} = \frac{E_{\text{local}}}{E} = \frac{1}{EP}$$

$$f_{\text{remote}} = 1 - \frac{1}{EP}$$

**每设备 ICI 出站**：

$$V_{\text{ICI-out}}^{S2\text{-disp}} = T_{\text{local}} \times k \times H \times \tau \times (1 - \frac{1}{EP})$$

**每设备 ICI 入站**（对称）：

$$V_{\text{ICI-in}}^{S2\text{-disp}} = V_{\text{ICI-out}}^{S2\text{-disp}}$$

**HBM 流量（本地 DMA）**：

$$V_{\text{DMA-local}}^{S2\text{-disp}} = T_{\text{local}} \times k \times H \times \tau \times \frac{1}{EP}$$

**HBM 写入（scratch buffer）**：

$$V_{\text{HBM-write}}^{S2\text{-disp}} = T_{\text{local}} \times k \times H \times \tau$$

Token 被复制到 `a2a_s_hbm` scratch buffer，组织为 `expert_buffer_count` 个 slot。

**延迟模型**：

$$T_{S2}^{\text{scatter}} \approx \max\left( \frac{V_{\text{DMA-local}}}{\Theta_{\text{HBM}}}, \frac{V_{\text{ICI-out}}}{\Theta_{\text{ICI}}} \right) + N_{\text{tokens}} \times t_{\text{DMA-setup}}$$

Ling 2.6 decode (EP=8)：$V_{\text{ICI-out}} = 256 \times 8 \times 8192 \times 2 \times 7/8 = 28.7 \text{ MB}$，$T_{S2}^{\text{scatter}} \approx 40\text{-}60\ \mu\text{s}$。

**EP=1（无 all-to-all）时**：$V_{\text{ICI-out}} = 0$，scatter 退化为纯本地 DMA。

#### 2.2.2 Expert FFN 计算

对每个本地专家 $e$，有 $n_e$ 个 token（动态，路由相关）：

**每专家权重加载量**：

Gate (W1)：$H \times I \times \tau$ bytes
Up (W3)：$H \times I \times \tau$ bytes
Down (W2)：$I \times H \times \tau$ bytes

$$W_{\text{per-expert}} = (2H \cdot I + I \cdot H) \times \tau = 3 \cdot H \cdot I \cdot \tau$$

**Tiling 因子**（权重被加载的次数）：

在 weight-streaming（方案 A）方法中，权重按 `(bf, bd1/bd2)` tile 加载，`btc` 为内部 M 维度。

$$n_{\text{bd1}} = H / bd1,\quad n_{\text{bd2}} = H / bd2,\quad n_{\text{bf}} = I / bf$$

每个专家 FFN 处理 $n_e$ 个 token：

**FFN1 (Gate + Up)**：对每个 `(bf_id, bd1_id)` tile，加载 W1/W3 tile 和 token tile，执行两次 MXU matmul。

$$\text{FLOPs}_{\text{FFN1}} = 4 \cdot n_{\text{bf}} \cdot n_{\text{bd1}} \cdot btc \cdot \frac{bd1}{p} \cdot bf$$

当 $btc = bd1/p = bf$（block config 常见情况）且 $n_{\text{bf}} \cdot bf = I$、$n_{\text{bd1}} \cdot bd1 = H$ 时：

$$\boxed{\text{FLOPs}_{\text{FFN1}} = 4 \cdot btc \cdot H \cdot I}$$

**FFN2 (Down)**：对每个 `(bf_id, bd2_id)` tile，加载 W2 tile 和激活值，执行一次 MXU matmul。

$$\boxed{\text{FLOPs}_{\text{FFN2}} = 2 \cdot btc \cdot H \cdot I}$$

$btc=64$（全 tile，含 padding）：$\text{FLOPs}_{\text{FFN1}} = 4 \times 64 \times 8192 \times 2048 = 4.29 \text{ GFLOPS}$，$\text{FLOPs}_{\text{FFN2}} = 2.15 \text{ GFLOPS}$。

**每专家有效 FLOPs**（扣除 padding）：

$$\boxed{\text{FLOPs}_{\text{useful}} = \text{FLOPs}_{\text{executed}} \times \frac{\bar{n}_e}{btc}}$$

Decode 场景 $\bar{n}_e = 8$：有效 FLOPs 仅为执行量的 $8/64 = 12.5\%$。

**每设备 MXU FLOPs 总量**：

$$\boxed{\text{FLOPs}_{\text{total}} = E_{\text{local}} \times 6 \cdot btc \cdot H \cdot I}$$

#### 2.2.3 每专家 HBM 流量（详细）

Tile 化权重加载下每专家 FFN 的访存量：

$$HBM_{\text{weights}} = 3 \cdot H \cdot I \cdot \tau$$

Token staging (HBM → VMEM)：

$$HBM_{\text{tokens}} = n_e \cdot H \cdot \tau$$

结果写回：

$$HBM_{\text{results}} = n_e \cdot H \cdot \tau$$

**每专家 HBM 读取总计**：

$$\boxed{HBM_{\text{read}}^{\text{expert}} = 3 \cdot H \cdot I \cdot \tau + n_e \cdot H \cdot \tau}$$

**每设备 HBM 读取总计（全部本地专家）**：

$$\boxed{HBM_{\text{read}}^{\text{all}} = E_{\text{local}} \times 3 \cdot H \cdot I \cdot \tau + T_{\text{local}} \times k \times H \times \tau}$$

**主导项**：对 $E=256, EP=8, H=8192, I=2048$：
- 权重项：$32 \times 3 \times 8192 \times 2048 \times 2 = 3.0 \text{ GB}$
- Token 项：$256 \times 8 \times 8192 \times 2 = 32 \text{ MB}$
- 权重主导，比例约 100:1

#### 2.2.4 算术强度 (Arithmetic Intensity)

每专家算术强度（含 padding）：

$$\boxed{AI_{\text{executed}} = \frac{6 \cdot btc \cdot H \cdot I}{3 \cdot H \cdot I \cdot \tau + n_e \cdot H \cdot \tau} \approx \frac{2 \cdot btc}{\tau}}$$

$\tau=2$ (BF16) 时：$AI_{\text{executed}} \approx btc = 64 \text{ FLOPs/byte}$。Ridge point = $2307/3.69 = 625$。

每专家有效 AI：

$$\boxed{AI_{\text{useful}} = AI_{\text{executed}} \times \frac{\bar{n}_e}{btc} \approx \frac{2 \cdot \bar{n}_e}{\tau}}$$

$\bar{n}_e = 8, \tau = 2$ 时：$AI_{\text{useful}} = 8.0 \text{ FLOPs/byte}$。

**核心洞察**：$AI_{\text{useful}}$ 仅取决于 $\bar{n}_e$ 和 $\tau$，与 block config 或隐藏维度无关。这意味着：

- **Decode** ($\bar{n}_e \approx 8$)：$AI \approx 8 \ll 625$ → **极度访存受限**
- **Prefill** ($\bar{n}_e \approx 512$)：$AI \approx 512 \approx 625$ → **接近 ridge point，可能计算受限**

#### 2.2.5 All-to-All Combine (Gather)

Scatter 的反向操作：每个专家的 FFN 输出被发送回 token 的原始设备。

每设备 gather 量（与 scatter 对称）：

$$V_{\text{gather}} = T_{\text{local}} \times k \times H \times \tau$$

$$V_{\text{ICI-in}}^{S2\text{-gather}} = V_{\text{ICI-out}}^{S2\text{-disp}}$$

#### 2.2.6 Stage 2 总时间估算

$$\boxed{T_{S2} \approx \max\left( \frac{HBM_{\text{read}}^{\text{all}}}{\Theta_{\text{HBM}}}, \frac{\text{FLOPs}_{\text{total}}}{\Phi_{\text{MXU}}} \right) + T_{\text{scatter}} + T_{\text{gather}}}$$

Ling 2.6 decode (EP=8)：$T_{S2}^{\text{HBM}} \approx 3.0 \text{ GB} / 3690 \text{ GB/s} = 0.81 \text{ ms}$，$T_{S2}^{\text{compute}} \approx 206 \text{ GFLOPS} / 2307 \text{ TFLOPS} = 0.09 \text{ ms}$。Stage 2 **受 HBM 带宽限制**，理论下界 ~0.8 ms。

### 2.3 Stage 3: 共享专家与 MoE 输出聚合

#### 2.3.1 输出聚合

对每个 token $t$，加权求和 top-k 专家输出，再加上共享专家输出：

$$y_t = \sum_{r=1}^{k} w_{t,r} \cdot \text{expert\_output}_{t,r} + y_t^{\text{se}}$$

**HBM 读取**：从 `a2a_g_hbm` 加载 top-k 专家输出。

$$HBM_{\text{read}}^{S3} = T_{\text{local}} \times k \times H \times \tau$$

**VPU FLOPs**：$T_{\text{local}} \times k \times H$ 次乘加。

$$FLOPs_{S3} = 2 \cdot T_{\text{local}} \cdot k \cdot H$$

Ling 2.6 decode：8 MB 读取，~4 MFLOPS——可忽略 (< 10 μs)。

#### 2.3.2 共享专家贡献

共享专家输出 (VMEM 中 F32 累加器) 加到 MoE 加权和：

```
`y_t_final = bf16(y_t_f32 + se_output_f32[t])`
```

此为 VMEM 内操作，无额外 HBM 流量。

#### 2.3.3 输出写回

$$HBM_{\text{write}}^{S3} = T_{\text{local}} \times H \times \tau$$

Ling 2.6：1 MB。

### 2.4 Stage 4: 共享专家 FFN（独立分析）

#### 2.4.1 共享专家架构

共享专家对所有 token 执行独立的 FFN (Gate + Up + 激活 + Down)，与路由无关：

$$y_{\text{se}} = W2_{\text{se}} \cdot \text{SiLU}(W1_{\text{se}} \cdot x) \odot (W3_{\text{se}} \cdot x)$$

权重：$W1_{\text{se}}, W3_{\text{se}} \in \mathbb{R}^{H \times I_{\text{se}}}$，$W2_{\text{se}} \in \mathbb{R}^{I_{\text{se}} \times H}$。

#### 2.4.2 SE HBM 流量（完整）

$$HBM_{\text{read}}^{S4\text{-weights}} = 3 \cdot H \cdot I_{\text{se}} \cdot \tau$$

$$HBM_{\text{read}}^{S4\text{-tokens}} = T_{\text{local}} \cdot H \cdot \tau$$

$$HBM_{\text{write}}^{S4} = 0 \quad \text{(输出以 F32 累加器形式保留在 VMEM)}$$

#### 2.4.3 SE FLOPs

$$\boxed{\text{FLOPs}_{S4} = 6 \cdot T_{\text{local}} \cdot H \cdot I_{\text{se}}}$$

Ling 2.6 decode：$6 \times 256 \times 8192 \times 2048 = 25.8 \text{ GFLOPS}$。

#### 2.4.4 SE 权重点比 (Weight-to-Token Ratio)

SE 调度的关键 tradeoff：

$$\frac{HBM_{\text{weights}}^{S4}}{HBM_{\text{tokens}}^{S4}} = \frac{3 \cdot H \cdot I_{\text{se}} \cdot \tau}{T_{\text{local}} \cdot H \cdot \tau} = \frac{3 \cdot I_{\text{se}}}{T_{\text{local}}}$$

Ling 2.6 decode ($I_{\text{se}}=2048, T_{\text{local}}=256$)：比例 = 24:1 — 权重加载主导。

Prefill ($T_{\text{local}} = 2048$)：比例 = 3:1 — 更均衡。

#### 2.4.5 SE 算术强度

$$AI_{\text{SE}} = \frac{6 \cdot T_{\text{local}} \cdot H \cdot I_{\text{se}}}{3 \cdot H \cdot I_{\text{se}} \cdot \tau + T_{\text{local}} \cdot H \cdot \tau} = \frac{6 \cdot T_{\text{local}} \cdot I_{\text{se}}}{3 \cdot I_{\text{se}} \cdot \tau + T_{\text{local}} \cdot \tau}$$

$T_{\text{local}} = 256$：$AI_{\text{SE}} \approx 98 \text{ FLOPs/byte}$（访存受限，但不如路由专家严重）。
$T_{\text{local}} = 2048$：$AI_{\text{SE}} \approx 384 \text{ FLOPs/byte}$（接近 ridge point）。

---

## 第三部分：LLO Overhead 与 VMEM 分析

### 3.1 理论—实测差距分解

现有 benchmark：理论下界 ~1.8 ms（EP=4, bt=64 时 HBM 限制），实测 ~149-150 ms — **~83x 差距**。

| Overhead 类别 | 估算贡献 | 机制 |
|-------------------|----------------------|-----------|
| DMA setup 开销 | ~40-60% | 每个 tile 的 DMA descriptor 构建、semaphore 操作（64 专家约 1,536 次 DMA 请求） |
| 同步 barrier | ~10-15% | 每次迭代的 `sync_barrier()`、`wait_scatter_recv()`、`wait_gather_recv_all()` |
| 循环控制 (SALU) | ~15-25% | 64 专家的 `lax.fori_loop`（unroll=false），每次迭代有控制流开销 |
| 小 DMA 低效 | ~5-10% | ~256 KB token tile、KB 级 metadata 传输——DMA 引擎利用率低 |
| VMEM spill/fill | ~2-5% | 约 14% 的 bundle 中观测到 VSTORE:SPILL |
| ICI 延迟 | ~2-5% | Scatter/gather 的跨芯片 all-to-all 延迟 |

**有效 HBM 带宽**：$6,249 \text{ MB} / 149 \text{ ms} \approx 42 \text{ GB/s}$ — 仅为峰值 3,690 GB/s 的 1.1%。

### 3.2 多专家处理开销模型

将 overhead 建模为加法项而非乘法因子：

$$\boxed{T_{\text{measured}} = T_{\text{HBM}} + T_{\text{compute}} + E_{\text{local}} \times t_{\text{overhead-per-expert}} + T_{\text{comm}} + T_{\text{fixed}}}$$

其中：
- $T_{\text{HBM}} = HBM_{\text{read}}^{\text{all}} / \Theta_{\text{HBM}}^{\text{effective}}$ — 由于大量小 DMA，有效 BW 远低于峰值
- $t_{\text{overhead-per-expert}}$ = 每次专家迭代的 DMA setup + semaphore wait + 循环控制
- $T_{\text{fixed}}$ = barrier + output acc + 其他固定开销

LLO 分析推导：$t_{\text{overhead-per-expert}} \approx 2.3 \text{ ms}$（149 ms / 64 experts），以 SALU 控制流为主导。

### 3.3 VMEM 占用公式

#### 3.3.1 Buffer 清单

| Buffer | Shape | 大小公式 | Ling 2.6 (bt=64, bf=1024, bd1=bd2=2048) |
|--------|-------|-------------|------------------------------------------|
| Token 输入 (×2) | `[bt, H]` BF16 | $2 \times bt \times H \times \tau$ | 2 MB |
| W1 双缓冲 | `[p, bd1/p, bf]` BF16 ×2 | $2 \times 2 \times bd1 \times bf \times \tau$ | 8 MB |
| W3 双缓冲 | 同 W1 | $2 \times 2 \times bd1 \times bf \times \tau$ | 8 MB |
| W2 双缓冲 | `[p, bf, bd2/p]` BF16 ×2 | $2 \times 2 \times bf \times bd2 \times \tau$ | 8 MB |
| Gate 累加器 | `[btc, bf]` F32 | $btc \times bf \times \tau_{\text{acc}}$ | 0.25 MB |
| Up 累加器 | `[btc, bf]` F32 | $btc \times bf \times \tau_{\text{acc}}$ | 0.25 MB |
| FFN2 累加器 | `[btc, bd2c]` F32 ×3 (三缓冲) | $3 \times btc \times bd2c \times \tau_{\text{acc}}$ | 1.5 MB |
| SE W1 buffer (×2) | `[p, bd1/p, bse]` BF16 ×2 | $2 \times 2 \times bd1 \times bse \times \tau$ | 2 MB |
| SE W2 buffer (×2) | `[p, bse, bd2/p]` BF16 ×2 | $2 \times 2 \times bse \times bd2 \times \tau$ | 2 MB |
| SE W3 buffer (×2) | 同 SE W1 | $2 \times 2 \times bd1 \times bse \times \tau$ | 2 MB |
| SE 累加器 | `[bt, H]` F32 | $bt \times H \times \tau_{\text{acc}}$ | 2 MB |
| 输出 buffer (×2) | `[bt, H]` BF16 | $2 \times bt \times H \times \tau$ | 2 MB |
| A2A scatter (×2) | `[E_buf, bt*EP, p, H/p]` BF16 | 随配置变化 | ~16-32 MB |

#### 3.3.2 VMEM 峰值约束

$$\boxed{M_{\text{VMEM}} = \sum_{\text{buffers}} \text{size} \leq 64 \text{ MiB}}$$

Ling 2.6 EP=4（大 bt=64）：~56 MB → 87% 利用率。

Ling 2.6 EP=8（bt 可能不同）：$E_{\text{local}}$ 减半，利用率更低。

**核心 tradeoff**：更大的 `bt` 增加 VMEM 压力（token buffer 随 $bt \times H$ 增长）但减少 $n_{bt}$（更少的外层循环迭代）。约束条件：

$$\boxed{bt_{\text{max}}} \approx \frac{64 \text{ MiB} - W_{\text{buffers}} - A_{\text{buffers}}}{2 \times H \times (\tau + \tau_{\text{acc}})}$$

$H=8192$ 时：$bt_{\text{max}} \approx 512$（理论值，扣除权重 buffer 前）。

### 3.4 DMA 效率模型

Kernel 的有效 HBM 带宽是 DMA 请求大小的函数：

$$\Theta_{\text{HBM}}^{\text{effective}}(s) = \Theta_{\text{HBM}}^{\text{peak}} \times \eta(s)$$

其中 $\eta(s)$ 为传输大小 $s$ 对应的 DMA 效率：

| 传输大小 | 典型 $\eta$ | 示例 |
|--------------|----------------|---------|
| 4 MB（权重 tile，bd1=2048, bf=1024） | ~80-90% | W1/W3 tile 加载 |
| 256 KB（token tile，bt=64） | ~40-60% | Token staging |
| 4 KB（metadata） | ~5-10% | 路由表 |

这解释了大部分理论—实测差距：kernel 发射了大量小 DMA 请求，HBM 引擎无法在这些场景下达到峰值吞吐。

---

## 第四部分：决策点与 Tradeoff 计算

### 4.1 共享专家与 Stage 1 的 Overlap

#### 4.1.1 当前方案：SE 与 Stage 2 交错

SE 当前被切片并与路由专家 FFN 交错执行（每个专家迭代的 `se_before` + `se_after`）。SE 计算与专家 FFN DMA 重叠——两者竞争 HBM 带宽。

#### 4.1.2 替代方案：SE 与 Stage 1 Overlap

**核心洞察**：SE 仅需要输入 token，不需要路由结果。Stage 1 不使用 token 的 HBM 带宽（仅 ~4 KB 的 metadata）。

**可行性**：Stage 1 期间（gate 计算 + allreduce metadata），token 在 HBM 中空闲。可以：

1. Stage 1 期间预取 SE 权重和 token 到 VMEM
2. 在等待 allreduce barrier 时执行 SE FFN1 (gate+up)
3. 在 scatter setup 期间执行 SE FFN2 (down)

**VMEM 约束**：SE buffer（W1/W3/W2 buffer + SE 累加器）与 S1/S2 buffer 共享 VMEM。需检查峰值：

$$M_{\text{VMEM}}^{\text{SE-during-S1}} = M_{\text{SE-buffers}} + M_{\text{S1-buffers}} \leq 64 \text{ MiB}$$

SE buffer：~8 MB（W1/W3/W2 双缓冲 + 累加器）。
S1 buffer：token 输入 (2 MB) + 后续专家 FFN 权重 buffer (8+8+8=24 MB)。
合计：~34 MB — 可行。

**收益**：SE 计算（~26 μs 理论值）从关键路径上"免费"，因为它与 Stage 1 的通信延迟（~15-20 μs）重叠。

**风险**：Scatter 阶段额外的 HBM 流量增加竞争：
- SE 权重 (96 MB) 在 Stage 1 期间加载
- 可能延迟 scatter DMA 从 HBM 获取 token

**定量评估**：SE 权重加载 (96 MB) 在峰值 HBM BW 下需 ~26 μs。Stage 1 需 ~15-20 μs。可实现部分 overlap：SE FFN1（权重+计算）≈ 13 μs 落入 15-20 μs 的 Stage 1 窗口。剩余 ~13 μs 溢入 Stage 2。

#### 4.1.3 结论

Overlap 有适度收益（~10-15 μs 节省）但增加复杂性。当前交错方案合理，因为：
1. SE 已不在关键路径上（被专家 FFN 掩盖）
2. SE 权重加载与专家权重加载共享 HBM 带宽
3. Stage 1 太短（15 μs）无法完全掩盖 SE 计算（26 μs）

### 4.2 A2A Combine 与下一次 Dispatch 的 DMA 竞争

#### 4.2.1 当前方案：流水线化 Scatter + Gather

Batch scatter 路径：
- 所有 scatter DMA 一次性发射（token-loop 单遍）
- 每个专家：wait scatter recv → ffn → start gather
- Gather DMA 逐专家发射，可能与后续专家的 scatter 重叠

Pipelined 路径（`expert_buffer_count < local_num_experts` 时）：
- 每个专家：在专家 N 的 FFN 期间 start scatter(N+1)
- 这产生背靠背 DMA：gather(N) DMA + scatter(N+1) DMA

#### 4.2.2 DMA 竞争模型

DMA 引擎有有限队列深度 ($D_{\text{queue}}$)，HBM 带宽被共享。背靠背 DMA 发射时：

$$T_{\text{DMA-total}} = \sum_i T_{\text{DMA}}(s_i) + \sum_{i,j} \delta(s_i, s_j)$$

其中 $\delta(s_i, s_j)$ 是 DMA 流 $i$ 和 $j$ 之间的切换开销。

**定量估算**：64 个专家，每个发射约 8 个 gather DMA（每个 bf×bd2 tile）和下一个专家的约 8 个 scatter DMA，交错产生 ~512 次 DMA 流切换。每次切换约 50-100 ns（HBM 控制器重仲裁）。总开销：~25-50 μs。

相对于总 kernel 时间（150 ms）较小，但对理论下界（1.8 ms）不可忽略。

#### 4.2.3 缓解方案：Batching

所有 FFN 完成后批量发射 gather（延迟方案，见 4.3），通过分离 DMA 阶段完全消除此竞争。

### 4.3 延迟 Combine + 共享专家 Overlap

#### 4.3.1 当前方案：交错 Gather

```python
for e in experts:
    expert_ffn(e)
    start_a2a_gather(e)  # 立即发射 DMA
```

#### 4.3.2 提议方案：延迟 Gather + SE 填充

```python
for e in experts:
    expert_ffn(e)
    # 不启动 gather —— 延迟

# 全部 FFN 完成后：
for e in experts:
    start_a2a_gather(e)  # 批量发射全部 gather DMA

# 同时：运行剩余 SE slices
```

#### 4.3.3 分析

**优势**：
- 计算（FFN）和通信（gather）阶段清晰分离
- SE 计算填充 FFN 完成到 gather 完成之间的间隙
- 消除 gather 和下一次 scatter 之间的 DMA 竞争

**劣势**：
- Gather 不再与专家 FFN 计算重叠
- 关键路径：$T_{\text{FFN-all}} + \max(T_{\text{SE-remaining}}, T_{\text{gather-all}})$
- 对比当前：$T_{\text{FFN-all}} + T_{\text{gather-tail}}$（大部分 gather 被掩盖）

**定量比较**：

当前方案：$T_{\text{crit}} \approx T_{\text{FFN-all}} + T_{\text{gather-tail}} \approx 64 \times 28 \mu\text{s} + 5 \mu\text{s} = 1797 \mu\text{s}$

延迟方案：$T_{\text{crit}} \approx T_{\text{FFN-all}} + \max(T_{\text{SE-rem}}, T_{\text{gather-all}})$

其中 $T_{\text{SE-rem}} \approx 26 \mu\text{s}$（当前方案中 SE 在 expert 4 基本完成，可全部延迟），$T_{\text{gather-all}} \approx 40 \mu\text{s}$。

$T_{\text{crit}} \approx 1792 + \max(26, 40) = 1832 \mu\text{s}$ — **略差** (+35 μs)。

但延迟方案有更好的 HBM 利用率（无 DMA 竞争），可能减少 1-2% 的每专家 DMA 开销。净效果在当前配置下近似中性。

**延迟方案更好的场景**：当 SE 相对于专家 FFN 更大（大 $I_{\text{se}} / I$），或 gather DMA 竞争严重（小 `expert_buffer_count`）。

### 4.4 共享专家 TP 切分

#### 4.4.1 当前方案：SE 复制

每个 EP 设备持有 SE 权重的完整副本（$3 \times H \times I_{\text{se}} \times \tau$ bytes），独立对其本地 token 计算 SE。SE 无跨设备通信。

SE 权重每设备 HBM 占用：$3 \times 8192 \times 2048 \times 2 = 96 \text{ MB}$（HBM，每个 bt tile 加载一次）。

#### 4.4.2 TP 切分方案

沿中间维度在 TP 设备间切分 SE 权重：

- 每个 TP shard 持有 W1/W3 列和 W2 行的 $I_{\text{se}} / TP$
- FFN1：部分输出，需 allreduce 获得完整激活值
- FFN2：每个 shard 计算部分输出，需 allreduce 获得完整结果

**每次 TP 切分的通信代价**：

$$V_{\text{comm}}^{\text{TP}} = 2 \times T_{\text{local}} \times I_{\text{se}} \times \tau_{\text{f32}} \quad \text{(FFN1 输出 + FFN2 输出)}$$

Ling 2.6 decode：$2 \times 256 \times 2048 \times 4 = 4 \text{ MB}$。

**每设备计算节省**：每设备计算 $1/TP$ 的 SE FLOPs。但 SE 已经小于总 FLOPs 的 1%（26 GFLOPS vs 412 GFLOPS），节省可忽略。

**每设备 HBM 节省**：每设备加载 $3 \times H \times I_{\text{se}} / TP \times \tau$ bytes。TP=2 时：48 MB vs 96 MB。

**延迟分析**：

无 TP：$T_{\text{SE}} \approx 96 \text{ MB} / \Theta_{\text{HBM}} = 26 \mu\text{s}$ (HBM 受限)

TP=2：$T_{\text{SE}} \approx 48 / 3690 + 4 / 300 = 13 + 13 = 26 \mu\text{s}$ (HBM + ICI)
加 allreduce 延迟：~10-20 μs → 总计 ~36-46 μs。

**结论**：Decode 场景下 SE TP 切分**无收益**，因为：
1. 通信延迟主导了节省的 HBM 时间
2. SE 已是总运行时的极小部分
3. SE 不在关键路径上（与专家 FFN 交错）

**TP 有收益的场景**：$I_{\text{se}}$ 非常大（如 8192+），SE 权重 HBM 流量显著。或 prefill 场景 $T_{\text{local}}$ 大，allreduce 带宽利用率高。

### 4.5 `o_proj` 与 DP Attention 的重组与 Overlap

#### 4.5.1 当前架构

Fused MoE kernel 后，输出 $y \in \mathbb{R}^{B \times H}$ 分布在 EP 设备上：
- 每个设备持有被路由到其本地专家的 token 的输出
- Token 分布取决于路由决策

下游 attention 需要：
- 每个 DP 副本需要其完整的 token 序列
- 若 DP>1，token 需重分布（allgather 或 reshape）

#### 4.5.2 MoE 后的 Token 分布

Kernel 后：token 在持有专家的设备上，不一定是其原始设备。对下一层：
- Gate/routing 从头重算（每层独立）
- 输入 token 需在正确的 DP+EP shard 上

当前 sglang-jax 模式：**跨所有 EP 设备复制 token**（MoE 后 allgather），使每个设备有完整输出。然后 DP sharding 用于 attention。

#### 4.5.3 Overlap 策略

输出写回 (`start_send_bo`) 已使用 DMA 将输出从 VMEM 流式传输到 HBM。这可以与下一层 gate 投影的开头 overlap：

```
Timeline:
  [Expert FFN] [Acc Output] [o_proj DMA] [下一层 Gate]
                                │              │
                                └── OVERLAP ──┘
```

这要求下一层在 `o_proj` DMA 完成前开始——若 gate 投影以 tile 方式处理 token 则可行。

**定量模型**：

$$T_{\text{o-proj}} = \frac{T_{\text{local}} \times H \times \tau}{\Theta_{\text{HBM}}} \quad \text{(DMA 写)}$$

Ling 2.6 decode：$256 \times 8192 \times 2 / 3.69 \times 10^{12} \approx 1.1 \mu\text{s}$ — 可忽略。

**真正的主要开销**是跨 EP token 复制的 allgather（如需要）：

$$V_{\text{allgather}} = T_{\text{local}} \times H \times \tau \times (EP - 1)$$

EP=8：$256 \times 8192 \times 2 \times 7 / 8 = 3.5 \text{ MB}$ 每设备出站。

#### 4.5.4 建议

1. **将 `o_proj` 与输出聚合融合**：不在写出 BF16 输出后再读回给 attention，直接在 VMEM 中对 F32 累加器做 `o_proj` matmul，然后写 `o_proj` 输出。
2. **`o_proj` DMA 与下一个 gate overlap**：`o_proj` 输出的第一个 token tile 进入 HBM 后立即启动下一层 gate matmul。
3. **避免不必要的 allgather**：若 attention 也是 EP sharded（如 DeepSpeed Ulysses 风格序列并行），可能不需要 token 重分布。

### 4.6 Prefill vs Decode 最优专家计算策略

#### 4.6.1 场景分类

核心判别量为 $\bar{n}_e$（每专家平均 token 数）：

| 场景 | $\bar{n}_e$ | $AI_{\text{useful}}$ | 瓶颈 | 示例 |
|--------|-------------|---------------------|------------|---------|
| **Decode** | 1-16 | 1-16 | 极度 HBM 受限 | $B=256, E=256, k=8 \rightarrow \bar{n}_e=8$ |
| **小 Prefill** | 32-128 | 32-128 | HBM 受限 | $B=1024, E=256, k=8 \rightarrow \bar{n}_e=32$ |
| **大 Prefill** | 256-2048 | 256-2048 | 接近 ridge / 计算受限 | $B=8192, E=256, k=8 \rightarrow \bar{n}_e=256$ |

BF16 的 ridge point 为 $AI_{\text{ridge}} = 625$。$\bar{n}_e \approx 300$ 时，$AI_{\text{useful}} \approx 300$，仍然偏访存受限。完全计算受限需要 $\bar{n}_e > 625$。

#### 4.6.2 Decode 最优设计 ($\bar{n}_e \ll btc$)

**问题**：$btc=64$ 但 $\bar{n}_e \approx 8$ → 87.5% 的 MXU FLOPs 浪费在 padding 上。

**优化策略**（按优先级）：

1. **减小 `btc`**：设 $btc = \min(\bar{n}_e, \text{aligned})$。但这会增加 $n_{\text{btc}}$（更多 sub-tile），成比例增加 DMA 开销。存在一个平衡点。

2. **FP8 量化**：权重 HBM 流量减半 → HBM 时间接近减半。这是 decode 场景下**收益最大**的改变。

3. **专家合并/批处理**：将多个小专家合并到共享 GEMM 调用中以摊薄 DMA setup 开销。例如，将 4 个 token 数相近的专家批量 dispatch 到一次 matmul。

4. **Token-stationary（方案 B）**：当 $n_{bt} > 1$（负载不均导致）时，从按 bt tile 扫描全部专家切换到每个专家一次性处理其完整 token batch。这消除了冗余权重加载，代价是 token 复制 ($k \times$)。

5. **减小 EP size**：更大的 $E_{\text{local}}$ 意味着每设备更多计算。但权重内存增长 $\propto EP$，可能超出 HBM 容量。

**Decode 最优 block config 选择**：

$$\min_{bt, btc, \ldots} \quad T_{\text{total}} = n_{bt} \times \left[ \frac{E_{\text{local}} \times 3 \cdot H \cdot I \cdot \tau}{\Theta_{\text{HBM}}^{\text{eff}}} + E_{\text{local}} \times \frac{6 \cdot btc \cdot H \cdot I}{\Phi_{\text{MXU}}} \right]$$

约束条件：
- $bt \leq T_{\text{local}}$，$bt \mid T_{\text{local}}$
- $btc \leq bt$，$btc \mid bt$
- $VMEM \leq 64 \text{ MiB}$
- $btc \geq 128$（MXU tile 粒度约束）

Decode 场景下 HBM 项主导 → 选择**尽可能大的 `bt`** 以最小化 $n_{bt}$，避免冗余权重加载。

#### 4.6.3 Prefill 最优设计 ($\bar{n}_e \gg btc$)

**问题**：$\bar{n}_e \approx 512\text{-}2048$，$btc$ 可能较小 → 计算受限或接近 roofline。权重加载成本被大量 token 摊销。

**优化策略**：

1. **最大化 `btc`**：设 $btc = \min(\bar{n}_e, \text{VMEM 约束上限})$ 以最大化 MXU 利用率。更大的 $btc$ → 更少 sub-tile → 更少 DMA 开销。

2. **Token-stationary（方案 B 优先）**：Token 留在 VMEM，权重流式加载。每个专家的权重只加载一次。由于 $\bar{n}_e$ 大，token 复制成本 ($k \times H \times \tau$) 变得显著——选择 $bt$ 以平衡 token VMEM 与权重 buffer。

3. **Token 双缓冲**：$bt$ 大时（prefill），跨 bf/bd 维度的 token 双缓冲对掩盖 DMA 延迟至关重要。

4. **通信与计算 overlap**：Token 量大意味着 all-to-all 量大。若 $n_{bt} > 1$，考虑在计算当前 tile 时启动下一个 bt tile 的 scatter。

**Prefill token-stationary VMEM 约束**：

$$\boxed{bt_{\text{max}} = \frac{64 \text{ MiB} - W_{\text{buffers}}}{H \times (\tau + 2 \times \tau_{\text{acc}} \times bf / H)}}$$

$H=8192, \tau=2, \tau_{\text{acc}}=4, bf=1024, W_{\text{buffers}}=24 \text{ MB}$：
$bt_{\text{max}} \approx 2048$ tokens — 足以应对大多数 prefill batch。

#### 4.6.4 策略切换点

最优策略在 $\bar{n}_e$ 跨过权重加载/Token 计算的 tradeoff 点时发生变化：

$$\bar{n}_e^* \approx \frac{3 \cdot I \cdot \tau}{H \cdot \tau} = \frac{3 \cdot I}{H}$$

Ling 2.6：$\bar{n}_e^* = 3 \times 2048 / 8192 = 0.75$。由于 $\bar{n}_e \geq 1$ 始终成立，prefill 类策略对 Ling 2.6 *始终更优*。但 VMEM 的实际约束使 $bt$ 不能超过 ~2048。

*有效*判别标准：

$$\boxed{\text{Strategy} = \begin{cases}
\text{Weight-streaming (方案 A)} & \text{if } T_{\text{local}} \leq bt_{\text{max}} \text{ 且 } n_{bt}=1 \\
\text{Token-stationary (方案 B)} & \text{if } \bar{n}_e \text{ 较大且权重重载代价 } > \text{ token 复制代价} \\
\text{混合} & \text{否则 — 小专家合并，大专家流式}
\end{cases}}$$

当前 Ling 2.6 配置 (decode, $T_{\text{local}}=256$, $bt=64$)：方案 A 且 $n_{bt}=1$ 为最优 — 权重恰好加载一次，控制流简单。

#### 4.6.5 Decode vs Prefill 定量对比

| 指标 | Decode (B=256) | 小 Prefill (B=2048) | 大 Prefill (B=8192) |
|--------|---------------|----------------------|----------------------|
| $T_{\text{local}}$ (EP=8) | 256 | 2048 | 8192 |
| $\bar{n}_e$ | 8 | 64 | 256 |
| $AI_{\text{useful}}$ | 8.4 | 67.1 | 268.4 |
| $AI_{\text{executed}}$ (btc=64) | 67.1 | 67.1 | 67.1 |
| HBM 读取 (权重) | 3.0 GB | 3.0 GB | 3.0 GB |
| HBM 读取 (Token) | 32 MB | 256 MB | 1 GB |
| MXU FLOPs (执行) | 206 GFLOPS | 206 GFLOPS | 206 GFLOPS |
| MXU FLOPs (有效) | 26 GFLOPS | 206 GFLOPS | 206 GFLOPS |
| $T_{\text{HBM}}$ | 0.81 ms | 0.88 ms | 1.08 ms |
| $T_{\text{compute}}$ | 0.09 ms | 0.09 ms | 0.09 ms |
| 瓶颈 | HBM-BW (9:1) | HBM-BW (10:1) | HBM-BW (12:1) |
| 最优 $btc$ | 64 (any) | 128 | 256+ |
| 量化收益 (FP8) | ~2× 加速 | ~2× 加速 | ~1.5× 加速 |

> **核心洞察**：即使在大 batch size 下，Ling 2.6 decode/prefill 仍然是 HBM 受限的，因为权重加载主导（3 GB 权重 vs 最多 ~1 GB token）。AI 被 $\bar{n}_e$ 限制，需要超过 ~300 才能接近 ridge point。对于该模型架构，kernel 是结构性的 HBM 受限。

---

## 参考资料

- 现有 Ling 2.6 分析：`primatrix/wiki/fused_moe_ling2.6_theoretical_perf_analysis.md`
- TPU 性能模型步骤：`primatrix/wiki/steps_fused_moe_per_expert.json`、`steps_fused_moe_bt_tile.json`
- Kernel 实现：`kernels/_fused_moe_impl.py`
- Benchmark 结果：`benchmark_results/benchmark_result.json`
- LLO dump：`benchmark_results/ir_dumps/llo/`
