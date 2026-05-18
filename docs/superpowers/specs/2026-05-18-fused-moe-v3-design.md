# RFC: Fused EP MoE v3 — 基于 multi_expert_pipeline 的重写设计

**日期**: 2026-05-18
**状态**: 设计文档
**前置文档**: `2026-05-18-fused-moe-v2-pipeline-analysis.md`（v2 流水分析）
**基线代码**:
- `kernels/_fused_moe_v2_impl.py`（当前 v2 实现）
- `kernels/multi_expert_pipeline.py`（核心循环模板）

**目标配置**:
- MiMo V2 Pro: E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=32
- Ling 2.6 1T: E=256, H=8192, I=2048, top_k=8, bf16, ep=4

---

## 1. 设计动机

### 1.1 v2 的核心问题

v2 分析（前置文档）识别出五个主要性能问题：

| # | 问题 | 根因 | 量化影响 |
|---|------|------|---------|
| 1 | Token 非持久 | 每个 bts tile 重新加载 token | 每 tile 增加 ~5-10 μs DMA 固定开销 |
| 2 | W2 优先级 bug | 所有 `start_fetch_w2` 传入 priority=1 | W2 DMA 与 gate/up MXU 竞争带宽 |
| 3 | Writeback 同步 | bts tile 间 `cast + DMA + wait` 阻塞 | MXU 空闲 ~5-15 μs/tile |
| 4 | 代码路径爆炸 | 3 compute × 2 A2A × decode flag | 维护困难，编译器难以全局优化 |
| 5 | VMEM 低利用率 | 小 bts buffer + 无跨专家重叠 | 33%（MiMo），48%（Ling） |

### 1.2 multi_expert_pipeline 的解决方案

`multi_expert_pipeline.py` 已在单设备场景下验证了以下优化：

| 优化 | 机制 | 验证状态 |
|------|------|---------|
| Token 持久驻留 | 加载一次，所有 bf tile 共享 | ✅ 已通过正确性测试 |
| 跨专家 token 双缓冲 | x_slot 0/1 交替，epilogue 预取 | ✅ DMA overlap 已验证 |
| 跨专家 output 双缓冲 | y_out 复用 x_slot，drain e-2 | ✅ 消除 writeback bubble |
| W2 低优先级 | `start_fetch_w2(..., priority=0)` | ✅ 与 gate/up MXU 重叠 |
| 简洁代码 | Python for-loop + fori_loop | ✅ 411 行 vs v2 的 2162 行 |

### 1.3 v3 目标

将 multi_expert_pipeline 的核心循环包装进 EP 通信框架，同时添加 fp8 支持：

```
v3 = multi_expert_pipeline 核心循环
   + v2 的 A2A scatter/gather（batch 模式）
   + v2 的 metadata precompute（JAX allreduce）
   + v2 的 output accumulation
   + fp8 direct_scaled_dot 路径
   − shared expert（后续添加）
   − prefill 支持（后续添加）
   − pipelined scatter（删除）
   − in-kernel metadata allreduce（删除）
   − VMEM dequant 路径（删除）
```

---

## 2. v3 全局流水架构

### 2.1 三层循环结构

v3 从 v2 的 5 层简化为 3 层：

```
for bt_id in 0..num_bt:                        ← L1: token 块循环
  metadata_allreduce()
  a2a_scatter_batch()
  for local_e_id in 0..local_num_experts:       ← L2: 专家循环（fori_loop）
    for bf_id in 0..num_bf:                     ← L3: 权重 tile 循环（Python for）
      compute_tile()
  a2a_gather_all()
  acc_and_store_output()
```

v2 的 L3（bts 分片）和 L5（btc 计算 tile）被消除：
- **bts 分片消除**：每个专家的全部 token（pad 到 bts）一次性加载到 VMEM
- **btc 消除**：直接对 bts 大小的 token 块做矩阵乘法（decode 下 bts 足够小，不需要进一步分片）

### 2.2 全局时间线（单 bt block）

```
时间 →

阶段 1: Metadata & Scatter
┌──────────┐┌─────────────────────────────┐
│ metadata ││ a2a_scatter_batch (all exp) │
└──────────┘└─────────────────────────────┘

阶段 2: Expert FFN Pipeline（核心 — multi_expert_pipeline 风格）
┌──────────────────────────────────────────────────────────────────────────┐
│ Expert 0    │ Expert 1    │ Expert 2    │ ... │ Expert E-1              │
│ [load x0]   │             │             │     │                         │
│ [FFN tiles] │ [load x1]   │             │     │                         │
│ [writeback] │ [FFN tiles] │ [load x2]   │     │                         │
│ [gather 0]  │ [writeback] │ [FFN tiles] │     │                         │
│             │ [gather 1]  │ [writeback] │     │                         │
│             │             │ [gather 2]  │ ... │                         │
│             │             │             │     │ [FFN tiles]             │
│             │             │             │     │ [writeback]             │
│             │             │             │     │ [gather E-1]            │
└──────────────────────────────────────────────────────────────────────────┘

阶段 3: Gather Wait & Output
┌──────────────────┐┌──────────────────────┐┌────────────┐
│ wait_gather_all  ││ acc_and_store_output  ││ send_output│
└──────────────────┘└──────────────────────┘└────────────┘
```

### 2.3 与 v2 全局时间线对比

**v2**:
```
Scatter → [Expert 0: load_x → FFN → wb → gather] → [Expert 1: load_x → FFN → wb → gather] → ...
                      ↑ 每个专家独立的 bts tile 循环，token 重载
                      ↑ writeback 与 FFN 串行
```

**v3**:
```
Scatter → [Expert 0: FFN + wb(async)] → [Expert 1: FFN + wb(async)] → ...
              ↑ token 已在 prologue 加载，持久驻留
              ↑ writeback 异步，与下一专家 FFN 重叠
              ↑ 跨专家预取 token + 首个权重 tile
```

---

## 3. Expert FFN 流水详解

### 3.1 单专家 FFN 时间线

以 MiMo fp8（H=6144, I=2048, bf=512, num_bf=4）为例：

```
时间 →

DMA:   [W1₀W3₀W2₀ already prefetched]  [prefetch W1₂W3₂W2₂]  [prefetch W1₃W3₃W2₃]
       [W1₁W3₁W2₁ already prefetched]
MXU:   ██ gate/up₀ ██ wait_W2₀ ██ act/down₀ ██ gate/up₁ ██ wait_W2₁ ██ act/down₁ ██ ...

详细时序（bf tile 0, slot 0）:
┌─────────────────────────────────────────────────────┐
│ wait W1[slot0]  ← DMA完成（prologue已发起）          │
│ wait W3[slot0]  ← DMA完成（prologue已发起）          │
│                                                      │
│ ┌─ MXU: gate = x @ W1 ─┐                            │
│ │ MXU: up   = x @ W3   │ ← 此时 W2[slot0] DMA     │
│ └───────────────────────┘   在后台传输（priority=0）  │
│                                                      │
│ wait W2[slot0]  ← deferred wait（通常已完成）         │
│                                                      │
│ ┌─ VPU: act = silu(gate) * up ─┐                    │
│ │ MXU: partial = act @ W2      │                    │
│ └──────────────────────────────┘                    │
│                                                      │
│ y_acc = partial  （首 tile 赋值）                     │
└─────────────────────────────────────────────────────┘

bf tile 1 (slot 1):
┌─────────────────────────────────────────────────────┐
│ start_fetch W1[slot0, tile2]  ← 复用 slot0           │
│ start_fetch W3[slot0, tile2]                         │
│ start_fetch W2[slot0, tile2, priority=0]             │
│                                                      │
│ wait W1[slot1]                                       │
│ wait W3[slot1]                                       │
│                                                      │
│ MXU: gate = x @ W1₁, up = x @ W3₁                  │
│                                                      │
│ wait W2[slot1]                                       │
│ MXU: partial = act₁ @ W2₁                           │
│ y_acc += partial                                     │
└─────────────────────────────────────────────────────┘

最后 tile (epilogue, slot = (num_bf-1)%2):
┌─────────────────────────────────────────────────────┐
│ ★ 预取下一专家的 token + 首个权重 tile ★              │
│ start_load_x(next_x_slot, next_expert)               │
│ start_fetch_w1(slot0, next_expert, tile0)            │
│ start_fetch_w3(slot0, next_expert, tile0)            │
│ start_fetch_w2(slot0, next_expert, tile0)            │
│                                                      │
│ wait W1[last_slot]; wait W3[last_slot]               │
│ MXU: gate/up 最后一个 tile                            │
│ wait W2[last_slot]                                   │
│ MXU: act/down 最后一个 tile                           │
│ y_acc += partial                                     │
└─────────────────────────────────────────────────────┘
```

### 3.2 专家边界流水

```
Expert E 尾声 → Expert E+1 序幕

时间 →

Expert E 最后 tile:
  [gate/up MXU]──[act/down MXU]
                       │
                       ├─ start_load_x(next_xs, e+1)     ← token 预取
                       ├─ start_fetch_w1(slot0, e+1, 0)   ← 权重预取
                       ├─ start_fetch_w3(slot0, e+1, 0)
                       └─ start_fetch_w2(slot0, e+1, 0)

Expert E 边界:
  wait_writeback(x_slot)  ← drain expert e-2
  y_out[x_slot] = y_acc.astype(bf16)   ← VPU: fp32→bf16 cast
  start_writeback(e, x_slot)            ← 异步 DMA，与后续重叠
  start_a2a_gather(e)                   ← 异步 DMA
  start_fetch_w1(slot1, e+1, 1)         ← 第二个权重 tile
  start_fetch_w3(slot1, e+1, 1)
  start_fetch_w2(slot1, e+1, 1, priority=0)
  wait_load_x(next_xs)                  ← 通常已完成

Expert E+1 首 tile:
  wait_fetch_w1(slot0)   ← 通常已完成（epilogue 中预取）
  wait_fetch_w3(slot0)
  [gate/up MXU]          ← MXU 立即开始
```

### 3.3 v2 vs v3 专家边界对比

**v2 专家边界（每个 bts tile 独立）**:
```
Expert E, bts tile last:
  [...bf tiles...]──[cast + DMA writeback + wait]──┐  ← MXU 空闲
                                                     │
Expert E+1:                                          │
  A2A scatter wait ─────────────────────────────────┘
  bts tile 0: [token load + wait] ← MXU 空闲
              [bf tile 0: weight wait + compute]
              ...
```

**v3 专家边界（multi_expert_pipeline 风格）**:
```
Expert E 最后 tile:
  [MXU: gate/up + act/down]
       ↑ 同时: token + weight DMA for E+1 已在后台
  [wait_wb(e-2)]       ← 等之前的 DMA（通常已完成）
  [cast + start_wb(e)] ← 异步，不等待
  [start_gather(e)]    ← 异步
  [start w1/w3/w2 tile1 for e+1]
  [wait_load_x]        ← 通常已完成
Expert E+1 首 tile:
  [wait w1, w3]        ← 通常已完成（epilogue 中预取）
  [MXU: gate/up]       ← MXU 几乎无间隔启动
```

**关键改进**: v3 消除了两个 MXU 空闲源：
1. writeback `wait` 推迟到 e+2（drain e-2 策略）
2. token load 在 epilogue 中预取（与最后 tile MXU 重叠）

---

## 4. DMA-Compute 重叠分析

### 4.1 MiMo V2 Pro 参数

```
H = 6144, I = 2048, bf = 512, num_bf = 4
t_packing = 2, h_per_t = 3072
bts = 16 (pad to)
qbk = 128 (fp8 quant_block_k)
n_sg = h_per_t / qbk = 24 (scale groups for FFN1)
n_sg2 = bf / qbk = 4 (scale groups for FFN2)
local_num_experts = 12 (384 / 32)
```

### 4.2 单 bf tile 计算时间估算

**FFN1 (gate + up)**:
- 每个 scale group: 2 × dot(bts, qbk) @ (qbk, bf) = 2 × 2 × bts × qbk × bf FLOPs
  - = 2 × 2 × 16 × 128 × 512 = 4.2M FLOPs/sg
- 24 个 scale groups × 2 packing: 24 × 2 × 4.2M = 201.3M FLOPs
- gate + up 各一遍: 2 × 201.3M = **402.7M FLOPs**

**FFN2 (act + down)**:
- 每个 scale group: 2 × dot(bts, qbk) @ (qbk, h_per_t) = 2 × 16 × 128 × 3072 = 12.6M FLOPs
- 4 个 scale groups × 2 packing: 4 × 2 × 12.6M = **100.7M FLOPs**
- Activation: ~5 × bts × bf ≈ 40K FLOPs（可忽略）

**单 bf tile 总计**: ~503.4M FLOPs

**TPU v7x MXU 峰值**: 275 TFLOPs (bf16)
- 理论时间: 503.4M / 275T = **1.83 μs/tile**
- 实际估算（含 MXU 效率损失、VPU activation、scale 乘法）: **~3-5 μs/tile**

**4 个 bf tiles 总计**: 4 × 3-5 = **12-20 μs/expert**

### 4.3 单 bf tile 权重 DMA 时间

**fp8 权重**:
- W1: h_per_t × bf × 1 byte × t_packing = 3072 × 512 × 1 × 2 = **3.0 MB**
- W3: 同 W1 = **3.0 MB**
- W2: bf × h_per_t × 1 byte × t_packing = 512 × 3072 × 1 × 2 = **3.0 MB**
- Scales: 3 × (小) ≈ **~300 KB**
- **单 tile 总计**: ~9.3 MB

**TPU v7x HBM 带宽**: 3.28 TB/s (per chip)
- DMA 时间: 9.3 MB / 3.28 TB/s = **2.83 μs/tile**

### 4.4 DMA-MXU 重叠效率

```
双缓冲稳态（bf tile N, slot = N%2）:

                DMA 带宽                    MXU
                ────────                    ────
    ┌─ fetch W1/W3/W2[tile N+2, 同slot] ─┐
    │  9.3 MB @ 3.28 TB/s               │
    │  = 2.83 μs                         │
    │                                    │
    │        ┌─ compute tile N ──────────────────────────┐
    │        │ gate/up: ~2-3 μs          │               │
    │        │ wait W2: 0 μs (已到)      │               │
    └────────│ act/down: ~1-2 μs         │               │
             └───────────────────────────────────────────┘
             总计 ~3-5 μs
```

**分析**:
- DMA 时间: ~2.83 μs
- MXU 时间: ~3-5 μs
- **DMA ≤ MXU → 权重 DMA 完全被 MXU 隐藏**
- MXU 利用率受限于 token 数量（M=16）和 HBM 带宽，不受 DMA-compute overlap 限制

### 4.5 W2 低优先级效果

**v2（priority=1 for all）**:
```
DMA 控制器队列: [W1(p=1)] [W3(p=1)] [W2(p=1)] [下一tile W1(p=1)] ...
→ 所有 DMA 公平调度，W2 可能与 W1/W3 竞争下一 tile 的带宽
```

**v3（W2 priority=0）**:
```
DMA 控制器队列: [W1(p=1)] [W3(p=1)] [下一tile W1(p=1)] [W2(p=0)] ...
→ W2 让位给下一 tile 的 W1/W3，确保 gate/up 计算不被阻塞
→ W2 在 gate/up 计算期间后台传输完成
```

**量化收益**: 难以精确估算，但在 DMA 带宽接近饱和时（多通道竞争），
priority=0 确保 W2 不会延迟下一 tile 的 W1/W3 到达，减少 MXU 气泡。

### 4.6 Token 预取重叠

**Token 大小**: bts × hidden_size × sizeof(t_dtype)
- MiMo: 16 × 6144 × 2 = **192 KB**
- Ling: 16 × 8192 × 2 = **256 KB**

**DMA 时间**: 192 KB / 3.28 TB/s = **0.06 μs** — 微不足道

**v2 的 token bubble**: 虽然 DMA 传输极快，但 DMA 的**启动/完成固定开销**（~2-5 μs）在每个 bts tile 都会发生。
v3 消除了这个固定开销——token 只在跨专家边界加载一次，且与 MXU 重叠。

---

## 5. 专家边界开销分析

### 5.1 v3 专家边界操作清单

专家 E → 专家 E+1 的边界操作：

| 操作 | 方向 | 大小 | 时间 | 阻塞? |
|------|------|------|------|-------|
| wait_writeback(x_slot) | — | — | 0 μs (drain e-2, 早已完成) | 否 |
| cast f32→bf16 | VREG/VPU | bts × H × 4 → bts × H × 2 | ~0.1 μs | 是(VPU) |
| start_writeback DMA | VMEM→HBM | bts × H × 2 = 192 KB | 0.06 μs DMA | 否(async) |
| start_a2a_gather | HBM→HBM | per-device sz | 变化 | 否(async) |
| start_fetch w1/w3/w2 tile1 | HBM→VMEM | 9.3 MB | 2.83 μs | 否(async) |
| wait_load_x(next_xs) | — | — | 0 μs (epilogue 中已完成) | 否 |
| wait_fetch_w1(slot0) | — | — | 0 μs (epilogue 中已完成) | 否 |
| wait_fetch_w3(slot0) | — | — | 0 μs (epilogue 中已完成) | 否 |

**有效 MXU 气泡**: 仅 ~0.1 μs（cast 操作），几乎为零。
所有 DMA 都是异步发起或在之前的 epilogue 中已开始。

### 5.2 v2 专家边界操作清单（对比）

| 操作 | 时间 | 阻塞? |
|------|------|-------|
| bts tile writeback: cast + DMA + wait | ~5-10 μs | **是** |
| A2A scatter wait (next expert) | 变化 | **是** |
| bts tile token load: DMA + wait | ~5-10 μs | **是** |
| Weight prologue: start + wait W1/W3 | ~3-5 μs | **是** |

**v2 MXU 气泡**: ~15-25 μs/专家（保守估算）

### 5.3 专家边界气泡对比（全层）

**MiMo V2 Pro** (12 local experts):
- v2: 12 × 15-25 = **180-300 μs** 气泡/层
- v3: 12 × 0.1 = **~1.2 μs** 气泡/层
- **改善: ~150-250× 减少专家边界气泡**

**Ling 2.6** (64 local experts):
- v2: 64 × 15-25 = **960-1600 μs** 气泡/层
- v3: 64 × 0.1 = **~6.4 μs** 气泡/层
- **改善: ~150-250× 减少专家边界气泡**

**注意**: 这是专家边界气泡的理论改善。实际性能还受 A2A 通信延迟、HBM 带宽利用率等因素影响。

---

## 6. 端到端性能模型

### 6.1 单 bt block 延迟分解

**v3（MiMo V2 Pro, 12 experts, fp8）**:

| 阶段 | 延迟估算 | 说明 |
|------|---------|------|
| Metadata (precomputed) | ~2-5 μs | JAX allreduce 结果加载 |
| A2A scatter batch | ~20-50 μs | 跨 32 设备通信 |
| Expert FFN ×12 | 12 × 12-20 = 144-240 μs | 4 bf tiles/expert |
| 专家边界 ×12 | ~1.2 μs | 几乎为零 |
| A2A gather wait | ~20-50 μs | 跨 32 设备通信 |
| Output accumulation | ~5-10 μs | weighted sum + store |
| **总计** | **~192-356 μs** | |

**v2（对比估算）**:

| 阶段 | 延迟估算 | 差异 |
|------|---------|------|
| Metadata | ~2-5 μs | 同 |
| A2A scatter | ~20-50 μs | 同 |
| Expert FFN ×12 | 12 × 12-20 = 144-240 μs | 同 |
| 专家边界 ×12 | ~180-300 μs | **v3 减少 180-300 μs** |
| A2A gather | ~20-50 μs | 同 |
| Output acc | ~5-10 μs | 同 |
| **总计** | **~371-655 μs** | |

**预期加速**: v3 相对 v2 在 MiMo 配置下约 **1.5-2.0×** 加速（主要来自消除专家边界气泡）。

### 6.2 带宽效率

**总 HBM 访问量（MiMo, 12 experts）**:
- 权重: 12 × 37.1 MB = 445 MB（fp8 + scale）
- Token I/O: ~2 MB（可忽略）
- A2A: ~1.5 MB
- **总计**: ~449 MB

**理论带宽利用率**:
- 449 MB / (192-356 μs) = **1.26-2.34 TB/s**
- TPU v7x 峰值: 3.28 TB/s
- **利用率: 38-71%**

v2 由于气泡导致相同数据量需要更长时间：
- 449 MB / (371-655 μs) = **0.69-1.21 TB/s**
- **利用率: 21-37%**

### 6.3 Ling 2.6 性能估算

**Ling 2.6 (64 experts, bf16, H=8192, I=2048, bf=256, num_bf=8)**:

单 bf tile:
- 权重 DMA: 3 × 8192 × 256 × 2 = 12 MB → 3.66 μs
- MXU: 2 × 3 × 16 × 8192 × 256 = 201M FLOPs → ~0.73 μs（理论）→ ~2-4 μs（实际）
- **DMA-bound**: 权重 DMA 大于 MXU 时间

单专家 (8 bf tiles):
- 权重 DMA 总量: 96 MB → 29.3 μs（如果完全流水化则隐藏在 MXU 后）
- MXU 总时间: 8 × 2-4 = 16-32 μs
- 受限于 max(DMA, MXU) = 16-32 μs（MXU bound 或接近平衡）

全层 (64 experts):
- v3: 64 × 16-32 + ~6 + ~40(A2A) + ~20(acc) = **1090-2114 μs**
- v2: 同 + 960-1600(边界气泡) = **2050-3714 μs**
- **预期加速: ~1.5-1.8×**

---

## 7. VMEM 占用分析

### 7.1 MiMo V2 Pro (fp8, bt=8, bf=512, bts=16)

| 缓冲区 | 形状 | 数据类型 | 大小 | v2 对应 | 变化 |
|--------|------|---------|------|---------|------|
| b_x_x2_vmem | (2, 16, 2, 3072) | bf16 | 384 KB | b_x_vmem (8, 2, 3072)=96KB | +288KB（双缓冲+更大bts） |
| b_w1_x2_vmem | (2, 2, 3072, 512) | fp8 | 6.0 MB | 同 | — |
| b_w3_x2_vmem | (2, 2, 3072, 512) | fp8 | 6.0 MB | 同 | — |
| b_w2_x2_vmem | (2, 2, 512, 3072) | fp8 | 6.0 MB | 同 | — |
| b_w1_scale_x2 | (2, 2, 24, 1, 512) | f32 | 384 KB | 同 | — |
| b_w3_scale_x2 | (2, 2, 24, 1, 512) | f32 | 384 KB | 同 | — |
| b_w2_scale_x2 | (2, 2, 4, 1, 3072) | f32 | 192 KB | 同 | — |
| b_y_acc_vmem | (16, 2, 3072) | f32 | 384 KB | (8, 2, 3072)=192KB | +192KB（更大bts） |
| **b_y_out_x2_vmem** | **(2, 16, 2, 3072)** | **bf16** | **384 KB** | **无（v2 用 b_y_stage）** | **新增** |
| b_gate_acc | (16, 512) | f32 | 32 KB | 同 | — |
| b_up_acc | (16, 512) | f32 | 32 KB | 同 | — |
| b_output_x2 | (2, 8, 6144) | bf16 | 192 KB | 同 | — |
| a2a_g_acc | (2, 8, 8, 2, 3072) | bf16 | 1.5 MB | 同 | — |
| topk + SMEM | — | — | ~100 KB | 同 | — |
| **v3 总计** | | | **~21.9 MB** | | |
| **v2 总计** | | | **~21.1 MB** | | |
| **差值** | | | **+0.8 MB** | | **可接受** |

**VMEM 利用率**: 21.9 / 64 = **34.2%**（vs v2 的 33.0%）

**注**: v3 增加了 `b_y_out_x2_vmem` 双缓冲输出，但由于 bts 很小（16），增量极小。
实际上 v3 还移除了 v2 的 `b_y_stage_vmem` (单缓冲) 换成了 `b_y_out_x2_vmem`(双缓冲)，
净增只有 384 KB - 96 KB = 288 KB。

### 7.2 Ling 2.6 (bf16, bt=32, bf=256, bts=32)

| 缓冲区 | 形状 | 大小 | v2 对应 | 变化 |
|--------|------|------|---------|------|
| b_x_x2_vmem | (2, 32, 2, 4096) | 1.0 MB | (32, 2, 4096)=512KB | +512KB |
| b_w1_x2_vmem | (2, 2, 4096, 256) | 8.0 MB | 同 | — |
| b_w3_x2_vmem | (2, 2, 4096, 256) | 8.0 MB | 同 | — |
| b_w2_x2_vmem | (2, 2, 256, 4096) | 8.0 MB | 同 | — |
| b_y_acc_vmem | (32, 2, 4096) | 1.0 MB | (32, 2, 4096)=1.0MB | — |
| b_y_out_x2_vmem | (2, 32, 2, 4096) | 1.0 MB | b_y_stage 512KB | +512KB |
| b_output_x2 | (2, 32, 8192) | 1.0 MB | 同 | — |
| a2a_g_acc | (2, 8, 16, 2, 4096) | 4.0 MB | 同 | — |
| 其他 | — | ~200 KB | 同 | — |
| **v3 总计** | | **~32.2 MB** | **v2: ~30.7 MB** | **+1.5 MB** |

**VMEM 利用率**: 32.2 / 64 = **50.3%**（vs v2 的 48.0%）

### 7.3 VMEM 余量

| 配置 | v3 VMEM 用量 | 剩余 | 可容纳 |
|------|-------------|------|--------|
| MiMo fp8 | 21.9 MB | 42.1 MB | 充裕：可扩大 bts 或增加更多缓冲 |
| Ling bf16 | 32.2 MB | 31.8 MB | 充裕：可容纳 shared expert 缓冲 |

---

## 8. 动态 Token 处理

### 8.1 Pad-to-bts 方案

每个专家的 token 数 `dyn_sz` 在运行时确定（0 到 a2a_max_tokens）。v3 采用：

1. **加载 bts 个 token**（pad 部分为 A2A buffer 中的未初始化数据）
2. **对 bts 个 token 执行完整 FFN**（包括 pad 部分的无效计算）
3. **写回 bts 个 token**（pad 部分的结果也写回）
4. **Output accumulation 阶段**只读取 `dyn_sz` 个有效 token

**正确性保证**: pad 部分的 FFN 输出虽然是垃圾值，但它们永远不会被 `acc_and_store_output` 使用——
该函数通过 `expert_offsets_x2_smem` 追踪的偏移量只读取有效 token。

### 8.2 计算浪费分析

**MiMo decode（bt=8, ep=32, top_k=8, E=384）**:
- local tokens: 8 (256 / 32)
- local experts: 12 (384 / 32)
- 平均 tokens/expert: 8 × 8 / 12 ≈ 5.3
- bts = 16（pad 到的目标）
- **计算浪费**: (16 - 5.3) / 16 = **66.9%**

看起来浪费很大，但关键观察：
1. **decode 是 bandwidth-bound**，不是 compute-bound（AI ≈ 10.8 vs 拐点 83.8）
2. 权重 DMA 才是瓶颈——权重不随 token 数变化
3. 额外的 pad token 计算仅消耗 MXU 时间（被 DMA 隐藏）
4. 只要 MXU 时间 ≤ DMA 时间，pad 的计算就是"免费的"

**验证**:
- 每 bf tile DMA 时间: 9.3 MB / 3.28 TB/s = 2.83 μs
- 每 bf tile MXU 时间 (bts=16): ~3-5 μs
- 每 bf tile MXU 时间 (实际5.3 tokens): ~1-2 μs
- bts=16 的 MXU 增量: ~1-3 μs（被 DMA 大部分隐藏）

**结论**: pad-to-bts 在 decode 场景下的实际性能损失极小（<5%），但大幅简化代码。

### 8.3 零 token 专家处理

当 `expert_sizes[e_id] == 0` 时：
- 跳过 FFN 计算
- 仍需发起 A2A gather（gather 空数据不影响正确性）
- 不需要 token 加载或 writeback
- 通过 `@pl.when(has_tokens)` 分支处理

---

## 9. bts 选取策略

### 9.1 约束条件

- `bts` 必须是 8 的倍数（VREG sublane 对齐）
- `bts` 必须能容纳最大可能的单专家 token 数
- `bts` 影响 VMEM 用量: `b_x_x2_vmem = 2 × bts × H × sizeof(dtype)`

### 9.2 最大 token 数估算

极端情况：所有 token 都路由到同一个专家。
- MiMo: 8 local tokens × 8 top_k = 64 token-expert pairs / 12 experts
  - 均匀: 5.3/expert，极端: 最多 64（如果所有 token 的所有 top_k 都选同一专家）
  - 实际上限: bt × top_k = 8 × 8 = 64（因为 bt 个 token 的 routing 决定了分布）
  - 但 a2a_max_tokens = align_to(bt × num_devices, bts) = align_to(8 × 32, bts)

- Ling: 32 local tokens × 8 top_k / 64 experts
  - 均匀: 4/expert，极端: 最多 256

### 9.3 推荐 bts 值

| 配置 | 推荐 bts | 理由 |
|------|---------|------|
| MiMo decode (bt=8) | 16 | 覆盖平均情况的 3× 安全余量 |
| Ling decode (bt=32) | 32 | 覆盖均匀分布，极端情况用 bts tiling |
| 通用 decode | `min(a2a_max_tokens, 64)` | 平衡 VMEM 占用与覆盖范围 |

**超出 bts 时的降级方案**: 如果某个专家的 dyn_sz > bts，当前 v3 设计会导致
超出部分的 token 被截断。**解决方案**：

1. **保守方案**: 设 bts = a2a_max_tokens（在 decode 下通常可接受）
2. **回退方案**: 添加 bts tile 循环作为 fallback（类似 v2，但仅在 dyn_sz > bts 时触发）
3. **实际方案**: 在 outer function 中验证 `a2a_max_tokens <= bts`，否则报错

v3 初版采用方案 3（保守且简单）。配置表中的 bts 必须 >= a2a_max_tokens。

---

## 10. 风险分析

### 10.1 技术风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| bts pad 超出 VMEM | 低 | 高 | 限制 bts ≤ 64；添加 VMEM budget 检查 |
| lax.fori_loop 开销 | 中 | 中 | 与 v2 相同（v2 也用 fori_loop） |
| fp8 精度问题 | 低 | 高 | 与 v2 使用相同的 direct_scaled_dot 路径 |
| A2A 通信延迟变化 | 中 | 中 | A2A 代码完全复用 v2 |
| drain e-2 时 y_out 未完成 | 极低 | 高 | multi_expert_pipeline 已验证此模式 |

### 10.2 性能风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| pad 计算非免费 | 如果 MXU 时间 > DMA 时间，pad 会增加延迟 | 监控 MXU/DMA 比例，必要时动态调 bts |
| 无 btc 分片 | bts > 128 时 GEMM shape 可能不友好 | decode 下 bts 通常 ≤ 64 |
| 无 cross-expert prefetch | v2 的 can_cross_expert_prefetch 被简化掉 | multi_expert_pipeline 的 epilogue 预取替代 |

### 10.3 功能风险

| 缺失特性 | 影响 | 计划 |
|----------|------|------|
| Shared expert | 无法运行需要 shared expert 的模型 | Phase 2 添加 |
| Prefill | 无法用于 prefill 阶段 | Phase 3 添加（需要 bts tiling） |
| Pipelined scatter | expert_buffer_count < local_num_experts 时降级 | 暂不支持，确保 budget 足够 |

---

## 11. 与 double_buffer_expert / multi_expert_pipeline 的映射关系

### 11.1 核心循环映射

```
multi_expert_pipeline            v3                           v2
────────────────────────         ────────────────────         ────────────────────
tokens_hbm[expert_idx]    →     a2a_s_x2_hbm[e_sem]    →     a2a_s_x2_hbm[e_sem]
output_hbm[expert_idx]    →     a2a_s_acc_x2_hbm[e_sem] →    a2a_s_acc_x2_hbm[e_sem]
b_x_x2_vmem[x_slot]      →     b_x_x2_vmem[x_slot]    →     b_x_vmem (单缓冲)
b_y_out_vmem[y_slot]      →     b_y_out_x2_vmem[y_slot] →    b_y_stage_vmem (单缓冲)
b_y_acc_vmem              →     b_y_acc_vmem            →     b_y_acc_vmem
compute_tile(x,w,first)   →     compute_tile(x,w,first) →    gate_up_btc + act_down_btc
start_writeback(e,y)      →     start_writeback(e_sem,y)→    bts_body 末尾同步写回
wait_writeback(y)         →     wait_writeback(y)       →    无（同步等待）
weight_sems[slot,ch]      →     weight_sems[slot,ch]    →    local_sems[slot,4-6]
x_sem[x_slot]             →     x_sem[x_slot]           →    x_stage_sem[0] (单)
y_out_sem[y_slot]         →     y_out_sem[y_slot]       →    y_store_sem[0] (单)
```

### 11.2 新增/保留的 v2 组件

v3 从 v2 保留以下组件（代码几乎不变）：

| 组件 | v2 位置 | 说明 |
|------|---------|------|
| FusedMoEBlockConfig | L42-82 | 数据类（可能简化字段） |
| all_reduce_metadata (precomputed) | L418-488 | 仅保留 `_copy_precomputed` 分支 |
| start_a2a_scatter_batch | L668-721 | 批量 scatter |
| wait_a2a_scatter_recv | L791-803 | 等待 scatter 接收 |
| wait_a2a_scatter_send_batch | L773-789 | 等待 scatter 发送 |
| start_a2a_gather | L818-858 | 发起 gather |
| wait_a2a_gather_recv_all | L879-894 | 等待所有 gather |
| wait_a2a_gather_send | L862-877 | 等待 gather 发送 |
| acc_and_store_output | L1390-1476 | 输出加权和 |
| start_send_bo / wait_store_output | L1480-1499 | 输出 DMA |
| jax_allreduce_metadata_by_bt | L1771-1800 | JAX 层 metadata |
| compute_local_expert_sizes | L1763-1768 | 本地专家大小 |
| Outer function structure | L1819-2161 | shard_map + pallas_call |

---

## 12. 实现验证方案

### 12.1 阶段性验证

| 阶段 | 验证内容 | 方法 |
|------|---------|------|
| 1. bf16 FFN only | 单专家 FFN 正确性 | 对比 `ref_moe`，disable_a2a=True |
| 2. bf16 + A2A | 多设备 EP 正确性 | 对比 `ref_moe`，全设备运行 |
| 3. fp8 FFN only | fp8 精度 | 对比 `ref_moe`，rel_err < 0.05 |
| 4. fp8 + A2A | 端到端 fp8 | 对比 v2 输出（bitwise 不要求，精度匹配） |
| 5. 性能 | 延迟对比 | benchmark_runner v2 vs v3 |

### 12.2 边界条件测试

- 零 token 专家（所有 token 跳过某些专家）
- 单 token 专家（dyn_sz=1，pad 到 bts）
- 全 token 集中到单专家（极端 pad 比例）
- num_bf=1（无双缓冲稳态）
- num_bf=2（最小双缓冲）

---

## 13. 总结

### 13.1 v3 vs v2 关键差异

| 维度 | v2 | v3 | 预期收益 |
|------|-----|-----|---------|
| Token 驻留 | 每 bts tile 重新加载 | 持久 + 跨专家双缓冲 | 消除 ~15-25 μs/expert 气泡 |
| Output writeback | bts tile 间同步 | 跨专家双缓冲(drain e-2) | 消除 writeback 阻塞 |
| W2 优先级 | priority=1 (bug) | priority=0 (deferred) | 更好的 DMA-compute overlap |
| Compute 路径 | 3 条 | 2 条 | 代码减半，编译器更易优化 |
| A2A 模式 | batch + pipelined | 仅 batch | 代码简化 |
| btc 分片 | 有 | 无 | 代码简化（decode 不需要） |
| 代码行数 | ~2200 | 目标 ~800-1000 | 2-3× 更少 |

### 13.2 预期性能

| 配置 | v2 估算 | v3 估算 | 加速 |
|------|---------|---------|------|
| MiMo fp8 decode 256tok | 370-650 μs | 190-360 μs | **1.5-2.0×** |
| Ling bf16 decode 256tok | 2050-3700 μs | 1090-2100 μs | **1.5-1.8×** |

加速主要来自**消除专家边界气泡**（180-1600 μs → ~1-6 μs）。
compute 和 bandwidth 受限部分不变（相同的 GEMM 形状和权重大小）。

### 13.3 后续路线

1. **Phase 1**: v3 初版（bf16 + fp8 + batch A2A + decode）— 本文档范围
2. **Phase 2**: 添加 shared expert（与路由专家交错执行）
3. **Phase 3**: 添加 prefill 支持（bts tile 循环 + btc 分片）
4. **Phase 4**: 性能调优（bts/bf 自动选择、cross-expert weight reuse）
