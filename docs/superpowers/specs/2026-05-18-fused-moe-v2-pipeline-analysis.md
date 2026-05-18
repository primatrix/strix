# RFC: Fused MoE v2 Kernel 流水排布与性能分析

**日期**: 2026-05-18
**状态**: 分析文档（为基于 double_buffer_expert 的重写提供基线参考）
**目标配置**:
- MiMo V2 Pro: E=384, H=6144, I=2048, top_k=8, fp8 e4m3, ep=32 (4x8x1)
- Ling 2.6 1T: E=256, H=8192, I=2048, top_k=8, bf16, ep=4 (2x2x1)

---

## 1. 内核整体结构

`_fused_moe_v2_impl.py` 实现了一个五层嵌套循环结构的 Pallas 内核：

```
sync_barrier()
for bt_id in 0..num_bt:                    ← 第1层：token 块循环
  all_reduce_metadata()
  for local_e_id in 0..local_num_experts:   ← 第2层：专家循环
    a2a_scatter_wait()
    expert_ffn():
      for bts_id in 0..num_bts_tiles:       ← 第3层：bts token 分片
        load_tokens()
        for bf_id in 0..num_bf:             ← 第4层：权重 tile 双缓冲
          for btc_id in 0..num_btc:         ← 第5层：compute tile
            gate_up_compute()
          act_down_compute()
        writeback()
    a2a_gather_start()
  acc_and_store_output()
```

### 1.1 各层级含义

| 层级 | 循环变量 | 含义 | 典型迭代数 |
|------|---------|------|-----------|
| L1: bt | bt_id | 将 local_num_tokens 分成 bt 大小的块 | MiMo: 1 (8/8), Ling: 2-8 |
| L2: expert | local_e_id | 遍历本设备承载的专家 | MiMo: 12, Ling: 64 |
| L3: bts | bts_id | 将专家的动态 token 数分成 bts 大小的片 | 动态，通常 1-4 |
| L4: bf | bf_id | 权重沿 intermediate dim 切分 | MiMo: 4 (2048/512), Ling: 8 (2048/256) |
| L5: btc | btc_id | 计算 tile，MXU 操作的最小粒度 | bts/btc |

### 1.2 两种 A2A 散播模式

内核根据 `expert_buffer_count` 选择两种 A2A 模式：

**Batch scatter**（`expert_buffer_count >= local_num_experts`）:
- 一次性为所有专家发起 scatter DMA
- 专家循环中直接 wait 各自的 recv
- 所有 gather 完成后统一 barrier

**Pipelined scatter**（`expert_buffer_count < local_num_experts`）:
- 每个专家独立发起 scatter DMA
- 下一个专家的 scatter 与当前专家的 FFN 重叠
- 通过 semaphore 环形缓冲区复用 `expert_buffer_count` 个槽

---

## 2. Expert FFN 流水排布

### 2.1 总体流水：每个 bts tile 的生命周期

```
┌─────────────────────── bts tile ───────────────────────┐
│                                                         │
│  Token Load    Weight Prefetch     Compute      Writeback
│  (sync)        (async)            (MXU)        (sync)
│                                                         │
│  ┌──────┐  ┌──────────────┐  ┌─────────────┐  ┌──────┐ │
│  │load x│→ │ W1[0],W3[0], │→ │ bf tile 0   │→ │ cast │ │
│  │ wait │  │ W2[0],       │  │ bf tile 1   │  │ store│ │
│  └──────┘  │ W1[1],W3[1], │  │   ...       │  │ wait │ │
│            │ W2[1]        │  │ bf tile N-1 │  └──────┘ │
│            └──────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 bf tile 双缓冲流水

非 decode 模式下，bf tile 使用 2 个权重槽（slot 0, slot 1）交替：

```
时间 →

Slot 0:  [W1/W3/W2 tile 0]──compute──┐  [W1/W3/W2 tile 2]──compute──┐ ...
                                       │                               │
Slot 1:        [W1/W3/W2 tile 1]──────compute──┐  [W1/W3/W2 tile 3]──compute──┐
                                                │                              │
MXU:     ████ gate/up[0] ████ act/down[0] ████ gate/up[1] ████ act/down[1] ████...

预取时序:
  bf tile 0: prologue 发起
  bf tile 1: prologue 发起
  bf tile N: bf tile N-2 的 act_down 完成后发起 (start_fetch at bf_id+2)
```

**关键等待点**（以 bf tile N 为例）:

```
                DMA 带宽         MXU 计算
                ──────────       ──────────
wait W1[N]  ← ┐
wait W3[N]  ← ┤ 同步点 ①
              │
              │ (dequant W1, W3 如果 fp8-非-direct)
              │
gate_up()   ← ┤                 ████████ MXU 活跃
              │
wait W2[N]  ← ┤ 同步点 ②        (W2 DMA 与 gate_up MXU 重叠)
              │
              │ (dequant W2 如果 fp8-非-direct)
              │
act_down()  ← ┤                 ████████ MXU 活跃
              │
prefetch    ← ┘ bf tile N+2
```

### 2.3 三条计算路径

内核根据配置选择不同的 gate/up 计算路径：

**路径 A: `direct_scaled_dot` + `bt <= 16`**（MiMo 小批量 decode）
```python
# W1 和 W3 分开等待，W3 DMA 与 gate MXU 重叠
wait_fetch_w1(slot)
for btc_id:  # gate-only loop
    for sg_id in 0..n_sg:  # 按 quant_block_k 分组
        gate += dot(x[btc, sg], w1[sg]) * scale1[sg]
wait_fetch_w3(slot)  # ← W3 DMA 与上面的 gate MXU 重叠
for btc_id:  # up-only loop
    for sg_id in 0..n_sg:
        up += dot(x[btc, sg], w3[sg]) * scale3[sg]
```

**路径 B: `direct_scaled_dot`（一般）**
```python
# W1/W3 同时等待，gate/up 在同一个 sg 循环中交替
wait_fetch_w1(slot)
wait_fetch_w3(slot)
for btc_id:
    for sg_id in 0..n_sg:
        gate += dot(x, w1[sg]) * s1
        up   += dot(x, w3[sg]) * s3
```

**路径 C: VMEM 反量化（fp8 非 direct_scaled_dot 或 bf16）**
```python
wait_fetch_w1(slot); wait_fetch_w3(slot)
dequant_w1(slot)  # fp8 → bf16 in VMEM
dequant_w3(slot)
for btc_id:
    gate += dot(x, w1_dq)
    up   += dot(x, w3_dq)
```

**路径 A 的 W2 也使用 direct_scaled_dot**:
```python
# act/down 也按 quant_block_k 分组
wait_fetch_w2(slot)
for btc_id:
    for sg_id in 0..n_sg2:  # n_sg2 = bf / quant_block_k
        act_slice = silu(gate[sg]) * up[sg]
        y_acc += dot(act_slice, w2[sg]) * scale2[sg]
```

### 2.4 cross-expert 预取

当 `can_cross_expert_prefetch=True`（条件：非 decode、direct_scaled_dot、fp8、num_bf >= 2 且为偶数）:

```
Expert E 的倒数第二个 bf tile (bf_id == num_bf - 2):
  ┌─────────────────────────────────────┐
  │ 常规操作：gate/up → act/down        │
  │                                     │
  │ 额外操作：start_prefetch_expert_bf0 │
  │   → 预取 Expert E+1 的 W1[0], W3[0], W2[0] │
  └─────────────────────────────────────┘

Expert E+1 的首个 bts tile:
  ┌─────────────────────────────────────┐
  │ 如果 bf0 已预取：跳过 bf tile 0 的  │
  │ weight DMA 发起（直接 wait+compute）│
  └─────────────────────────────────────┘
```

### 2.5 decode 模式特殊处理

Decode 模式下（`decode_mode=True`）:
- 只使用 1 个权重槽（`slot = 0`）
- 每个 bf tile 串行：发起 DMA → wait → compute → 下一个 tile 再发起
- 无双缓冲，无预取
- 无 cross-expert 预取

```
decode 模式 bf tile 流水（无重叠）:

bf 0: [DMA W1/W3/W2]──[wait]──[compute]
bf 1:                           [DMA W1/W3/W2]──[wait]──[compute]
bf 2:                                                    [DMA]──[wait]──[compute]
```

---

## 3. DMA 调度分析

### 3.1 优先级方案

| DMA 操作 | 默认优先级 | 实际使用优先级 | 说明 |
|----------|-----------|-------------|------|
| Token load (x) | — | 1 | 同步等待，优先级仅影响 DMA 控制器排队 |
| W1 fetch | 1 (default) | 1 | gate 权重 |
| W3 fetch | 1 (default) | 1 | up 权重 |
| W2 fetch | **0** (default) | **1** (被覆盖) | 代码中始终传 `priority=1` |
| Topk fetch | 0 (default) | 0 | 路由信息 |
| Output store | 0 (default) | 0 | 最终输出 |
| Cross-expert prefetch | — | 1 | 下一个专家的 bf0 权重 |

**重要发现**: v2 内核中 W2 的默认优先级虽定义为 0，但**所有调用点都显式传入 `priority=1`**，与设计文档中"低优先级 W2"的描述不符。实际上所有权重 DMA 都以相同优先级运行。

对比 `double_buffer_expert.py`: W2 在除首 tile 外的所有 tile 中使用 `priority=0`，真正实现了低优先级延迟加载。

### 3.2 semaphore 分配

```
local_sems[2, 10]:
  [slot, 0]:  topk_weights + topk_ids
  [slot, 1]:  metadata starts + precomputed copy
  [slot, 2]:  metadata sizes
  [slot, 3]:  d2e counts
  [slot, 4]:  W1 + W1_scale
  [slot, 5]:  W3 + W3_scale
  [slot, 6]:  W2 + W2_scale
  [slot, 7]:  output store
  [slot, 8]:  expert offsets copy
  [slot, 9]:  routing copy

x_stage_sem[1]:     token load/wait
y_store_sem[1]:     expert output writeback
send_x2_sems[E]:    A2A scatter send
recv_x2_sems[E]:    A2A scatter receive
gather_send_x2_sems[E]: A2A gather send
a2a_gather_sem:     A2A gather receive
a2a_acc_sems[1]:    output accumulation load
barrier_sem:        device sync barrier
```

### 3.3 阻塞点分析

以下是流水中 MXU 必然空闲的阻塞点：

| 阻塞点 | 位置 | 原因 | 影响 |
|--------|------|------|------|
| **Token load sync** | bts tile 开始 | `start() + wait()` 紧邻 | MXU 在等待 HBM→VMEM 传输 |
| **W1/W3 first wait** | bf tile 0 开始 | prologue 刚发起，可能未完成 | 首个 bf tile 的启动延迟 |
| **Writeback sync** | bts tile 结束 | `y_acc→y_stage→HBM` 同步写回 | MXU 在等待 VMEM→HBM 传输 |
| **A2A scatter wait** | expert 开始 | 等待 token 从其他设备到达 | 跨设备通信延迟 |
| **Metadata allreduce** | bt block 开始 | 设备间 routing 信息同步 | 全局同步延迟 |
| **sync_barrier** | 多处 | 设备同步 | 全局同步延迟 |

### 3.4 bts tile 间的 bubble

```
bts tile N:                                      bts tile N+1:
[...bf tiles...][writeback: cast+DMA+wait]  →  [token load: DMA+wait][bf tiles...]
                ├──── MXU 空闲 ────┤            ├── MXU 空闲 ──┤
                   ~cast + DMA延迟                    ~DMA延迟
```

**Token load 量**: `bts × hidden_size × 2 bytes`
- MiMo (bts=8, H=6144): 8 × 6144 × 2 = 96 KB → ~30 ns @ 3.28 TB/s
- Ling (bts=32, H=8192): 32 × 8192 × 2 = 512 KB → ~150 ns

**Writeback 量**: `bts × hidden_size × 2 bytes`（同 token load）+ fp32→bf16 cast

实际上这些传输量极小，bubble 主要来自 DMA 启动/完成的固定开销（~1-5 μs per DMA op）。

---

## 4. 计算强度分析

### 4.1 单专家 FLOPs

对于 N 个 token 经过一个专家的 SwiGLU FFN:

| 操作 | 形状 | FLOPs |
|------|------|-------|
| x @ W1 (gate) | (N, d) @ (d, f) | 2Ndf |
| x @ W3 (up) | (N, d) @ (d, f) | 2Ndf |
| silu(gate) * up | (N, f) | ~2Nf (忽略不计) |
| act @ W2 (down) | (N, f) @ (f, d) | 2Nfd |
| **总计** | | **6Ndf** |

### 4.2 各配置计算量

**MiMo V2 Pro** (d=6144, f=2048, fp8):

| 场景 | N/expert | FLOPs/expert | 本地专家数 | 总 FLOPs/bt | 总 FLOPs/layer |
|------|----------|-------------|-----------|------------|---------------|
| decode 256tok | ~5.3 avg | 400 MFLOPs | 12 | 4.8 GFLOPs | 4.8 GFLOPs |
| decode 512tok | ~10.7 avg | 800 MFLOPs | 12 | 9.6 GFLOPs | 9.6 GFLOPs |

（注: 256 tokens / 32 devices = 8 local tokens, 8 × top_k=8 / 12 experts ≈ 5.3 tokens/expert 均匀分布）

**Ling 2.6** (d=8192, f=2048, bf16):

| 场景 | N/expert | FLOPs/expert | 本地专家数 | 总 FLOPs/bt | 总 FLOPs/layer |
|------|----------|-------------|-----------|------------|---------------|
| decode 256tok | ~8 avg | 805 MFLOPs | 64 | 51.5 GFLOPs | 51.5 GFLOPs |

（256 tokens / 4 devices = 64 local, 64 × 8 / 64 experts = 8 tokens/expert）

### 4.3 MXU 利用率

单个 GEMM 的 MXU 效率取决于矩阵尺寸：

| GEMM | 形状 | TPU v7x 理论 MXU 时间 | 备注 |
|------|------|---------------------|------|
| (8, 6144) @ (6144, 512) | M=8, K=6144, N=512 | 50.3M FLOPs / 275 TFLOPs = 0.18 μs | 极度 M-受限 |
| (8, 128) @ (128, 512) | M=8, K=128, N=512 | 1.05M FLOPs / 275 TFLOPs = 0.004 μs | direct_scaled_dot 子块 |
| (256, 8192) @ (8192, 256) | M=256, K=8192, N=256 | 1.07G FLOPs / 275 TFLOPs = 3.9 μs | Ling 较大批量 |

**MiMo decode 的 MXU 利用率极低**: M=8 (btc=8) 意味着 MXU 128×128 阵列的每行只有 8 个有效 token，利用率上限 8/128 = 6.25%。

**direct_scaled_dot 路径额外开销**: 每个 GEMM 被分解为 `n_sg` 个子块（MiMo: 24 个），每个子块后需要一次 scale 乘法和累加。子块 FLOPs = 2 × 8 × 128 × 512 = 1.05M，远小于 MXU 的最小高效工作负载。

### 4.4 activation 计算开销

SwiGLU: `silu(gate) * up = gate * sigmoid(gate) * up`

FLOPs: ~5Nf（sigmoid + mul + mul）

占总 FLOPs 比例: 5Nf / 6Ndf = 5 / (6d)
- MiMo: 5/(6×6144) ≈ 0.014%
- Ling: 5/(6×8192) ≈ 0.010%

**结论**: 激活函数计算量完全可忽略。

### 4.5 direct_scaled_dot 的额外计算开销

fp8 direct_scaled_dot 模式下，每个 quant group 需要额外的 scale 乘法和累加：

| 操作 | 额外 FLOPs | 占 GEMM 比例 |
|------|-----------|-------------|
| FFN1 gate: 24 次 `gate += d1 * scale` | 2 × btc × bf × 24 = 2×8×512×24 = 196K | ~0.4% |
| FFN1 up: 同上 | 196K | ~0.4% |
| FFN2: 4 次 `y += d * scale` + 4 次 activation | 2×8×h_per_t×4 + 5×8×128×4 | ~0.05% |

**结论**: scale 开销可忽略不计，但循环启动开销（`fori_loop` unroll）可能显著。

---

## 5. 访存强度分析

### 5.1 单专家权重加载量

| 权重 | 形状 | bf16 大小 | fp8 大小 | fp8+scale 大小 |
|------|------|----------|---------|---------------|
| W1 | (d, f) | 2df | df | df + df/qbk × 4 |
| W3 | (d, f) | 2df | df | df + df/qbk × 4 |
| W2 | (f, d) | 2fd | fd | fd + fd/qbk × 4 |
| **总计** | | **6df** | **3df** | **3df(1 + 4/qbk)** |

**MiMo fp8** (d=6144, f=2048, qbk=128):
- 权重: 3 × 6144 × 2048 × 1 = 36 MB
- Scale: 3 × 6144 × 2048 / 128 × 4 = 1.125 MB
- **总计: 37.1 MB/expert**

**Ling bf16** (d=8192, f=2048):
- **总计: 3 × 8192 × 2048 × 2 = 96 MB/expert**

### 5.2 Token/Output DMA 量

每个 bts tile:
- Token load: `bts × d × 2` bytes (bf16 tokens)
- Output writeback: `bts × d × 2` bytes (bf16 output)

| 配置 | bts | Token load | Output WB | 占权重比 |
|------|-----|-----------|-----------|---------|
| MiMo | 8 | 96 KB | 96 KB | 0.5% |
| Ling | 8-32 | 128-512 KB | 128-512 KB | 0.3-1.1% |

**结论**: Token/Output DMA 量相比权重加载完全可忽略。

### 5.3 A2A 通信量

每个 bt block 的 A2A 总通信量：

| 操作 | 数据量 | 说明 |
|------|-------|------|
| Scatter | bt × top_k × d × 2 bytes | 每 token 发送给 top_k 个专家 |
| Gather | bt × top_k × d × 2 bytes | 每 token 从 top_k 个专家收回结果 |
| Metadata | ~padded_num_experts × num_devices × 4 bytes | routing 信息 |

MiMo: scatter + gather = 2 × 8 × 8 × 6144 × 2 = 1.5 MB/bt
Ling: scatter + gather = 2 × 32 × 8 × 8192 × 2 = 8 MB/bt（假设 bt=32）

### 5.4 全层 HBM 总访问量

**MiMo V2 Pro** (256 tokens, ep=32, 12 local experts):

| 项 | 读取 | 写入 | 总计 |
|----|------|------|------|
| 权重 (12 experts) | 12 × 37.1 = 445 MB | — | 445 MB |
| Tokens (scatter) | — | ~0.75 MB | 0.75 MB |
| Tokens (load to VMEM) | ~0.75 MB | — | 0.75 MB |
| Output (writeback) | — | ~0.75 MB | 0.75 MB |
| Output (gather+acc) | ~0.75 MB | ~0.1 MB | 0.85 MB |
| Topk weights/ids | ~16 KB | — | 16 KB |
| **总计** | | | **~448 MB** |

**Ling 2.6** (256 tokens, ep=4, 64 local experts):

| 项 | 读取 | 写入 | 总计 |
|----|------|------|------|
| 权重 (64 experts) | 64 × 96 = 6144 MB | — | 6144 MB |
| Tokens | ~4 MB | ~4 MB | 8 MB |
| Output | ~4 MB | ~4 MB | 8 MB |
| **总计** | | | **~6160 MB ≈ 6 GB** |

---

## 6. Roofline 分析

### 6.1 算术强度（Arithmetic Intensity）

```
AI = FLOPs / Bytes_transferred
```

| 配置 | 每专家 FLOPs | 每专家 Bytes | AI (FLOPs/Byte) |
|------|-------------|-------------|----------------|
| MiMo fp8, N=5.3 | 400 MFLOPs | 37.1 MB | **10.8** |
| MiMo fp8, N=8 | 604 MFLOPs | 37.1 MB | **16.3** |
| Ling bf16, N=2 | 201 MFLOPs | 96 MB | **2.1** |
| Ling bf16, N=8 | 805 MFLOPs | 96 MB | **8.4** |
| Ling bf16, N=32 | 3.22 GFLOPs | 96 MB | **33.5** |

### 6.2 TPU v7x Roofline

```
峰值计算:    275 TFLOPs (BF16) / 550 TOPs (INT8)
峰值带宽:    3.28 TB/s (HBM)
Roofline 拐点: 275T / 3.28T = 83.8 FLOPs/Byte (BF16)
             550T / 3.28T = 167.7 FLOPs/Byte (INT8)
```

### 6.3 瓶颈判定

| 配置 | AI | 拐点 | 比值 | 瓶颈 | 理论 MXU 利用率上限 |
|------|-----|------|------|------|-------------------|
| MiMo fp8 (N=5.3) | 10.8 | 83.8 | 0.13 | **内存** | 12.9% |
| MiMo fp8 (N=8) | 16.3 | 83.8 | 0.19 | **内存** | 19.4% |
| Ling bf16 (N=2) | 2.1 | 83.8 | 0.025 | **内存** | 2.5% |
| Ling bf16 (N=8) | 8.4 | 83.8 | 0.10 | **内存** | 10.0% |
| Ling bf16 (N=32) | 33.5 | 83.8 | 0.40 | **内存** | 40.0% |

**所有 decode 场景均深度内存受限**。fp8 通过将权重大小减半（96 MB → 37 MB），使 AI 提升约 4 倍，但仍远低于 roofline 拐点。

---

## 7. VMEM 占用分析

### 7.1 MiMo V2 Pro (fp8, bt=8, bf=512, direct_scaled_dot)

| 缓冲区 | 形状 | 数据类型 | 大小 |
|--------|------|---------|------|
| b_w1_x2_vmem | (2, 2, 3072, 512) | fp8 | 6.0 MB |
| b_w3_x2_vmem | (2, 2, 3072, 512) | fp8 | 6.0 MB |
| b_w2_x2_vmem | (2, 2, 512, 3072) | fp8 | 6.0 MB |
| b_w1_scale_x2 | (2, 2, 24, 1, 512) | f32 | 384 KB |
| b_w3_scale_x2 | (2, 2, 24, 1, 512) | f32 | 384 KB |
| b_w2_scale_x2 | (2, 2, 4, 1, 3072) | f32 | 192 KB |
| b_x_vmem | (8, 2, 3072) | bf16 | 96 KB |
| b_y_acc_vmem | (8, 2, 3072) | f32 | 192 KB |
| b_y_stage_vmem | (8, 2, 3072) | bf16 | 96 KB |
| b_gate_acc | (8, 512) | f32 | 16 KB |
| b_up_acc | (8, 512) | f32 | 16 KB |
| b_output_x2 | (2, 8, 6144) | bf16 | 192 KB |
| a2a_g_acc | (2, 8, 8, 2, 3072) | bf16 | 1.5 MB |
| 其他 (topk, sems) | — | — | ~50 KB |
| **总计** | | | **~21.1 MB** |

VMEM 利用率: 21.1 / 64 = **33%**，剩余 ~43 MB 未使用。

### 7.2 Ling 2.6 (bf16, bt=32, bf=256)

| 缓冲区 | 形状 | 数据类型 | 大小 |
|--------|------|---------|------|
| b_w1_x2_vmem | (2, 2, 4096, 256) | bf16 | 8.0 MB |
| b_w3_x2_vmem | (2, 2, 4096, 256) | bf16 | 8.0 MB |
| b_w2_x2_vmem | (2, 2, 256, 4096) | bf16 | 8.0 MB |
| b_x_vmem | (bts, 2, 4096) | bf16 | bts × 16 KB |
| b_y_acc_vmem | (bts, 2, 4096) | f32 | bts × 32 KB |
| b_output_x2 | (2, 32, 8192) | bf16 | 1.0 MB |
| a2a_g_acc | (2, 8, 16, 2, 4096) | bf16 | 4.0 MB |
| 其他 | — | — | ~200 KB |
| **总计 (bts=32)** | | | **~30.7 MB** |

VMEM 利用率: 30.7 / 64 = **48%**，剩余 ~33 MB。

### 7.3 与 double_buffer_expert.py 对比

`double_buffer_expert.py` (d=8192, f=2048, bf=256, bt=256):

| 缓冲区 | 形状 | 大小 |
|--------|------|------|
| b_w1_x2_vmem | (2, 8192, 256) | 8.0 MB |
| b_w3_x2_vmem | (2, 8192, 256) | 8.0 MB |
| b_w2_x2_vmem | (2, 256, 8192) | 8.0 MB |
| b_x_vmem | (256, 8192) | 4.0 MB |
| b_y_acc_vmem | (256, 8192) | 8.0 MB (fp32!) |
| b_y_out_vmem | (256, 8192) | 4.0 MB |
| **总计** | | **40.0 MB** |

VMEM 利用率: 40.0 / 64 = **62.5%**。

`double_buffer_expert` 的 VMEM 利用率更高，因为:
1. Token buffer 更大（持久驻留，256×8192 vs 小 bts 分片）
2. y_acc 使用 fp32（8 MB vs v2 的 bts-sized f32 buffer）
3. 无 A2A 相关缓冲区

---

## 8. 流水效率总评

### 8.1 有效 DMA-compute 重叠

| 重叠机制 | 描述 | 效果 |
|---------|------|------|
| W2 延迟等待 | W2 DMA 与 gate/up MXU 重叠 | ✅ 主要重叠源（但 priority=1 可能降低效果）|
| bf tile 双缓冲 | 下一 tile DMA 与当前 tile compute 重叠 | ✅ 稳态下权重加载完全隐藏 |
| W3 与 gate 重叠 | 路径 A (bt≤16): W3 DMA 与 gate MXU 重叠 | ✅ 小批量 decode 特定优化 |
| Cross-expert prefetch | 当前专家末尾预取下一专家 bf0 | ⚠️ 条件严格，需要多条件同时满足 |

### 8.2 流水 bubble 源

| Bubble 源 | 延迟 | 可避免性 |
|----------|------|---------|
| Token load (bts tile 间) | DMA 固定开销 + 传输 | 可通过 token 持久驻留消除 |
| Writeback (bts tile 间) | fp32→bf16 cast + DMA | 可通过延迟写回消除 |
| A2A scatter wait (expert 间) | 跨设备通信 | 不可避免，可与其他操作重叠 |
| Metadata allreduce (bt 间) | 设备同步 | 可预计算到 JAX 层 |
| sync_barrier (多处) | 全局同步 | 部分可消除 |
| Expert 间转换 | A2A + 权重加载 | Cross-expert prefetch 可部分隐藏 |

### 8.3 v2 vs double_buffer_expert 流水对比

| 特性 | v2 | double_buffer_expert |
|------|-----|---------------------|
| Token 驻留 | 每 bts tile 重新加载 | 持久驻留，加载一次 |
| y_acc 精度 | fp32（bts-sized） | fp32（全 bt-sized） |
| 权重双缓冲 | 有（2 slots） | 有（2 slots） |
| W2 优先级 | **实际 priority=1** | priority=0（除首 tile）|
| bf tile 预取深度 | bf+2 | bf+1 |
| 循环展开 | `lax.fori_loop` (dynamic) | Python for-loop (static) |
| Quant 支持 | fp8 + direct_scaled_dot | 仅 bf16 |
| 多专家支持 | 有（A2A scatter/gather） | 无 |
| 共享专家 | 有（与路由专家交错） | 无 |

### 8.4 v2 Expert FFN 核心循环与 double_buffer_expert 的关键差异

`double_buffer_expert` 的核心循环极其简洁：

```python
# Prologue
start_load_x(); start_fetch_w1(0,0); start_fetch_w3(0,0); start_fetch_w2(0,0,priority=1)
start_fetch_w1(1,1); start_fetch_w3(1,1); start_fetch_w2(1,1)  # priority=0
wait_load_x(); wait_fetch_w1(0); wait_fetch_w3(0)
compute_tile(slot=0, is_first_tile=True)

# Steady state
for tile in range(1, n_w - 1):
    slot = tile % 2
    start_fetch_{w1,w3,w2}(1-slot, tile+1)
    wait_fetch_{w1,w3}(slot)
    compute_tile(slot, is_first_tile=False)

# Epilogue
wait_fetch_{w1,w3}(last_slot)
compute_tile(last_slot, is_first_tile=False)

# Writeback
b_y_out = b_y_acc.astype(bf16); store(b_y_out)
```

v2 的 expert_ffn 在此基础上增加了：
1. 动态 token 数量（bts tile 循环）
2. 三条 compute 路径（direct_scaled_dot 分 quant group、VMEM 反量化、bf16 直接 dot）
3. btc 级计算分片
4. cross-expert 预取逻辑
5. A2A buffer 交互

---

## 9. 瓶颈总结与重写方向

### 9.1 核心瓶颈

1. **内存带宽受限**: 所有 decode 场景 AI 远低于 roofline 拐点（2-16 vs 84 FLOPs/Byte），性能完全由 HBM 带宽决定。

2. **MXU 利用率低**: 小批量 decode (bt≤16) 导致 GEMM 的 M 维极小（8-16），MXU 128×128 阵列利用率 ≤ 12.5%。

3. **W2 priority 未如设计**: v2 内核所有 W2 DMA 均 priority=1，未利用低优先级实现更好的 DMA 调度。

4. **Token 非持久**: bts tile 间的 token 重新加载引入 bubble（虽量级小但固定开销累积）。

5. **代码路径复杂**: 三条 compute 路径 × 两种 A2A 模式 × decode/non-decode 模式，增加了维护和优化难度。

### 9.2 基于 double_buffer_expert 重写的潜在收益

| 改进方向 | 预期收益 | 依据 |
|---------|---------|------|
| Token 持久驻留 | 消除 bts tile 间 token load bubble | double_buffer_expert 已验证 |
| 恢复 W2 低优先级 | 更好的 DMA-compute 重叠 | double_buffer_expert 已实现 |
| 简化代码路径 | 降低维护成本，利于编译器优化 | 单一路径 vs 三路径 |
| 静态循环展开 | 消除 `fori_loop` 动态开销 | Python for-loop 编译为固定 schedule |
| 增大 VMEM 利用 | 33% → 60%+，更大 token buffer | 当前有 ~40 MB 未使用 |

### 9.3 重写时需保留的 v2 特性

1. **EP 通信**: A2A scatter/gather 是多设备必需
2. **fp8 支持**: direct_scaled_dot 路径（MiMo 核心需求）
3. **共享专家**: 与路由专家交错执行
4. **动态 token 数**: 每专家 token 数运行时确定
5. **预计算 metadata**: JAX 层 all-reduce routing 信息

---

**下一步**: 设计基于 `double_buffer_expert` 核心循环的 v3 内核，将 EP 通信和 fp8 支持作为外层包装，保持核心计算路径的简洁性。
