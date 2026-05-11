# 融合 MoE 算子性能分析

**模型**: Ling 2.6 1T（万亿参数旗舰版）
**阶段**: Decode（自回归，每条序列 1 个 token）
**配置**: `fusedMoE_ling2.6.yaml`
**目标硬件**: TPU v7x（2x2x1 拓扑）

---

## 1. 模型架构概览

### MoE 配置（Ling 2.6 1T）
```
路由专家:                256
每 token 激活专家:       8
共享专家:                1（始终激活）
每 token 总专家数:       9（8 个路由 + 1 个共享）
隐藏维度 (d):            8192
专家中间维度:            2048
激活函数:                SwiGLU（silu）
精度:                    BF16
批次大小:                256 tokens
```

---

## 2. 专家计算逻辑

### 2.1 单个专家 FFN 结构

每个专家实现一个 **SwiGLU 门控 FFN**：

```
输入:   x ∈ ℝ^(d)           其中 d = 8192
门控:   W1 ∈ ℝ^(d × f)      其中 f = 2048
上升:   W3 ∈ ℝ^(d × f)
下降:   W2 ∈ ℝ^(f × d)

前向计算:
  gate_proj = x @ W1          # (d) @ (d × f) → (f)
  up_proj   = x @ W3          # (d) @ (d × f) → (f)
  act       = silu(gate_proj) * up_proj  # 逐元素乘法
  output    = act @ W2        # (f) @ (f × d) → (d)
```

### 2.2 共享专家

共享专家对**所有 token 始终激活**：
```
共享 W1: (d × f_se)  其中 f_se = 2048
共享 W3: (d × f_se)
共享 W2: (f_se × d)

共享输出与路由专家输出相加:
  final_output = routed_output + shared_output
```

---

## 3. 工作负载分析（Decode，bs=256）

### 3.1 单 Token 计算量

**1 个 token** 经过 **1 个专家**：

| 操作 | 形状 | FLOPs | 内存读取（BF16） |
|-----------|-------|-------|-------------------|
| x @ W1 | (8192) @ (8192×2048) | 2 × 8192 × 2048 = 33.6M | x: 16KB, W1: 32MB |
| x @ W3 | (8192) @ (8192×2048) | 2 × 8192 × 2048 = 33.6M | x: 16KB, W3: 32MB |
| silu + mul | (2048) | ~4K | gate: 4KB, up: 4KB |
| act @ W2 | (2048) @ (2048×8192) | 2 × 2048 × 8192 = 33.6M | act: 4KB, W2: 32MB |
| **总计** | | **~100.8M FLOPs** | **~96MB** |

### 3.2 批次级计算（256 tokens）

**路由专家**（每 token 激活 8 个）:
- 总专家调用次数: 256 tokens × 8 experts = 2048 次专家调用
- 总 FLOPs: 2048 × 100.8M = **206.4 GFLOPs**
- 总权重读取: 2048 × 96MB = **196.6 GB**（无复用，上界）
  - 这是"每 token 独立加载所选 8 个专家"的 naive 上界；
  - 若按专家聚合加载（每专家每层只读一次，参见 §4.1），下界约 **24 GB/层**（均匀路由下几乎所有 256 个专家都会被触及）；
  - 实际算子介于两者之间，取决于专家批处理粒度（§9.3）。

**共享专家**（所有 token 始终激活）:
- 批量 GEMM: (256 × 8192) @ (8192 × 2048) → (256 × 2048)
- FLOPs:
  - W1: 2 × 256 × 8192 × 2048 = 8.6 GFLOPs
  - W3: 2 × 256 × 8192 × 2048 = 8.6 GFLOPs
  - W2: 2 × 256 × 2048 × 8192 = 8.6 GFLOPs
  - 总计: **~25.8 GFLOPs**
- 总权重读取: 3 × 32MB = **96 MB**（每个权重矩阵复用 256 次）

**每层总计**:
- FLOPs: 206.4 + 25.8 = **~232 GFLOPs**
- 权重读取: 196.6 GB + 96 MB ≈ **~196.7 GB**

---

## 4. 内存分析

### 4.1 专家权重内存

每个专家:
```
W1: 8192 × 2048 × 2 bytes = 32 MB
W2: 2048 × 8192 × 2 bytes = 32 MB
W3: 8192 × 2048 × 2 bytes = 32 MB
单个专家总计: 96 MB
```

**全部 256 个专家**: 256 × 96 MB = **24 GB**
**共享专家**: **96 MB**
**每层总权重**: **~24 GB**

### 4.2 设备内存容量（TPU v7x）

```
每个芯片 HBM:          32 GB
2x2x1 拓扑总 HBM:      128 GB（4 芯片）
可用 HBM（~80%）:       ~102 GB
每层权重:               ~24 GB（占用 ~23% HBM）
```

### 4.3 权重加载时间

```
HBM 带宽（每芯片）:     820 GB/s
2x2x1 总带宽:          3.28 TB/s
每个 token 权重读取:    ~864 MB（9 × 96 MB；8 路由 + 1 共享）
加载时间:               ~864 MB / 3.28 TB/s ≈ 260 μs
```

---

## 5. 数据流分析

### 5.1 VMEM 约束与 Tile 设定

```
VMEM 总容量:                64 MB
权重单 Tile 最优大小:        4 MB / 8 MB
每次运算 token 数上限 (bt):  512
计算目标:                   y = (silu(x @ W1) * (x @ W3)) @ W2
```

**核心张量尺寸（bt = 512，BF16；命名遵循 §2）**：
```
x:      (512, 8192)     = 8 MB
W1:     (8192, 2048)    = 32 MB（门控，全量）
W3:     (8192, 2048)    = 32 MB（上升，全量）
W2:     (2048, 8192)    = 32 MB（下降，全量）
gate:   (512, 2048)     = 2 MB（x @ W1 的结果）
up:     (512, 2048)     = 2 MB（x @ W3 的结果）
act_up: (512, 2048)     = 2 MB（silu(gate) * up）
y:      (512, 8192)     = 8 MB（act_up @ W2，最终输出）
```

**三个权重沿 f = 2048 同步切分**：
- W1、W3：沿 **输出维 f** 切分，第 i 片形状 `(d, f_tile)`
- W2：沿 **reduction 维 f** 切分，第 i 片形状 `(f_tile, d)`，与 W1/W3 的第 i 片对齐

```
4 MB tile: f_tile = 256    → 2048/256 = 8 轮；W1/W3/W2 每片均 4 MB
8 MB tile: f_tile = 512    → 2048/512 = 4 轮；W1/W3/W2 每片均 8 MB
```

### 5.2 场景 A：SwiGLU 中间结果驻留 VREG + y 在 VMEM 累加

**伪代码**：
```
y_acc = zeros(bt, d)                        # 8 MB, 驻留 VMEM
for i in 0..N-1:
    load W1_i, W3_i, W2_i                   # 各一个 f_tile
    gate_i   = x @ W1_i                     # (bt, f_tile)  VREG
    up_i     = x @ W3_i                     # (bt, f_tile)  VREG
    act_up_i = silu(gate_i) * up_i          # (bt, f_tile)  VREG
    y_acc   += act_up_i @ W2_i              # (bt, d) 累加到 VMEM
return y_acc
```

**关键性质**：
- W1/W3 沿 f 切输出维，W2 沿 f 切 reduction 维——**对齐切分**使 `act_up_i` 能与 `W2_i` 直接相乘；
- 每轮 `act_up_i @ W2_i` 产出 y 的**部分累加**（partial sum），非完整切片，必须累加 N 轮；
- 累加器 `y_acc` (bt, d) = 8 MB 始终驻留 VMEM，无额外 HBM 往返。

**VMEM 占用**：

| 项 | 4 MB tile | 8 MB tile |
|-----|-----------|-----------|
| x（驻留） | 8 MB | 8 MB |
| W1 tile | 4 MB | 8 MB |
| W3 tile | 4 MB | 8 MB |
| W2 tile | 4 MB | 8 MB |
| y_acc（驻留累加器） | 8 MB | 8 MB |
| **小计（单缓冲）** | **28 MB** | **40 MB** |
| **小计（三权重双缓冲）** | **40 MB** | **64 MB**（恰满）|

### 5.3 场景 B：全部物化到 VMEM

权重、中间结果、输出全部整块驻留：
```
x       : 8 MB
W1 全量 : 32 MB
W3 全量 : 32 MB
W2 全量 : 32 MB
gate    : 2 MB
up      : 2 MB
act_up  : 2 MB
y       : 8 MB
---------
总计    : 118 MB
```
**118 MB > 64 MB VMEM 上限，严重超限**，权重必须分 tile。

### 5.4 折中：权重分 Tile、中间结果物化到 VMEM

| 项 | 4 MB tile | 8 MB tile |
|-----|-----------|-----------|
| x | 8 MB | 8 MB |
| W1 tile | 4 MB | 8 MB |
| W3 tile | 4 MB | 8 MB |
| W2 tile | 4 MB | 8 MB |
| gate（整块） | 2 MB | 2 MB |
| up（整块） | 2 MB | 2 MB |
| act_up（整块） | 2 MB | 2 MB |
| y_acc（驻留） | 8 MB | 8 MB |
| **总计** | **34 MB** | **46 MB** |

### 5.5 另一种方案：W2 沿 d 切（不与 f 对齐）

若将 W2 沿输出维 d 切片（而非 reduction 维 f），每轮可输出 y 的完整列切片 `(bt, d_tile)`，代价：
- 必须先**物化完整 act_up** `(bt, 2048)` = 2 MB 到 VMEM；
- SwiGLU 阶段与下降阶段串行化，无法流水。

适合 prefill（大 bt、高算术强度），不适合 decode 流水。

### 5.6 多批次调度：权重常驻 vs Token 常驻

当 `bt_total > bt_tile` 时（例如总 1024 tokens，每次只能处理 512），存在两种嵌套循环顺序。

**Strategy A：权重外循环，Token 内循环**（权重仅加载 1 次，维护多份 y_acc）
```
for i in 0..N-1:                     # 遍历权重 tile
    load W1_i, W3_i, W2_i            # 每 tile 只加载一次
    for b in 0..B-1:                 # 遍历 token 批
        act_up_b = silu(x_b @ W1_i) * (x_b @ W3_i)
        y_acc[b] += act_up_b @ W2_i  # B 份 y_acc 常驻 VMEM
flush y_acc[*]
```

**Strategy B：Token 外循环，权重内循环**（每批重新加载全部权重）
```
for b in 0..B-1:
    load x_b
    y_acc = 0
    for i in 0..N-1:
        load W1_i, W3_i, W2_i        # 每批重新加载 3 × 32 MB
        y_acc += ...
    flush y_acc
```

**Strategy A 的 x 放置方式又分两种**：
- **A-1 持久 x**：所有 B 批 x 常驻 VMEM（B × bt_tile × d × 2B）
- **A-2 轮换 x**：仅 1 批 x 在 VMEM（+ 预取缓冲），进入下一权重 tile 时**重读** x

x 是否持久驻留决定了 HBM 上 x 是读 1 次还是 N_w 次——因为在 A-2 中进入 tile `i+1` 时必须回到 batch 0，而 x_0 已被覆盖。

**HBM 流量对比（单专家，bt_total = 1024，B = 2，4 MB tile → N_w = 8）**：

| 项 | A-1 持久 x | A-2 轮换 x | Strategy B |
|-----|-----------|-----------|-----------|
| W1 | 32 × 1 | 32 × 1 | 32 × 2 = 64 |
| W3 | 32 × 1 | 32 × 1 | 32 × 2 = 64 |
| W2 | 32 × 1 | 32 × 1 | 32 × 2 = 64 |
| x  | 16 × 1 = 16 | 16 × N_w = **128** | 16 |
| y  | 16 | 16 | 16 |
| **合计** | **128 MB** | **240 MB** | **224 MB** |

（若换 8 MB tile → N_w = 4：A-2 的 x 读变为 64 MB，总计 176 MB）

**关键观察**：A-2 为节省 VMEM 付出 HBM 代价——x 被每个权重 tile 重读一次；当 `(N_w - 1) × bt_total × d × 2B > (B - 1) × W_total` 时，**A-2 甚至比 B 更差**。当前参数：`7 × 16 = 112 MB > 1 × 96 = 96 MB`，轮换 x 劣于重载权重。

**VMEM 占用（4 MB tile 单缓冲权重）**：

| 项 | A-1 持久 x | A-2 轮换 x（单缓冲） | A-2 轮换 x（双缓冲） | Strategy B |
|-----|-----------|-----------|-----------|-----------|
| x | 16 MB（两批常驻） | 8 MB（一批） | 16 MB（当前+预取） | 8 MB |
| W tile × 3 | 12 MB | 12 MB | 12 MB | 12 MB |
| y_acc | 16 MB（两份） | 16 MB（两份） | 16 MB（两份） | 8 MB（一份） |
| **合计** | **44 MB** | **36 MB** | **44 MB** | **28 MB** |

**一般化规则**：
- HBM(A-1) ≈ `W_total + bt_total × (d_in + d_out) × 2B`
- HBM(A-2) ≈ `W_total + N_w × bt_total × d × 2B + bt_total × d × 2B`
- HBM(B)  ≈ `B × W_total + bt_total × (d_in + d_out) × 2B`
- 只要 VMEM 装得下 **A-1**，它就是最优：`B × bt_tile × d × 2B`（x）+ `3 × tile`（W）+ `B × bt_tile × d × 2B`（y_acc）≤ VMEM。

**选择决策树**：
1. 若 A-1 VMEM 能装下 → **选 A-1**（当前 B=2 正好符合，44 MB < 64 MB）。
2. 若 A-1 装不下（B ↑ 导致 x + y_acc 超限）→ 在 A-2 和 B 中比较：
   - `(N_w - 1) × bt_total × d × 2B` 小于 `(B - 1) × W_total` → 选 A-2；
   - 否则选 B。
3. 极端大 B 时，B 更可控（y_acc 只需一份 8 MB，随时 flush）。

### 5.7 具体流水：单槽 x 滚动 + 双槽权重预取（B = 2）

在 §5.6 的基础上，进一步把"权重外循环"展开成具体的 DMA/MXU 流水，观察到：
- **x 只需单槽**——在 x1 完成 `x1 @ W1_a` / `x1 @ W3_a` 后（gate_1a、up_1a 入 VREG），x1 的 VMEM 地址即可立刻被 x2 覆盖；
- **权重双槽交替**（slot_0 装 tile a、c、e…，slot_1 装 tile b、d、f…），使下一权重 tile 的 DMA 与当前计算重叠；
- **W2 低优先级预取**——与 W1、W3 同槽但 DMA 优先级最低，只需在该 tile 的 `act_up @ W2` 阶段之前到达即可；
- **y_acc × B 持久**——跨所有权重 tile 累加。

#### 流水阶段

| 阶段 | 操作 | 槽位变化 |
|-----|-----|---------|
| 1 | load (x1, W1_a, W3_a, W2_a[低优先级]) | x_slot ← x1; W_slot_0 ← W_a |
| 2 | compute gate_1a = x1 @ W1_a, up_1a = x1 @ W3_a | 结果入 VREG |
| 3 | load (x2, W1_b, W3_b, W2_b) | x_slot ← x2（覆盖 x1）; W_slot_1 ← W_b |
| 4 | y1_acc ← silu(gate_1a) * up_1a @ W2_a | y1_acc 首次写入 VMEM |
| 5 | compute gate_2a = x2 @ W1_a, up_2a = x2 @ W3_a | 仍用 W_slot_0 |
| 6 | y2_acc ← silu(gate_2a) * up_2a @ W2_a | y2_acc 首次写入 VMEM |
| 7 | compute gate_2b = x2 @ W1_b, up_2b = x2 @ W3_b | 切到 W_slot_1 |
| 8 | load (x1, W1_c, W3_c, W2_c) — 复用 x_slot 与 W_slot_0 | x_slot ← x1; W_slot_0 ← W_c（覆盖 W_a） |
| 9 | y2_acc += silu(gate_2b) * up_2b @ W2_b | 累加到 y2_acc |
| 10+ | compute x1 @ W_b、y1_acc += ... @ W2_b；再 x1 @ W_c …… 循环至 N_w 完 | 稳态 |

每两个权重 tile 吞下一对 (x1, x2)；每两对 tile 中 x_slot、W_slot_0、W_slot_1 各被覆盖一次。

#### 阶段 VMEM 占用（4 MB tile）

tile 权重 bundle = W1 + W3 + W2 = 3 × 4 = 12 MB；x 单批 = 8 MB；y_acc 单份 = 8 MB。

| 阶段 | x | W_slot_0 | W_slot_1 | y1_acc | y2_acc | **合计** |
|-----|---|----------|----------|--------|--------|---------|
| 1 | 8 | 12 (a) | – | – | – | **20 MB** |
| 2 | 8 | 12 | – | – | – | **20 MB** |
| 3 | 8 | 12 | 12 (b) | – | – | **32 MB** |
| 4 | 8 | 12 | 12 | 8 | – | **40 MB** |
| 5 | 8 | 12 | 12 | 8 | – | **40 MB** |
| 6 | 8 | 12 | 12 | 8 | 8 | **48 MB** |
| 7 | 8 | 12 | 12 | 8 | 8 | **48 MB** |
| 8 | 8 | 12 (c) | 12 (b) | 8 | 8 | **48 MB** |
| 9 | 8 | 12 | 12 | 8 | 8 | **48 MB** |
| 稳态 | 8 | 12 | 12 | 8 | 8 | **48 MB** |

**峰值 48 MB，在 64 MB VMEM 之内**，可完整支持"双权重 tile 预取 + x 滚动 + 2 份 y_acc"。

#### 阶段 VMEM 占用（8 MB tile）

tile 权重 bundle = 3 × 8 = 24 MB；x 与 y_acc 同前。

| 阶段 | x | W_slot_0 | W_slot_1 | y1_acc | y2_acc | **合计** |
|-----|---|----------|----------|--------|--------|---------|
| 1 | 8 | 24 (a) | – | – | – | **32 MB** |
| 2 | 8 | 24 | – | – | – | **32 MB** |
| 3 | 8 | 24 | 24 (b) | – | – | **56 MB** |
| 4 | 8 | 24 | 24 | 8 | – | **64 MB** ⚠️ 满额 |
| 5 | 8 | 24 | 24 | 8 | – | **64 MB** |
| 6 | 8 | 24 | 24 | 8 | 8 | **72 MB** ❌ 超限 |

**8 MB tile 无法容纳此流水**。若坚持 8 MB tile，只能退到权重**单槽**（放弃下一 tile 预取）：
- 稳态：x (8) + W_current (24) + y1_acc (8) + y2_acc (8) = **48 MB** ✓，但失去权重 DMA 与计算重叠。

#### 结论

- **4 MB tile** 是此流水的最优 tile 尺寸：峰值 48 MB，可同时容纳两组权重 tile + x + 2 份 y_acc，权重 DMA 与 MXU 计算完全重叠；
- **8 MB tile** 只能做权重单槽调度（稳态 48 MB），失去预取窗口，HBM 延迟无法隐藏；
- 4 MB tile 的 N_w = 8 意味着更频繁的 DMA 启动开销，但 DMA 启动可被两组权重槽的交替启动分摊（每两个 tile 一次切换），实际影响小。

### 5.8 Decode 具体流水：单批 x（B = 1，bt ≤ 512）

§5.7 针对 `bt_total > bt_tile` 的多批次场景。**decode 阶段 bt 通常 ≤ 512**（单专家激活 token 数极少），可进一步简化：
- **不需要 x2**：只有一批 x，x1 整个流水期间持久驻留，**无需覆盖、无需重读**；
- **只需 1 份 y_acc**：VMEM 开销减半；
- 权重仍保持双槽交替，隐藏 DMA 延迟。

#### 流水阶段（显式 load / wait / compute）

DMA 是异步的：`load` 仅发起传输并占用 VMEM 槽，真正消费前需 `wait` 该张量的 DMA 完成。`W2` 标为低优先级——与 `W1/W3` 同步发起 DMA，但计算顺序上最晚用到，因此其 `wait` 可推迟到 SwiGLU 之后，让 `W2` 的 DMA 与 `x @ W1/W3` 的 MXU 计算完全重叠。

| # | 阶段 | 说明 |
|---|-----|-----|
| L1 | **load** (x1, W1_a, W3_a, W2_a[低优]) | 启动 DMA；x_slot ← x1（持久）; W_slot_0 ← W_a |
| W1 | **wait** (x1, W1_a, W3_a) | 同步：等 x1/W1_a/W3_a 到位即可开始计算（W2_a 还在传） |
| L2 | **load** (W1_b, W3_b, W2_b) | 启动下一 tile DMA；W_slot_1 ← W_b（与后续 compute 重叠） |
| C1 | **compute** gate_1a = x1 @ W1_a, up_1a = x1 @ W3_a | VREG |
| W2 | **wait** (W2_a) | 低优先级 DMA 完成（通常此时已到） |
| C2 | **compute** y_acc ← silu(gate_1a) \* up_1a @ W2_a | y_acc 首次写入 VMEM |
| W3 | **wait** (W1_b, W3_b) | 等下一 tile 的 gate/up 权重 |
| L3 | **load** (W1_c, W3_c, W2_c) — 复用 W_slot_0 | W_slot_0 ← W_c（覆盖 W_a） |
| C3 | **compute** gate_1b = x1 @ W1_b, up_1b = x1 @ W3_b | |
| W4 | **wait** (W2_b) | |
| C4 | **compute** y_acc += silu(gate_1b) \* up_1b @ W2_b | |
| W5 | **wait** (W1_c, W3_c) | |
| L4 | **load** (W1_d, W3_d, W2_d) — 复用 W_slot_1 | W_slot_1 ← W_d（覆盖 W_b） |
| C5 | **compute** gate_1c = x1 @ W1_c, up_1c = x1 @ W3_c | |
| W6 | **wait** (W2_c) | |
| C6 | **compute** y_acc += silu(gate_1c) \* up_1c @ W2_c | |
| …  | 稳态循环：`wait(W1/W3)` → `load` → `compute` → `wait(W2)` → `compute`，两槽交替 | 直至 N_w 个 tile 处理完 |

稳态节拍（每处理一个权重 tile）：`wait(W1/W3)` + `load(next-tile)` + `compute(gate/up)` + `wait(W2)` + `compute(y_acc)`。只要 DMA 速率跟得上 MXU 消耗，`wait` 不阻塞（time 为 0），权重加载完全被计算隐藏。

#### 阶段 VMEM 占用（4 MB tile，bt = 512）

x = 512 × 8192 × 2 = 8 MB；W bundle（W1+W3+W2）= 12 MB；y_acc = 8 MB。`wait` / `compute` 不改变 VMEM 占用，仅 `load` 使占用跳变。

| # | 阶段 | x | W_slot_0 | W_slot_1 | y_acc | **合计** |
|---|-----|---|----------|----------|-------|---------|
| L1 | load (x1, W_a[低优 W2_a]) | 8 | 12 (a) | – | – | **20 MB** |
| W1 | wait (x1, W1_a, W3_a) | 8 | 12 | – | – | 20 MB |
| L2 | load (W_b) | 8 | 12 | 12 (b) | – | **32 MB** |
| C1 | compute gate_1a, up_1a | 8 | 12 | 12 | – | 32 MB |
| W2 | wait (W2_a) | 8 | 12 | 12 | – | 32 MB |
| C2 | y_acc ← … @ W2_a | 8 | 12 | 12 | 8 | **40 MB** |
| W3 | wait (W1_b, W3_b) | 8 | 12 | 12 | 8 | 40 MB |
| L3 | load (W_c) — 复用 slot_0 | 8 | 12 (c) | 12 | 8 | 40 MB |
| C3 | compute gate_1b, up_1b | 8 | 12 | 12 | 8 | 40 MB |
| W4 | wait (W2_b) | 8 | 12 | 12 | 8 | 40 MB |
| C4 | y_acc += … @ W2_b | 8 | 12 | 12 | 8 | 40 MB |
| W5 | wait (W1_c, W3_c) | 8 | 12 | 12 | 8 | 40 MB |
| L4 | load (W_d) — 复用 slot_1 | 8 | 12 | 12 (d) | 8 | 40 MB |
| … | 稳态 | 8 | 12 | 12 | 8 | **40 MB** |

**峰值 40 MB，留 24 MB 余量**（可进一步容纳下一专家的权重 tile 预取）。

#### 阶段 VMEM 占用（8 MB tile，bt = 512）

W bundle = 24 MB。

| # | 阶段 | x | W_slot_0 | W_slot_1 | y_acc | **合计** |
|---|-----|---|----------|----------|-------|---------|
| L1 | load (x1, W_a) | 8 | 24 (a) | – | – | **32 MB** |
| W1 | wait (x1, W1_a, W3_a) | 8 | 24 | – | – | 32 MB |
| L2 | load (W_b) | 8 | 24 | 24 (b) | – | **56 MB** |
| C1 | compute gate_1a, up_1a | 8 | 24 | 24 | – | 56 MB |
| W2 | wait (W2_a) | 8 | 24 | 24 | – | 56 MB |
| C2 | y_acc ← … @ W2_a | 8 | 24 | 24 | 8 | **64 MB** ⚠️ 满额 |
| W3 … | wait/load 循环 | 8 | 24 | 24 | 8 | **64 MB** |
| 稳态 | | 8 | 24 | 24 | 8 | **64 MB** |

**峰值 64 MB 恰好填满 VMEM**，无余量跨专家预取。

#### 更典型场景：bt = 256（decode 默认 bs）

x = 4 MB, y_acc = 4 MB。

| tile | 峰值 VMEM | 说明 |
|------|----------|-----|
| 4 MB tile | 4 + 12 + 12 + 4 = **32 MB** | 留 32 MB 余量 |
| 8 MB tile | 4 + 24 + 24 + 4 = **56 MB** | 留 8 MB 余量 |

对 decode 默认 bs，**8 MB tile 也可行**（与 §5.7 的 B=2 场景相反——那时 8 MB tile 被 2 份 y_acc 挤爆）。

#### 选择建议

| 场景 | 推荐 tile | 峰值 VMEM | 理由 |
|------|----------|----------|-----|
| decode, bt ≤ 256 | **8 MB tile** | 56 MB | 更大 tile → 更少 DMA 启动、更长的 MXU 持续时间 |
| decode, bt = 512 | **4 MB tile** | 40 MB | 8 MB tile 会填满 VMEM（64 MB），留无余量做跨专家预取 |
| prefill/长序列, B ≥ 2 | **4 MB tile**（§5.7） | 48 MB | 2 份 y_acc + 双槽权重需要更小 tile |

核心结论：**tile 尺寸的选择取决于 bt 和 B**——bt 越小、B 越小，可用越大的 tile；bt 大或 B≥2 时回落到 4 MB tile。

### 5.9 结论

1. 三个权重**必须**沿 f 维度分 tile 流式加载（全量 = 96 MB，远超 VMEM）。
2. W1/W3 的输出维切分与 W2 的 reduction 维切分**对齐**，是 SwiGLU + 下降流水的前提。
3. W2 分片产出 y 的**部分和**，非完整切片；`y_acc` (bt, d) 必须作为 VMEM 中的累加器驻留。
4. 多批 token 场景下采用 **权重外循环 + y_acc 分批累加**（Strategy A），权重加载摊薄到所有批次。
5. **单槽 x 滚动 + 双槽权重交替**（§5.7）适用于 B ≥ 2；**x 持久 + 双槽权重**（§5.8）适用于 decode 单批。
6. **Tile 选择**：
   - decode bt ≤ 256：**8 MB tile**（VMEM 峰值 56 MB，DMA 启动最少）；
   - decode bt = 512：**4 MB tile**（峰值 40 MB，留余量跨专家预取）；
   - B ≥ 2：**4 MB tile**（峰值 48 MB，容纳 2 份 y_acc + 双槽权重）。

---

## 6. Roofline 分析

### 6.1 算术强度

```
单专家 AI: 100.8M FLOPs / 96 MB = 1.05 FLOPs/Byte
路由专家 AI（每 token）: 8 × 100.8M / (8 × 96MB) = 1.05 FLOPs/Byte
共享专家 AI: 25.8 GFLOPs / 96 MB = 269 FLOPs/Byte
加权平均 AI: ~1 FLOPs/Byte
```

### 6.2 Roofline 对比

```
TPU v7x 峰值计算:       275 TFLOPs（BF16）
TPU v7x 峰值 HBM 带宽:  3.28 TB/s
Roofline 拐点:          275T / 3.28T = 84 FLOPs/Byte
当前 AI:                ~1 FLOPs/Byte

结论: 严重受限于内存带宽（内存瓶颈）
理论最大利用率: 1/84 ≈ 1.2%
```

---

## 7. 延迟分解（专家部分）

### 7.1 每层延迟估算

```
权重加载（路由专家）:    ~400 μs
权重加载（共享专家）:    ~30 μs
计算（路由专家 GEMM）:   ~100 μs
计算（共享专家 GEMM）:   ~10 μs
激活函数:                ~5 μs
总计:                    ~545 μs
```

### 7.2 瓶颈识别

```
1. 权重加载:            ~79%（主要瓶颈）
2. 计算:                ~20%
3. 激活:                ~1%
```

---

## 8. MXU 利用率分析

### 8.1 计算利用率

```
每 GEMM 的理论 MXU 时间:  ~10 μs（33.6M FLOPs / 275T FLOPs/s 峰值）
每 GEMM 的实际 MXU 时间:  ~25 μs（考虑启动开销）
权重加载时间:             ~130 μs（32MB / 3.28TB/s）
MXU 空闲时间:             ~130 μs（等待权重加载）
MXU 利用率:               25/(130+25) ≈ 16%
```

### 8.2 端到端利用率

```
每专家总时间:    ~160 μs
MXU 活跃时间:    ~25 μs
端到端利用率:    25/160 ≈ 15.6%
```

---

## 9. 优化策略

### 9.1 权重预取

**当前**: 权重加载与计算串行执行
**方案**: 在当前专家计算时预取下一个专家的权重

```
无预取: load_W1 → compute_W1 → load_W2 → compute_W2 → ...
有预取: load_W1 → [load_W2 | compute_W1] → [load_W3 | compute_W2] → ...

收益: 隐藏 50-70% 的权重加载延迟
延迟降低: ~200-280 μs（~35-45%）
```

### 9.2 多专家预取

**当前**: 单专家处理
**方案**: 同时预取多个专家的权重

```
prefetch_distance = 2-4 个专家
需要: 额外的 HBM 带宽余量
收益: 更好地隐藏权重加载延迟
风险: HBM 带宽争用
```

### 9.3 专家批处理（权重复用）

**当前**: 每个 token 独立处理 8 个专家
**方案**: 将访问同一专家的 token 分组

```
排序 → 分组 → 批量 GEMM
如果 token 分布均匀: 256/256 ≈ 1 token/expert（无收益）
实际分布: 偏斜的 Zipf 分布 → 热门专家获得更多 token
收益: 根据分布偏斜程度，2-5× 更好的权重复用
```

### 9.4 INT8/INT4 量化

**当前**: BF16 权重（2 bytes/param）
**方案**: **INT8/INT4 量化**
- 权重大小: 96 MB → 48 MB（INT8）或 24 MB（INT4）
- 加载时间: 160 μs → 80 μs（INT8）或 40 μs（INT4）
- 收益: **~2-4× 加速**（如果反量化开销较低）

---

## 10. 与 Prefill 对比

| 指标 | Decode（bs=256） | Prefill（bs=1, seq=8192） |
|--------|-----------------|--------------------------|
| 每专家 token 数 | ~2-8 | ~32-128 |
| 权重复用 | ~2× | ~32-128× |
| 算术强度 | ~1 FLOPs/Byte | ~32-128 FLOPs/Byte |
| MXU 利用率 | ~15% | ~25-50% |
| 瓶颈 | 内存（权重） | 计算（均衡） |

**关键洞察**: Decode 由于每专家 token 数低而**根本上受限于内存**。Prefill 受益于高 token 数和更好的权重复用。

---

## 11. 结论

### 总结

Ling 2.6 1T 在 decode 模式（bs=256）下的融合 MoE 专家计算表现出：

1. **严重的内存瓶颈**: AI ≈ 1 FLOPs/Byte << 84（roofline 拐点）
2. **MXU 利用率极低**: ~15%（计算大部分时间等待权重加载）
3. **权重复用低**: 路由专家 ~2×（共享专家为 256×）

### 建议

**算子级别**:
1. 实现多专家预取（隐藏权重加载）
2. INT8/INT4 量化（减少权重传输量）
3. 专家批处理（分摊权重加载开销）

**架构级别**:
1. 增加批次大小（提高每专家 token 数）
2. 连续批处理（将 decode 与 prefill 混合）
3. 推测性解码（提高有效批次大小）

### 预期影响

优化后:
- **2-4× 加速** 来自量化 + 预取
- **MXU 利用率**: 15% → **30-50%**（仍然受限于内存，但有所改善）

---

**文档版本**: 1.1
**日期**: 2026-05-10
**作者**: 性能分析（AI 助手）
