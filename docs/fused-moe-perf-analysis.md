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
- 总权重读取: 2048 × 96MB = **196.6 GB**（无复用）

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
每个 token 权重读取:    ~1.5 GB（8 个专家 + 共享）
加载时间:               ~1.5 GB / 3.28 TB/s ≈ 460 μs
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

**HBM 流量对比（单专家，bt_total = 1024，B = 2）**：

| 项 | Strategy A | Strategy B |
|-----|-----------|-----------|
| W1 | 32 × 1 = 32 MB | 32 × 2 = 64 MB |
| W3 | 32 × 1 = 32 MB | 32 × 2 = 64 MB |
| W2 | 32 × 1 = 32 MB | 32 × 2 = 64 MB |
| x  | 16 MB | 16 MB |
| y  | 16 MB | 16 MB |
| **合计** | **128 MB** | **224 MB** |

**A 节省 96 MB，HBM 流量下降 ~43%**。

**VMEM 占用对比（4 MB tile）**：

| 项 | Strategy A | Strategy B |
|-----|-----------|-----------|
| x | 16 MB（两批） | 8 MB（一批） |
| W tile × 3（单缓冲） | 12 MB | 12 MB |
| y_acc | 16 MB（两份） | 8 MB（一份） |
| **单缓冲合计** | **44 MB** | **28 MB** |
| **双缓冲合计** | **56 MB** | **40 MB** |

**一般化规则**：
- HBM(Strategy A) ≈ `W_total + bt_total × (d_in + d_out) × 2B`
- HBM(Strategy B) ≈ `W_total × ⌈bt_total / bt_tile⌉ + bt_total × (d_in + d_out) × 2B`
- 只要权重体积大于单批 I/O（`W_total > bt_tile × (d_in + d_out) × 2B`，MoE 几乎总成立），**Strategy A 更省**。

**何时不得不用 B**：
- bt_total ↑ 到 ~4096 以上时，多份 y_acc 加 x 超出 VMEM（例如 bt_total=8192 时 y_acc 需 128 MB）；
- 8 MB tile + 双缓冲 + 2 份 y_acc：16 + 48 + 16 = 80 MB，超 VMEM 上限。

### 5.7 结论

1. 三个权重**必须**沿 f 维度分 tile 流式加载（全量 = 96 MB，远超 VMEM）。
2. W1/W3 的输出维切分与 W2 的 reduction 维切分**对齐**，是 SwiGLU + 下降流水的前提。
3. W2 分片产出 y 的**部分和**，非完整切片；`y_acc` (bt, d) 必须作为 VMEM 中的累加器驻留。
4. 多批 token 场景下采用 **权重外循环 + y_acc 分批累加**（Strategy A），权重加载摊薄到所有批次。
5. **推荐**：4 MB tile + 三权重双缓冲 + VREG 流式 SwiGLU + y_acc 累加 ≈ 40 MB（单批）/ 56 MB（双批），留余量用于下一专家权重预取或共享专家权重驻留。

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
| MXU 利用率 | ~0.5% | ~25-50% |
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
