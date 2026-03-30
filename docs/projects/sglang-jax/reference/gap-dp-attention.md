---
title: 实现差距：DP Attention
---

# 实现差距：DP Attention（5D Mesh）

> tpu-inference 实现了基于 5D Mesh 的 DP Attention，解决 MoE 模型中 KV Head 数远小于 TP 度导致的 KV Cache 复制浪费问题。sglang-jax 当前仅使用 2D Mesh，无 DP Attention 支持。

---

## 一、差距总览

| 组件 | tpu-inference | sglang-jax |
|---|---|---|
| Mesh 维度 | 5D `("data", "attn_dp", "attn_dp_expert", "expert", "model")` | 2D `("data", "tensor")` |
| Attention/MLP 分片解耦 | ✅ 不同阶段使用不同分片轴 | ❌ 统一使用 `"tensor"` 轴 |
| DP Degree 自动计算 | ✅ 基于 KV heads 与 TP 的比例 | ❌ 无 |
| 多进程 DP 调度器 | ✅ 每个 DP rank 独立 Scheduler 进程 | ❌ 单进程 Scheduler |
| Prefix Cache 亲和性路由 | ✅ 探测所有 rank 的 cache 命中 | ❌ 无 |

---

## 二、问题背景

### 为什么需要 DP Attention

在 MoE 模型（如 DeepSeek V3，128 KV heads）中，当 TP 度远大于 KV heads 数时：

- 标准 TP 分片：每个设备分到 `128/TP` 个 KV heads
- 若 TP=256（多 slice），每设备仅 0.5 个 head → KV Cache 被完整复制到每个设备
- 浪费 HBM，限制可服务的并发请求数

**DP Attention 解决方案**：将部分 TP 度转为 Attention 的数据并行度，每个设备处理不同的 batch 子集但共享完整 KV heads。

---

## 三、tpu-inference 实现原理

### 3.1 5D Mesh 设计

```python
MESH_AXIS_NAMES = ("data", "attn_dp", "attn_dp_expert", "expert", "model")
```

**分片轴语义**：

| 轴名 | 映射维度 | 用途 |
|---|---|---|
| `ATTN_DATA` | `('data', 'attn_dp', 'attn_dp_expert')` | Attention Batch/Seq 分片 |
| `ATTN_HEAD` | `('model', 'expert')` | KV Head 分片 |
| `MLP_DATA` | `'data'` | MLP/MoE Batch 分片 |
| `MLP_TENSOR` | `('attn_dp', 'attn_dp_expert', 'model', 'expert')` | MLP 权重分片 |
| `EXPERT` | `('attn_dp', 'attn_dp_expert', 'expert', 'model')` | 专家权重分片 |

**核心思想**：Attention 和 MLP 阶段使用不同的分片策略。

- **Attention 阶段**：Batch 跨所有 DP 轴分片（`data × attn_dp × attn_dp_expert`），每个设备看到小 batch 但有独立 KV heads

- **MLP/MoE 阶段**：Batch 仅按 `data` 轴分片，权重跨所有剩余轴分片（完整模型并行容量）

**关键文件**: `tpu_inference/layers/common/sharding.py`

### 3.2 DP Degree 自动计算

```python
# sharding.py (lines 163-209)
num_kv_heads_per_device = max(1, (num_kv_heads * 2) / packing)

if tensor_parallelism > num_kv_heads_per_device:
    # KV heads 不够分 → 启用 DP Attention
    attn_dp = tensor_parallelism // num_kv_heads_per_device
    tensor_parallelism //= attn_dp

    # 专家并行度转移到 attn_dp_expert
    attn_dp_expert = expert_parallelism
    expert_parallelism = 1
```

**示例**：

- DeepSeek V3 (128 KV heads)，TP=16 → `kv_per_device=64`，TP < 64 → 不启用
- 某模型 (8 KV heads)，TP=16 → `kv_per_device=4`，TP > 4 → `attn_dp=4`, `tp=4`

### 3.3 Mesh 构建

```python
# tpu_runner.py (lines 324-394)
mesh_shape = (model_dp_size, attn_dp_size, attn_dp_expert_size, expert_size, tp_size)

# 单 Slice: 直接映射
# 多 Slice: data 轴拆分到 ICI + DCN
```

### 3.4 DP 调度器（多进程）

```text
DPScheduler
  ├── Worker 0 (独立进程, Pipe 通信)
  │     └── Scheduler + KV Cache (blocks // dp_size)
  ├── Worker 1 (独立进程)
  │     └── Scheduler + KV Cache
  └── Worker N (独立进程)
        └── Scheduler + KV Cache
```

**负载均衡** (`_find_best_rank_for_request`):

1. **Prefix Cache 亲和性**：探测所有 rank 的 prefix cache，分配到缓存最多的 rank
2. **最少负载回退**：无 cache 命中时分配到总 token 数最少的 rank
3. **Sticky 分配**：一旦分配，请求不迁移

**通信**：使用 `multiprocessing.Pipe`（避免 Queue 的 feeder thread GIL 竞争）+ `cloudpickle` 序列化。

**关键文件**: `tpu_inference/core/sched/dp_scheduler.py`

### 3.5 Runner 端 DP 输入准备

```python
# tpu_runner.py: _prepare_dp_input_metadata()
# 1. 按 rank 分组请求
# 2. Per-rank padding 到相同大小
# 3. Block table 重排：每个 rank 占据连续段
# 4. 所有输入标记 PartitionSpec(ATTN_DATA) 分片

input_ids     → P(ATTN_DATA)
positions     → P(ATTN_DATA)
seq_lens      → P(ATTN_DATA)
block_tables  → P(ATTN_DATA)
```

---

## 四、sglang-jax 现状

### 4.1 当前 Mesh

```python
# mesh_utils.py
mesh = jax.make_mesh((dp_size, tp_size), ("data", "tensor"))
```

仅支持 2D 分片。所有层（Attention + MLP）使用同一 `"tensor"` 轴。

### 4.2 当前分片策略

- Attention Q/K/V：沿 `"tensor"` 轴按 KV head 分片
- MLP 权重：沿 `"tensor"` 轴列/行分片
- MoE 专家：`shard_map` 在 `("data", "tensor")` 上

### 4.3 已有相关代码

- `server_args.py` 中有 `dp_attention` 字段引用（仅文档/metrics 中提及）
- `forward_batch_info.py` 中 `ForwardMode.IDLE` 注释提及 DP Attention

---

## 五、实现路线

### Phase 1: 5D Mesh 基础设施

**工作量**: ~3 天

1. **扩展 `mesh_utils.py`**：

   ```python
   # 新增 5D mesh 创建
   def create_5d_mesh(
       dp_size, attn_dp_size, attn_dp_expert_size,
       expert_size, tp_size
   ):
       return jax.make_mesh(
           (dp_size, attn_dp_size, attn_dp_expert_size, expert_size, tp_size),
           ("data", "attn_dp", "attn_dp_expert", "expert", "model")
       )
   ```

2. **定义分片轴别名**：

   ```python
   class ShardingAxisName:
       ATTN_DATA = ("data", "attn_dp", "attn_dp_expert")
       ATTN_HEAD = ("model", "expert")
       MLP_DATA = "data"
       MLP_TENSOR = ("attn_dp", "attn_dp_expert", "model", "expert")
       EXPERT = ("attn_dp", "attn_dp_expert", "expert", "model")
   ```

3. **DP Degree 自动计算逻辑**：基于 `num_kv_heads` 和配置的 `tensor_parallelism`，自动决定是否启用 DP Attention。

### Phase 2: 分片策略更新

**工作量**: ~3-5 天

1. **Attention 层**：
   - Q 分片：`P(ATTN_DATA, ATTN_HEAD, None)`
   - KV Cache 分片：`P(ATTN_DATA, None, ATTN_HEAD)`
   - 输入（input_ids, positions, seq_lens）：`P(ATTN_DATA)`

2. **MLP/Linear 层**：
   - 列并行：权重 `P(None, MLP_TENSOR)`
   - 行并行：权重 `P(MLP_TENSOR, None)`
   - 输入：`P(MLP_DATA, None)`

3. **MoE 层**：
   - EP 模式：专家权重 `P(EXPERT, None, None)`
   - TP 模式：专家权重 `P(None, MLP_TENSOR, None)`

4. **Attention ↔ MLP 过渡**：
   - Attention 输出 `P(ATTN_DATA, None)` → MLP 输入 `P(MLP_DATA, None)` 需要自动 reshape/allgather

### Phase 3: DP 调度器

**工作量**: ~5 天

1. **`DPScheduler` 类**：
   - 管理 N 个独立 Scheduler 实例
   - 多进程架构（每个 rank 一个进程，Pipe 通信）
   - 负载均衡策略：prefix cache 亲和性 + 最少负载回退

2. **`_prepare_dp_input_metadata()`**：
   - 按 rank 分组请求
   - Per-rank padding
   - Block table 重排
   - 输入分片约束

3. **集成到 Engine**：
   - `server_args.py` 添加 `--enable-dp-attention` 标志
   - 当 `tp > num_kv_heads` 时自动启用

### Phase 4: KV Cache 适配

**工作量**: ~2 天

- `MHATokenToKVPool` 分片从 `P("tensor")` 改为 `P(ATTN_DATA, ATTN_HEAD)`
- `ReqToTokenPool` 按 DP rank 分区
- `RadixCache` 按 DP rank 独立管理（每个 rank 独立 Radix Tree）

---

## 六、关键设计决策

### 6.1 渐进式 vs 一步到位

**建议渐进式**：先在 2D mesh 上实现 DP Attention 的调度器逻辑（多进程 + 负载均衡），再升级到 5D mesh。这样可以分别验证调度正确性和分片正确性。

### 6.2 多进程 vs 单进程

tpu-inference 使用多进程避免 GIL。sglang-jax 当前是单进程 Scheduler。

**建议**：先在单进程中实现 DP 逻辑（简单循环 N 个 Scheduler），验证正确性后再切换多进程架构。

### 6.3 与 Expert Parallel 的交互

5D mesh 中 `attn_dp_expert` 轴同时用于 DP Attention 和 Expert Parallel。在 DeepSeek V3 场景下：

```python
# 专家并行度转移到 attn_dp_expert
attn_dp_expert = expert_parallelism
expert_parallelism = 1
```

sglang-jax 当前的 EP 实现需要适配这个轴复用逻辑。

---

## 七、参考文件

### tpu-inference

| 文件 | 内容 |
|---|---|
| `layers/common/sharding.py` | 5D Mesh、分片轴、DP degree 计算 |
| `core/sched/dp_scheduler.py` | 多进程 DP 调度器 |
| `runner/tpu_runner.py:324-394` | Mesh 构建 |
| `runner/tpu_runner.py:1139-1203` | DP 输入准备 |

### sglang-jax 待修改文件

| 文件 | 修改内容 |
|---|---|
| `srt/utils/mesh_utils.py` | 5D Mesh 创建 + DP degree 计算 |
| `srt/layers/attention/*.py` | 分片策略更新 |
| `srt/layers/linear.py` | MLP 分片策略 |
| `srt/layers/moe.py`, `fused_moe.py` | Expert 分片轴适配 |
| `srt/managers/scheduler.py` | DP 调度器 |
| `srt/model_executor/model_runner.py` | DP 输入准备 |
| `srt/mem_cache/memory_pool.py` | KV Cache DP 分片 |
| `srt/mem_cache/radix_cache.py` | Per-rank Radix Tree |
