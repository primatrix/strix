# Transformer 算子性能参考

> 本文基于 [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) 系列整理，为 Pallas kernel 开发者提供 Transformer 各组件的 FLOPs、内存、性能特征参考。

---

## 1. FLOPs 计算规则

### 1.1 基本规则

矩阵乘法 $A[N,P] \times B[P,M]$：**2NPM FLOPs**（P 次乘法 + P 次加法，遍历 N×M 输出元素）。

| 操作 | FLOPs | 数据传输 (bytes, bf16) |
|------|-------|----------------------|
| 点积 $x \cdot y$（长度 P） | 2P | 4P |
| 矩阵-向量 $Ax$ | 2NP | 2NP + 2P |
| 矩阵-矩阵 $AB$ | 2NPM | 2NP + 2PM + 2NM |

**核心洞察**：计算 $O(N^3)$，数据传输 $O(N^2)$ → 更大的 matmul 更容易 compute-bound。这是 Transformer 架构以 matmul 为主体的根本原因。

### 1.2 训练 FLOPs = 3× 推理

对于权重矩阵 B 的层 C = AB：

| 阶段 | 计算 | FLOPs |
|------|------|-------|
| 前向 | $C = AB$ | 2NPM |
| 反向（权重梯度） | $dB = A^T \cdot dC$ | 2NPM |
| 反向（激活梯度） | $dA = dC \cdot B^T$ | 2NPM |
| **总计** | | **6NPM** |

近似：每 token 训练 FLOPs ≈ **6 × 参数量**。

---

## 2. Shape 符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| B | batch（tokens） | — |
| T | 查询序列长度 | 4096-131072 |
| S | KV 序列长度 | = T (训练) |
| D | d_model | 4096-8192 |
| F | FFN 维度 | 4D (无 gate) 或 ~2.7D (有 gate) |
| N | Q 头数 | 32-128 |
| K | KV 头数 | 8-32 (GQA) |
| H | 头维度 | 128 |
| G | 组大小 N/K | 4-16 |
| L | 层数 | 32-128 |
| V | 词表大小 | 32K-256K |

---

## 3. 各组件 FLOPs 与参数

### 3.1 MLP Block（带 Gating Einsum）

```text
In[B,T,D] → W_in1[D,F] + W_in2[D,F] → σ(A1) * A2 → W_out[F,D] → Out[B,T,D]
```

| 操作 | 训练 FLOPs | 参数量 |
|------|-----------|--------|
| In × W_in1 | 6BTDF | DF |
| In × W_in2 | 6BTDF | DF |
| σ(A1) * A2（逐元素） | O(BTF) | — |
| A × W_out | 6BTDF | DF |
| **总计** | **≈18BTDF** | **3DF** |

### 3.2 Attention Block（GQA）

**Q/K/V/O 投影**：

| 操作 | 训练 FLOPs | 参数量 |
|------|-----------|--------|
| W_Q: [D, N, H] | 6BTDNH | DNH |
| W_K: [D, K, H] | 6BTDKH | DKH |
| W_V: [D, K, H] | 6BTDKH | DKH |
| W_O: [N, H, D] | 6BTDNH | NHD |
| **投影总计** | **12BTD(N+K)H** | **2D(N+K)H** |

**点积注意力**：

| 操作 | 训练 FLOPs |
|------|-----------|
| QK^T | 6BTSNH |
| softmax | O(BTSN) |
| Score × V | 6BTSNH |
| **注意力总计** | **≈12BTSNH = 12BT²NH** |

> 因果掩码使有效 FLOPs 减半，但实现这个减半"实际上需要使用 attention kernel"（如 Flash Attention）。

### 3.3 每层汇总

| 组件 | 参数/层 | 训练 FLOPs/层 |
|------|--------|-------------|
| MLP | 3DF | 18BTDF |
| Attention | 2D(N+K)H | 12BTD(N+K)H + 12BT²NH |
| LayerNorm | D | O(BTD) |
| 词表（全模型） | DV | 12BTDV |

---

## 4. 性能特征分析

### 4.1 注意力 vs MLP FLOPs 占比

假设 F=4D, D=NH, N=K：

$$\frac{\text{attention}}{\text{MLP}} = \frac{T}{8D}$$

| D | 注意力主导阈值 |
|---|-------------|
| 4096 | T > 32K |
| 8192 | T > 64K |

**结论**：对于大模型，注意力的二次复杂度在合理序列长度下并非主要瓶颈。

### 4.2 各组件 Bound 类型

| 组件 | 训练 | 推理 Prefill | 推理生成 |
|------|------|------------|---------|
| MLP matmul | compute-bound | compute-bound | memory-bound (batch 小) |
| QKV 投影 | compute-bound | compute-bound | memory-bound |
| 注意力 QK^T, SV | compute-bound (T > 480) | compute-bound | **永远 memory-bound** |
| LayerNorm | memory-bound | memory-bound | memory-bound |

---

## 5. Flash Attention

### 5.1 核心思想

避免 materialize 完整的 $[B, T, S, N]$ 注意力矩阵，改为分块计算并维护运行统计量：

| 统计量 | 含义 |
|--------|------|
| M | 运行中的 $q \cdot k$ 最大值 |
| L | 运行中的 $\sum_i \exp(q \cdot k_i - M)$ |
| O | 运行中的注意力输出 |

### 5.2 两个局部 softmax 的合并

$$L^\text{combined} = \exp(M^1 - M^\text{max}) \cdot L^1 + \exp(M^2 - M^\text{max}) \cdot L^2$$

其中 $M^\text{max} = \max(M^1, M^2)$。

### 5.3 TPU 上的优势

- Q 块驻留 VMEM（on-chip SRAM），KV 块从 HBM 流式加载
- 提升算术强度：避免将 $O(T^2)$ 的注意力矩阵写回 HBM

### 5.4 反向传播关键恒等式

softmax 梯度需要：

$$S_{ij} \cdot_j dS_{ij} = dO_{id} \cdot_d O_{id}$$

这将沿序列维度的收缩转换为沿特征维度的局部收缩，使得 **ring attention 和序列分片成为可能**。

---

## 6. KV Cache

### 6.1 尺寸计算

$$\text{KV cache} = 2 \times \text{bytes\_per\_float} \times H \times K \times L \times S$$

| 模型 | L | K | H | 每 token (int8) | 8192 序列 (bf16) |
|------|---|---|---|----------------|-----------------|
| LLaMA-2 13B | 40 | 40 | 128 | 400 kB | 6.7 GB |
| LLaMA-3 70B | 80 | 8 | 128 | 160 kB | 2.6 GB (int8) |

> "仅 4 个 KV cache 就超过了模型参数的内存占用！" — KV cache 在推理中常常是内存的主要消耗者。

### 6.2 推理延迟公式

$$T_\text{step} = \underbrace{\frac{BS \times \text{KV\_size}}{W_\text{hbm}}}_{\text{Attention (始终 BW-bound)}} + \underbrace{\max\left(\frac{2 \times BS \times \text{params}}{C},\quad \frac{\text{param\_bytes}}{W_\text{hbm}}\right)}_{\text{MLP (可能 compute-bound)}}$$

### 6.3 优化策略

| 策略 | 效果 |
|------|------|
| GQA/MQA | 减少 KV 头数 → 缓存缩小 |
| Local Attention 层 | 限制上下文窗口 → 部分层缓存缩小 |
| 量化 (int8/int4) | 减少每 token 字节数 |
| Paged Attention | OS 式页表管理 → 消除 padding 浪费 |
| Ragged HBM Reads | 只读取非 padding 部分 |

---

## 7. Gradient Checkpointing（Rematerialization）

### 7.1 无 checkpointing 的内存

保存每层 ~20 个中间节点：$2 \times 20 \times B \times T \times D \times L$ bytes (bf16)。

对于 BT=4M, L=64, D=8192：**~84 TB**。

### 7.2 策略对比

| 策略 | 额外 FLOPs | 内存节省 |
|------|-----------|---------|
| Block remat（仅存层输入） | ~2ND → 总 8ND | 最大（仅存 L 个检查点） |
| Big matmuls only（存 7 个 matmul 输出） | 4BT²NH + O(BTD+BTF) | 中等 |

---

## 8. MoE 性能分析

### 8.1 Compute-bound 条件

E 个专家，每 token 激活 k 个，int8 权重 + bf16 计算：

$$B > 120 \times \frac{E}{k}$$

| E/k | 临界 batch |
|-----|-----------|
| 8 (E=16, k=2) | 960 |
| 32 (E=256, k=8) | 3,840 |

### 8.2 通信开销

每个 MoE 块引入 2 次 AllToAll（token 路由），每次开销为同等 AllGather 的 1/4。

---

## 9. 推理 Serving 架构

### 9.1 Disaggregated Serving（分离式推理）

```text
请求 → Prefill Server → KV Cache → Generate Server → tokens
         (compute-bound)              (memory-bound)
```

- Prefill 和 Generate 使用不同的 TPU 集群
- 各自独立的 sharding 策略和扩缩容
- KV cache 通过网络传输（驱动 KV cache 压缩的动力）

### 9.2 Continuous Batching

编译两个函数：

- **Prefill function**：处理变长上下文
- **Generate function**：同时推进所有活跃请求

Orchestrator 管理槽位的插入和淘汰。

### 9.3 Prefix Caching

利用自回归特性：相同前缀产生相同 KV cache。

- 存储在 LRU trie 结构中
- 优先使用 prefill 服务器的空闲 HBM
- 也可使用 host DRAM（~450 GiB on 8×v5e，远大于 128 GiB HBM）

---

## 10. 参考链接

- [How to Scale Your Model — Ch4: Transformers](https://jax-ml.github.io/scaling-book/transformers)
- [How to Scale Your Model — Ch5: Training](https://jax-ml.github.io/scaling-book/training)
- [How to Scale Your Model — Ch7: Inference](https://jax-ml.github.io/scaling-book/inference)
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
