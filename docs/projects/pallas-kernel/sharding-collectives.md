# Sharding 与集合通信参考

> 本文基于 [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) 系列整理，为 Pallas kernel 开发者提供多芯片场景下的 sharding 和通信原语参考。

---

## 1. Sharding 标记法

### 1.1 基础标记

数组分布在 **device mesh**（加速器的 2D/3D 网格）上，各轴命名为 X, Y, Z。

| 标记 | 含义 |
|------|------|
| $A[I_X, J_Y]$ | I 维沿 X 轴分片，J 维沿 Y 轴分片 |
| $A[I_{XY}, J]$ | I 维沿 X 和 Y 两轴分片（展平） |
| $A[I, J]$ | 完全复制（无下标 = 不分片） |
| $A[I_X, J_X]$ | **禁止**：同一轴不能分片两个维度 |
| $\{U_X\}$ | 沿 X 轴 unreduced（部分和） |

**局部 shape** = 全局 shape ÷ 各分片轴的设备数。

### 1.2 JAX 代码

```python
import jax

mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

def P(*args):
    return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))

# In[8, 2048] 沿 X 分 batch, 沿 Y 分 D → 每设备 [2, 1024]
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
```

---

## 2. 集合通信原语

### 2.1 运行时间公式

| 操作 | 签名 | 运行时间 |
|------|------|---------|
| **AllGather** | $[A_X, B] \to [A, B]$ | $V / W_\text{ici}$ |
| **ReduceScatter** | $[A, B]\{U_X\} \to [A_X, B]$ | $V / W_\text{ici}$ |
| **AllReduce** | $[A, B]\{U_X\} \to [A, B]$ | $2V / W_\text{ici}$ |
| **AllToAll** | $[A, B_X] \to [A_X, B]$ | $V / (4 W_\text{ici})$ |

其中 $V$ = 全局数组字节数，$W_\text{ici}$ = 双向 ICI 带宽。

> **关键洞察**：AllGather 和 ReduceScatter 的开销 **与设备数无关**，仅受链路带宽限制。多轴 AllGather 带宽按轴数倍增：$T = V / (W_\text{ici} \times N_\text{axes})$。

### 2.2 延迟受限区域

当数据量很小时，通信受每跳延迟（~1μs on v5e）限制而非带宽：

$$T = \max\left(\frac{T_\text{min} \times |X|}{2},\quad \frac{V}{W_\text{ici}}\right)$$

TPU v5e 上约 **45 kB** 以下为延迟受限。~10 MB 时达到 ~95% 峰值带宽。

### 2.3 反向传播中的对偶关系

| 前向 | 反向 |
|------|------|
| AllGather | ReduceScatter |
| ReduceScatter | AllGather |

这是数学上的转置关系，在设计训练 kernel 的 `custom_vjp` 时很重要。

---

## 3. Sharded MatMul 四种情况

### Case 1：收缩维不分片 → 无通信

$$A[I_X, J] \cdot B[J, K_Y] \rightarrow C[I_X, K_Y]$$

每个设备独立完成局部 matmul，输出自然继承输入的分片。

### Case 2：单个输入的收缩维分片 → AllGather

$$A[I, J_X] \xrightarrow{\text{AllGather}_X} A[I, J] \quad\Rightarrow\quad A[I, J] \cdot B[J, K] \rightarrow C[I, K]$$

先 AllGather 恢复完整收缩维，再做局部 matmul。

### Case 3：两个输入的收缩维沿同轴分片 → ReduceScatter / AllReduce

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K]\{U_X\}$$

局部 matmul 产生部分和，然后：

- **AllReduce**：$C[I, K]\{U_X\} \to C[I, K]$（结果复制到所有设备）
- **ReduceScatter**：$C[I, K]\{U_X\} \to C[I, K_X]$（结果分片，更高效）

### Case 4：两个输入的非收缩维沿同轴分片 → 先 AllGather

$$A[I_X, J] \cdot B[J, K_X] \quad\text{(无效，必须先 AllGather 一个输入)}$$

---

## 4. 并行策略速查

### 4.1 四种策略的 sharding 模式

| 策略 | 前向公式 | 关键特征 |
|------|---------|---------|
| **DP** | $\text{In}[B_X, D] \cdot W[D, F]$ | 参数复制；反向 AllReduce 梯度（可异步） |
| **FSDP** | $\text{In}[B_X, D] \cdot W[D_X, F]$ | 参数分片；前向 AllGather 权重（可 prefetch） |
| **TP** | $\text{In}[B, D_Y] \cdot W[D, F_Y]$ | 激活分片；通信在 **关键路径** 上 |
| **FSDP+TP** | $\text{In}[B_X, D_Y] \cdot W[D_X, F_Y]$ | 混合；FSDP 移动权重，TP 移动激活 |

### 4.2 Compute-bound 条件

| 策略 | 条件 | TPU v5p 值 |
|------|------|-----------|
| DP / FSDP | $B/X > C/(W_\text{ici} \times M_X)$ | ≥ 850/芯片（3 轴） |
| TP | $Y < F \times M_Y / (C/W_\text{ici})$ | $Y < F/2550$ |
| FSDP+TP | $B/N > (C/W_\text{ici})^2/(M_X M_Y F)$ | ~100/芯片 (F=30K) |
| 跨 pod DP | $B/\text{slice} > C/W_\text{dcn}$ | ≥ 73,440/pod |

### 4.3 FSDP+TP 最优分配

$$X_\text{opt} = \sqrt{\frac{B \times M_X}{F \times M_Y} \times N}$$

**FSDP 移动权重（大小与 batch 无关），TP 移动激活（大小与 F 无关）**。

---

## 5. Collective Matmul（通信与计算重叠）

### 5.1 核心思想

将 blocking AllGather + matmul 拆分为流水线：每步做一个 shard 的 matmul，同时用 `ppermute` 循环传递下一个 shard。

```python
# 伪代码：AllGather-fused matmul
def collective_matmul(lhs, rhs):
    axis_size = jax.lax.axis_size('Y')
    accum = jnp.zeros(output_shape)

    for i in range(axis_size):
        # 取当前 shard 对应的 rhs 块
        rhs_chunk = dynamic_slice(rhs, chunk_offset)
        accum += lhs @ rhs_chunk
        # 同时将 lhs 传给下一个设备
        lhs = ppermute(lhs, circular_shift)

    return accum
```

### 5.2 性能提升

实测（TPU v5e 8 芯片）：

| 方法 | 耗时 |
|------|------|
| Naive AllGather + matmul | 311 μs |
| Collective matmul | 244 μs |
| 无分片基线 | 224 μs |

Collective matmul 将 AllGather 的通信时间几乎完全隐藏在计算中。

---

## 6. 推理中的 sharding 要点

### 6.1 推理不能用 FSDP

FSDP 通过 ICI 搬运权重，比直接从 HBM 读取慢很多。**训练结束后务必关闭 FSDP**，否则推理性能退化数量级。

### 6.2 推理只用 Model Parallelism

生成阶段只有 1 个 token/步，无法通过 batch 分片提速。仅 model parallelism（沿 D/F 分片权重）有效，通过增加聚合 HBM 带宽来降低延迟。

### 6.3 ICI 限制

$$Y_\text{max} = \frac{F}{B \times \beta}, \quad \beta = W_\text{hbm} / W_\text{ici} \approx 8 \text{ (v5e/v6e)}$$

batch 越小，可用的 model parallelism 度越高（因为激活通信量也越小）。

---

## 7. 对 Pallas Kernel 开发的启示

1. **单芯片 kernel 无需处理 sharding**：XLA 编译器自动在 kernel 外插入通信。kernel 内部只处理局部 tile。

2. **设计 Grid 时考虑 sharding 后的 shape**：kernel 的输入 shape 是分片后的局部 shape，Grid 维度应据此设计。

3. **AllReduce/ReduceScatter 的选择影响 kernel 输出**：
   - 如果后续需要 AllReduce，kernel 可输出完整维度
   - 如果用 ReduceScatter，kernel 可只计算局部分片（更高效）

4. **关注 Scalar ALU 瓶颈**：跨设备通信通常由 XLA 管理，但 kernel 内部的索引计算（Scalar ALU）可能成为瓶颈，尤其在通信与计算重叠的 pipeline 中。

5. **KV cache sharding 影响 attention kernel 设计**：推理时 KV cache 可能沿 batch 维度分片，attention kernel 需处理 AllToAll 通信前后的 reshape。

---

## 8. 参考链接

- [How to Scale Your Model — Ch3: Sharded Matmuls](https://jax-ml.github.io/scaling-book/sharding)
- [How to Scale Your Model — Ch5: Training](https://jax-ml.github.io/scaling-book/training)
- [How to Scale Your Model — Ch7: Inference](https://jax-ml.github.io/scaling-book/inference)
- [JAX Distributed Arrays](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
