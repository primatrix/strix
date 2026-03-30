---
title: 实现差距：PD Disaggregation
---

# 实现差距：PD Disaggregation

> tpu-inference 支持两种 Prefill-Decode 分离模式（进程内 Local 和跨进程 Multi-host），sglang-jax 仅有 stub 代码。

---

## 一、差距总览

| 组件 | tpu-inference | sglang-jax |
|---|---|---|
| Local Disagg（进程内） | ✅ `_DisaggOrchestrator` + 三类线程 | ❌ 无 |
| Multi-host Disagg（跨进程） | ✅ JAX P2P + ZMQ + Proxy Server | ❌ 无 |
| KV Cache 跨 Mesh 传输 | ✅ `jax.device_put` / `experimental_reshard` | ❌ 无 |
| Pallas DMA KV 复制 | ✅ `multi_layer_copy` 异步 HBM-to-HBM | ❌ 无 |
| 设备分配 | ✅ `PREFILL_SLICES` / `DECODE_SLICES` | ❌ 无 |
| 调度器端 KV Connector | ✅ `TPUConnectorScheduler` / `Worker` | ❌ `process_batch_result_disagg_prefill` stub |

---

## 二、tpu-inference 实现详解

### 2.1 Mode A: 进程内分离 (Local Disagg)

**适用场景**：单机多 chip（如 TPU v5e 8 chip → 4 prefill + 4 decode）。

```text
┌───────────────────────────────────────────────────┐
│                   单一进程                          │
│                                                   │
│  ┌──────────┐   Transfer Queue  ┌──────────┐      │
│  │ Prefill  │ ───────────────→  │  Decode  │      │
│  │ Engine   │   jax.device_put  │  Engine  │      │
│  │ (4 chips)│   / reshard       │ (4 chips)│      │
│  └──────────┘                   └──────────┘      │
│       ↑                              ↑            │
│  _prefill()                    _decode()          │
│  线程池                         线程池              │
│                                                   │
│            _DisaggOrchestrator                     │
│     (管理 prefill/transfer/decode 线程)              │
└───────────────────────────────────────────────────┘
```

#### 2.1.1 设备分配 (`DisaggExecutor`)

通过环境变量将 TPU 芯片分为 prefill 和 decode 两组：

```python
# disagg_executor.py: 设备分配
class DisaggExecutor:
    def _init_executor(self, ...):
        # 从环境变量解析设备分片
        prefill_slices = get_prefill_slices()  # e.g., "4" 或 "2x2"
        decode_slices = get_decode_slices()    # e.g., "4" 或 "2x4"

        # 创建独立的 Device Mesh
        prefill_devices = jax.devices()[:num_prefill]
        decode_devices = jax.devices()[num_prefill:]

        prefill_mesh = jax.sharding.Mesh(prefill_devices, axis_names)
        decode_mesh = jax.sharding.Mesh(decode_devices, axis_names)

        # 每个 mesh 上创建独立的 Engine 实例
        self.prefill_engine = Engine(mesh=prefill_mesh, ...)
        self.decode_engines = [Engine(mesh=decode_mesh, ...) for ...]
```

Slice 解析支持 1D 和 2D 格式：

```python
# disagg_utils.py
def _parse_slices(slices_str: str) -> tuple[int, ...]:
    """解析 '4' → (4,) 或 '2x4' → (2, 4)"""
    return tuple(int(x) for x in slices_str.split("x"))

def get_prefill_slices() -> tuple[int, ...]:
    return _parse_slices(os.environ.get("PREFILL_SLICES", "4"))

def get_decode_slices() -> tuple[int, ...]:
    return _parse_slices(os.environ.get("DECODE_SLICES", "4"))
```

配置示例：

```bash
# 1D 分配：8 chips → 4 prefill + 4 decode
PREFILL_SLICES=4 DECODE_SLICES=4

# 2D 分配：跨 host
PREFILL_SLICES=2x2 DECODE_SLICES=2x4
```

#### 2.1.2 编排器 (`_DisaggOrchestrator`)

编排器管理三类线程，通过有界队列协调数据流：

```python
# core_tpu.py: _DisaggOrchestrator 初始化
class _DisaggOrchestrator:
    def __init__(self, prefill_engine, decode_engines, scheduler):
        self.prefill_engine = prefill_engine
        self.decode_engines = decode_engines

        # 有界队列：控制内存使用
        self.transfer_backlog = queue.Queue(maxsize=4)  # prefill → transfer

        # Decode backlog 大小动态计算
        max_kv_per_request = estimate_max_kv_size(model_config)
        decode_backlog_size = remaining_hbm // max_kv_per_request
        self.decode_backlogs = [
            queue.Queue(maxsize=decode_backlog_size)
            for _ in range(len(decode_engines))
        ]

        # 启动线程（使用 JetThread — 崩溃即 SIGKILL 进程）
        self.prefill_threads = [
            JetThread(target=self._prefill, daemon=True)
            for _ in range(num_prefill_threads)
        ]
        self.transfer_thread = JetThread(target=self._transfer, daemon=True)
        self.decode_threads = [
            JetThread(target=self._decode, args=(i,), daemon=True)
            for i in range(len(decode_engines))
        ]
```

#### 2.1.3 Prefill 线程

```python
# core_tpu.py: _prefill 线程
def _prefill(self):
    """轮询 prefill 调度器，运行 prefill，提取 KV 放入 transfer backlog"""
    while True:
        # 1. 从调度器获取 batch
        batch = self.scheduler.get_next_prefill_batch()
        if batch is None:
            time.sleep(0.001)
            continue

        # 2. 在 prefill mesh 上运行 prefill
        with self.prefill_engine.mesh:
            output = self.prefill_engine.forward(batch)

        # 3. 提取 KV Cache（按 block_id 选取）
        kv_cache = self.prefill_engine.get_kv_cache_for_block_ids(
            batch.block_ids
        )

        # 4. 放入 transfer backlog（有界队列，满时阻塞）
        self.transfer_backlog.put(TransferItem(
            request=batch,
            kv_cache=kv_cache,
            output=output,
        ))
```

#### 2.1.4 Transfer 线程

```python
# core_tpu.py: _transfer 线程
def _transfer(self):
    """从 transfer backlog 取 KV，重分片到 decode mesh，放入 decode backlog"""
    while True:
        # 1. 从 transfer backlog 取数据
        item = self.transfer_backlog.get()

        # 2. 跨 mesh 传输 KV Cache
        decode_sharding = self.decode_engines[target_idx].kv_sharding
        kv_decode = jax.device_put(item.kv_cache, decode_sharding)
        # 或使用 experimental_reshard 进行更高效的重分片：
        # kv_decode = jax.experimental.reshard(
        #     item.kv_cache, decode_sharding
        # )

        # 3. 选择活跃请求最少的 decode engine
        target_idx = min(
            range(len(self.decode_engines)),
            key=lambda i: self.decode_engines[i].active_request_count()
        )

        # 4. 放入对应 decode engine 的 backlog
        self.decode_backlogs[target_idx].put(TransferItem(
            request=item.request,
            kv_cache=kv_decode,
            output=item.output,
        ))
```

#### 2.1.5 Decode 线程

```python
# core_tpu.py: _decode 线程
def _decode(self, engine_idx):
    """消费 decode backlog，分配 KV blocks，开始解码"""
    engine = self.decode_engines[engine_idx]
    while True:
        # 1. 从 decode backlog 取数据
        item = self.decode_backlogs[engine_idx].get()

        # 2. 在 decode engine 分配新的 KV block slots
        slots = engine.allocate_slots(item.request)

        # 3. 将 KV Cache 写入 decode engine 的 KV Pool
        engine.insert_request_with_kv_cache(
            request=item.request,
            kv_cache=item.kv_cache,
            slots=slots,
        )

        # 4. 标记为 RUNNING，开始解码
        item.request.status = RequestStatus.RUNNING
```

#### 2.1.6 线程安全机制

```python
# core_tpu.py: JetThread — 子线程崩溃即终止整个进程
class JetThread(threading.Thread):
    """线程的 uncaught exception 会触发 SIGKILL。
    避免子线程悄悄崩溃而主线程不知情的问题。"""
    def run(self):
        try:
            super().run()
        except Exception as e:
            logger.fatal(f"JetThread {self.name} crashed: {e}")
            os.kill(os.getpid(), signal.SIGKILL)
```

### 2.2 Mode B: 跨进程分离 (Multi-host Disagg)

**适用场景**：跨机器部署（独立的 Prefill Server 和 Decode Server）。

#### 2.2.1 三层通信架构

```text
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: HTTP (Proxy Server)                                    │
│                                                                 │
│  Client ──→ Proxy ──→ Prefill (max_tokens=1)                   │
│                   │     ↓ 提取 kv_transfer_params               │
│                   └──→ Decode (stream=True, 附带 transfer_params)│
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: JAX P2P Transfer (数据平面)                             │
│                                                                 │
│  Prefill Worker:                                                │
│    select_from_kv_caches() → D2H (可选) → await_pull(uuid, kv)  │
│                                                                 │
│  Decode Worker:                                                 │
│    connect(remote_addr) → pull(uuid, kv_spec)                   │
│    → scatter_kv_slices() → 写入本地 KV Pool                     │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: ZMQ Side Channel (通知平面)                             │
│                                                                 │
│  Decode 拉取完成 → ZMQ DEALER → Prefill ZMQ ROUTER              │
│  → Prefill 释放 KV 缓冲区                                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.2 P2P Transfer Server 启动

Decode 端启动 JAX P2P Transfer Server，Prefill 端连接：

```python
# tpu_connector.py: TPUConnectorWorker
class TPUConnectorWorker:
    def _maybe_start_p2p_server(self):
        """Decode 端启动 P2P Transfer Server"""
        if self.is_decode:
            port = int(os.environ.get("TPU_KV_TRANSFER_PORT", 9100))
            self.p2p_server = jax.experimental.transfer.start_transfer_server(
                port=port,
                num_channels=int(os.environ.get(
                    "TPU_KV_TRANSFER_CHANNEL_NUMBER", 8
                )),
            )
            logger.info(f"Started P2P transfer server on port {port}")

    def _connect_to_prefill(self, remote_addr: str):
        """Decode 端连接 Prefill 的 P2P Server"""
        self.p2p_client = jax.experimental.transfer.connect(
            remote_addr,
            num_channels=int(os.environ.get(
                "TPU_KV_TRANSFER_CHANNEL_NUMBER", 8
            )),
        )
```

#### 2.2.3 Prefill 端：准备 KV 并等待拉取

```python
# tpu_connector.py: Prefill 端 — 准备 KV 数据并等待 Decode 拉取
def _prepare_kv_and_wait(self, request_id: str, block_ids: list[int]):
    """Prefill 完成后：选取 KV → 可选 D2H → 等待 Decode 拉取"""

    # 1. 从 KV Cache Pool 中选取指定 block 的 KV
    kv_tensors = self.model_runner.select_from_kv_caches(block_ids)
    # kv_tensors: list of (key, value) pairs, one per layer

    # 2. 可选：Device-to-Host 传输（减少设备内存占用）
    if os.environ.get("TPU_ENABLE_D2H_TRANSFER", "true") == "true":
        kv_tensors = jax.device_get(kv_tensors)

    # 3. 生成唯一传输 UUID
    transfer_uuid = str(uuid.uuid4())

    # 4. 注册到 P2P server，等待 Decode 端拉取
    timeout = int(os.environ.get("TPU_P2P_WAIT_PULL_TIMEOUT", 30))
    self.p2p_server.await_pull(
        uuid=transfer_uuid,
        tensors=kv_tensors,
        timeout_seconds=timeout,
    )

    # 5. 返回 transfer params 给调度器（通过 HTTP 传递给 Decode）
    return {
        "transfer_uuid": transfer_uuid,
        "prefill_addr": f"{self.local_ip}:{self.p2p_port}",
        "num_layers": len(kv_tensors),
        "kv_shape": kv_tensors[0][0].shape,
    }
```

#### 2.2.4 Decode 端：拉取 KV 并写入

```python
# tpu_connector.py: Decode 端 — 拉取 KV 并 scatter 到本地 KV Pool
def _pull_kv(self, transfer_params: dict):
    """Decode 端：连接 Prefill → 拉取 KV → scatter 到本地 KV Pool"""

    # 1. 连接到 Prefill P2P server
    prefill_addr = transfer_params["prefill_addr"]
    self._connect_to_prefill(prefill_addr)

    # 2. 构造 KV 形状规范（告诉 Prefill 端要拉什么形状的数据）
    kv_spec = self._build_kv_spec(transfer_params)

    # 3. Pull-based 传输：Decode 主动拉取
    kv_tensors = self.p2p_client.pull(
        uuid=transfer_params["transfer_uuid"],
        specs=kv_spec,
    )

    # 4. Scatter KV slices 到本地 KV Cache Pool
    self.scatter_kv_slices(kv_tensors, transfer_params["block_ids"])

    # 5. ZMQ 通知 Prefill 端：拉取完成，可以释放缓冲区
    self._notify_pull_complete(transfer_params["transfer_uuid"])

def scatter_kv_slices(self, kv_tensors, block_ids):
    """将拉取的 KV 数据写入本地 KV Cache Pool 的指定 block 位置"""
    for layer_idx, (key, value) in enumerate(kv_tensors):
        # 写入对应 layer 和 block 位置
        self.model_runner.kv_cache_pool.scatter_kv(
            layer_idx=layer_idx,
            block_ids=block_ids,
            key=key,
            value=value,
        )
```

#### 2.2.5 ZMQ Side Channel 通知

```python
# tpu_connector.py: ZMQ 通知机制
class TPUConnectorWorker:
    def _start_zmq_listener(self):
        """Prefill 端：启动 ZMQ ROUTER socket 监听拉取完成通知"""
        zmq_port = int(os.environ.get("TPU_SIDE_CHANNEL_PORT", 9600))
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.ROUTER)
        self.zmq_socket.bind(f"tcp://*:{zmq_port}")

        # 后台线程监听
        self.zmq_thread = threading.Thread(
            target=self._pull_notify_listener, daemon=True
        )
        self.zmq_thread.start()

    def _pull_notify_listener(self):
        """监听 Decode 端发来的拉取完成通知"""
        while True:
            identity, _, message = self.zmq_socket.recv_multipart()
            notify = json.loads(message)
            transfer_uuid = notify["transfer_uuid"]

            # 释放 Prefill 端为该传输保留的 KV 缓冲区
            self._release_kv_buffer(transfer_uuid)
            logger.info(f"Released KV buffer for {transfer_uuid}")

    def _notify_pull_complete(self, transfer_uuid: str):
        """Decode 端：通过 ZMQ DEALER 通知 Prefill 拉取完成"""
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        socket.connect(f"tcp://{self.prefill_addr}:{self.zmq_port}")
        socket.send_json({"transfer_uuid": transfer_uuid})
        socket.close()
```

#### 2.2.6 Proxy Server

Proxy Server 负责请求路由：先发往 Prefill → 提取 transfer params → 转发 Decode：

```python
# examples/disagg/proxy_server.py: HTTP 路由
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # 1. 发送到 Prefill server（max_tokens=1，仅做 prefill）
    prefill_response = await forward_to_prefill(
        request, max_tokens=1
    )

    # 2. 提取 KV transfer params
    kv_transfer_params = prefill_response["kv_transfer_params"]

    # 3. 转发到 Decode server（附带 transfer params，流式返回）
    decode_request = request.copy()
    decode_request.kv_transfer_params = kv_transfer_params

    return StreamingResponse(
        forward_to_decode_stream(decode_request)
    )
```

### 2.3 Pallas DMA KV 传输内核

用于 Local Disagg 中的高效 KV Cache 复制，通过异步 DMA 实现层间管线化：

```python
# kv_transfer.py: multi_layer_copy — 多层异步 DMA 复制
def multi_layer_copy(
    src_kv: list[tuple[jax.Array, jax.Array]],
    dst_kv: list[tuple[jax.Array, jax.Array]],
    block_ids: jax.Array,
    num_layers: int,
):
    """多层 KV Cache 异步 DMA 复制。
    Layer N 的 DMA 启动与 Layer N-1 的 DMA 等待重叠，最大化带宽利用。

    管线时序：
      Layer 0: start_copy ──────────────────────────────────
      Layer 1:              start_copy ──────────────────────
      Layer 0:                          wait_copy ───────────
      Layer 2:                                    start_copy
      Layer 1:                                    wait_copy
      ...
    """
    # 启动第一层的 DMA
    sem_0 = _start_chunked_copy_kernel(
        src_kv[0], dst_kv[0], block_ids
    )

    for layer_idx in range(1, num_layers):
        # 启动下一层的 DMA（与上一层的等待重叠）
        sem_next = _start_chunked_copy_kernel(
            src_kv[layer_idx], dst_kv[layer_idx], block_ids
        )
        # 等待上一层完成
        _wait_for_chunked_copy_kernel(sem_0)
        sem_0 = sem_next

    # 等待最后一层完成
    _wait_for_chunked_copy_kernel(sem_0)
```

Pallas DMA 内核实现：

```python
# kv_transfer.py: Pallas 异步 DMA 内核
def _start_chunked_copy_kernel(src_ref, dst_ref, block_ids):
    """启动异步 HBM-to-HBM DMA 复制"""
    def kernel(src_ref, dst_ref, sem_ref):
        # 使用 pltpu.make_async_copy 发起非阻塞 DMA
        copy_op = pltpu.make_async_copy(
            src_ref=src_ref,
            dst_ref=dst_ref,
            sem=sem_ref,
        )
        copy_op.start()  # 非阻塞：DMA 引擎独立执行

    return pl.pallas_call(kernel, ...)(src_ref, dst_ref)

def _wait_for_chunked_copy_kernel(sem):
    """等待之前启动的 DMA 复制完成"""
    def kernel(sem_ref):
        copy_op = pltpu.make_async_copy(sem=sem_ref)
        copy_op.wait()  # 阻塞直到 DMA 完成

    pl.pallas_call(kernel, ...)(sem)
```

### 2.4 Scheduler 端集成 (`TPUConnectorScheduler`)

```python
# tpu_connector.py: Scheduler 端
class TPUConnectorScheduler:
    """管理 Producer (Prefill) 和 Consumer (Decode) 角色的调度逻辑"""

    def __init__(self, is_prefill: bool):
        self.is_prefill = is_prefill
        self.reqs_to_send: dict[str, RequestMetadata] = {}  # Prefill 待发送
        self.reqs_to_load: dict[str, RequestMetadata] = {}  # Decode 待加载

    def process_send_load(self, batch_result):
        """Prefill 完成后：构建 transfer metadata"""
        if self.is_prefill:
            for req in batch_result.finished_requests:
                # 构建传输元数据
                self.reqs_to_send[req.request_id] = RequestMetadata(
                    block_ids=req.block_ids,
                    seq_len=req.seq_len,
                    kv_transfer_params=req.kv_transfer_params,
                )
        else:
            # Decode 端：接收传输元数据
            for req in batch_result.new_requests:
                if req.kv_transfer_params:
                    self.reqs_to_load[req.request_id] = req.kv_transfer_params
```

---

## 三、sglang-jax 现状

### 3.1 已有代码

1. `schedule_batch.py:332` — `process_batch_result_disagg_prefill` 方法名存在，有 `tmp_end_idx` 字段和 "overlap schedule / kv transfer" 注释，但为 stub

2. `scheduler.py:850` — `"prefill_decode_size"` metrics key（仅统计，非分离逻辑）

### 3.2 完全缺失

- 无 Disagg Orchestrator / Engine 分离
- 无 KV Cache 跨 Mesh 传输
- 无 P2P Transfer Server
- 无 ZMQ 通信
- 无设备分配机制
- 无 Proxy Server

---

## 四、实现路线

### Phase 1: Local Disagg（推荐先实现）

**工作量**: ~5-7 天

Local 模式更简单，适合验证概念：

1. **设备分配器**：
   - 解析 `PREFILL_SLICES` / `DECODE_SLICES` 环境变量
   - 将 TPU 芯片分为两组，创建独立 Device Mesh

2. **双 Engine 架构**：
   - 在同一进程中创建两个 `ModelRunner` 实例（prefill / decode）
   - 各自拥有独立的 KV Cache Pool

3. **编排器** (`DisaggOrchestrator`)：
   - Prefill 线程：运行 prefill → 提取 KV Cache blocks
   - Transfer 线程：`jax.device_put` 重分片到 decode mesh
   - Decode 线程：写入 KV → 开始解码
   - 线程安全队列协调

4. **KV Cache 传输**：

   ```python
   # 简单方案：jax.device_put 跨 mesh
   kv_decode = jax.device_put(kv_prefill, decode_sharding)

   # 优化方案：Pallas DMA kernel
   multi_layer_copy(kv_prefill, kv_decode, block_ids)
   ```

### Phase 2: Multi-host Disagg

**工作量**: ~7-10 天

1. **JAX P2P Transfer Server**：
   - 使用 `jax.experimental.transfer.start_transfer_server`
   - Pull-based 模型

2. **ZMQ Side Channel**：
   - Prefill Worker 运行 ZMQ ROUTER socket
   - Decode 拉取完成后通知释放

3. **Proxy Server**：
   - HTTP 路由：先 prefill → 提取 params → 转发 decode
   - Round-robin 或负载感知调度

4. **TPUConnector 接口**：
   - 与 sglang-jax 调度器集成
   - 管理 Producer/Consumer 角色

---

## 五、关键设计决策

### 5.1 先 Local 还是先 Multi-host

**建议先 Local**：

- 实现更简单（单进程，无网络）
- 验证 KV Cache 传输正确性
- 适合常见的单机多 chip 场景
- Local Disagg 的核心组件（KV 提取、传输、注入）可在 Multi-host 中复用

### 5.2 KV Cache 传输方式

**简单方案**：`jax.device_put` — 简单但可能有性能问题（同步阻塞）

**生产方案**：Pallas DMA Kernel (`multi_layer_copy`) — 异步 HBM-to-HBM 复制，层间管线化

**建议**：Phase 1 用 `jax.device_put`，Phase 2 实现 Pallas DMA Kernel。

### 5.3 与 sglang-jax Scheduler 的集成

sglang-jax 的 Scheduler 已有 `process_batch_result_disagg_prefill` stub。可在此基础上扩展：

- Prefill Scheduler：仅调度 prefill 请求，完成后标记为待传输
- Decode Scheduler：从传输队列消费，分配 KV blocks 后开始解码

### 5.4 线程安全

tpu-inference 使用 `JetThread`（崩溃即 SIGKILL）确保可靠性。sglang-jax 可使用标准 Python threading + daemon 线程作为初始方案。

### 5.5 环境变量配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `PREFILL_SLICES` | `"4"` | Prefill 设备数量 |
| `DECODE_SLICES` | `"4"` | Decode 设备数量 |
| `TPU_KV_TRANSFER_PORT` | 9100 | P2P 传输端口 |
| `TPU_SIDE_CHANNEL_PORT` | 9600 | ZMQ 通知端口 |
| `TPU_KV_TRANSFER_CHANNEL_NUMBER` | 8 | 并行传输通道 |
| `TPU_ENABLE_D2H_TRANSFER` | true | 传输前先移到 Host memory |
| `TPU_P2P_WAIT_PULL_TIMEOUT` | 30 | 缓冲区保持秒数 |

---

## 六、参考文件

### tpu-inference

| 文件 | 内容 |
|---|---|
| `core/core_tpu.py` | `DisaggEngineCore`, `_DisaggOrchestrator`（三线程编排） |
| `core/disagg_executor.py` | `DisaggExecutor`（设备分配） |
| `core/disagg_utils.py` | `_parse_slices` 等工具函数 |
| `distributed/tpu_connector.py` | `TPUConnectorScheduler` / `TPUConnectorWorker`（Multi-host P2P） |
| `distributed/kv_transfer.py` | `multi_layer_copy`（Pallas DMA 异步复制） |
| `examples/disagg/` | 部署示例脚本 |

### sglang-jax 待修改/新建文件

| 文件 | 修改内容 |
|---|---|
| `srt/managers/scheduler.py` | Disagg prefill/decode 调度逻辑 |
| `srt/managers/schedule_batch.py` | `process_batch_result_disagg_prefill` 实现 |
| `srt/model_executor/model_runner.py` | 双 ModelRunner 支持 |
| `srt/entrypoints/engine.py` | Disagg Engine 入口 |
| `srt/distributed/disagg_orchestrator.py` | 新建：编排器 |
| `srt/distributed/kv_transfer.py` | 新建：KV 传输 |
| `srt/distributed/proxy_server.py` | 新建：Multi-host Proxy |
