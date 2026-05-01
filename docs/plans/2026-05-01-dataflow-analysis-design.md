# SSA 数据流分析与 Graphviz 可视化

## 概述

为 Strix 新增数据流分析能力，从模拟后的 OpEvent 树中提取 SSA def-use 依赖关系，生成 Graphviz DOT 文件。
图中展示所有指令执行节点、硬件分派（VPU/DMA）、时序信息和数据依赖边，支持可视化 kernel 中数据如何在硬件单元之间流动。

## 关联

- RFC: primatrix/wiki#143 (RFC-0024)
- 扩展现有 `strix analyze` 管线，不影响现有模块

## 设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 核心信息 | SSA 数据依赖图 | 展示指令间 def-use 关系和硬件分派 |
| 输出格式 | Graphviz DOT | Strix 保持零依赖，用户用 `dot` 渲染 |
| 数据源 | OpEvent 树 | 包含时序 + SSA + 硬件分派，信息最丰富 |
| 节点范围 | 全部指令 | 不过滤，展示完整数据流 |
| 布局策略 | 时间分区 DOT | 同时展示结构 + 时序 + 并行度 |

## 数据模型

### DataFlowGraph

从 OpEvent 树提取的轻量图结构：

```python
# strix/dataflow.py

@dataclass
class DFNode:
    """数据流图中的一个节点 = 一次指令执行"""
    id: str                        # 唯一标识, e.g. "ev_42"
    name: str                      # 操作名, e.g. "vmatmul", "enqueue_dma"
    stream: OpStream               # VPU / DMA / Control
    kind: OpKind                   # LEAF / BLOCK / STALL / LOOP
    start_ns: int
    end_ns: int
    flops: int
    bytes: int
    ssa_outputs: List[str]         # 该事件定义的 SSA 变量
    ssa_inputs: List[str]          # 该事件使用的 SSA 变量
    attributes: Dict[str, Any]     # 原始 OpEvent 属性
    loop_depth: int                # 循环嵌套深度 (用于 cluster)
    parent_loop_id: Optional[str]  # 所属循环的 id

@dataclass
class DFEdge:
    """SSA 依赖边"""
    src: str       # 源节点 id (producer)
    dst: str       # 目标节点 id (consumer)
    variable: str  # SSA 变量名, e.g. "%token_1"

@dataclass
class DataFlowGraph:
    nodes: List[DFNode]
    edges: List[DFEdge]
    loops: Dict[str, List[str]]  # loop_id -> [child node ids]
```

### 提取逻辑

1. 遍历 OpEvent 树，对每个叶节点/块节点创建 DFNode
2. 从 `ev.attributes["ssa_inputs"]` 和 `ev.attributes["ssa_outputs"]` 读取 SSA 信息
   （需在 simulator.py 的 `_schedule_event` 中存储）
3. 构建 `variable → producer_node_id` 映射
4. 对每个 consumer 节点的每个 input，创建 DFEdge 连接到对应 producer

## DOT 生成策略

### 时间分区

- 将 makespan 切分为等宽时间桶（bucket），每个桶 = 一个 `rank=same` 约束组
- 默认桶数 = `min(100, total_events)`
- 节点按 `start_ns` 落入对应桶
- `rankdir=LR`，时间从左到右

### 硬件分区

- `subgraph cluster_vpu`: VPU 计算节点
- `subgraph cluster_dma`: DMA 传输节点
- STALL 事件放在 VPU cluster 内

### 循环表达

- `scf.for` 循环体 → `subgraph cluster_loop_N`
- Label 标注 trip count
- 未展开的大循环只展示单次迭代，label 标注 `×N`

### 节点样式

| stream | shape | fillcolor | 示例 |
|--------|-------|-----------|------|
| VPU compute | box | lightblue | vmatmul, vadd |
| DMA | box | lightyellow | enqueue_dma |
| STALL | box | salmon | STALL, DMA_STALL |
| Control | diamond | lightgrey | scf.for |

节点 label 格式: `opcode\n{start_ns}→{end_ns}\n{flops}F {bytes}B`

### 边样式

- 同 stream 内: 实线黑色
- 跨 stream（VPU↔DMA）: 粗红色虚线，label = SSA 变量名

## CLI 集成

```bash
# 新增 --dataflow-output flag
python -m strix.cli path/to/post-finalize-llo.txt --dataflow-output dataflow.dot

# 渲染
dot -Tsvg dataflow.dot -o dataflow.svg
```

作为现有 CLI 的新 flag，不引入新子命令。

## 文件结构

### 新增文件

| 文件 | 职责 |
|------|------|
| `strix/dataflow.py` | DFNode, DFEdge, DataFlowGraph + `extract_dataflow(root_event)` |
| `strix/dataflow_exporter.py` | `DataFlowDotExporter`: DataFlowGraph → DOT 文本 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `strix/simulator.py` | `_schedule_event` 中将 SSA inputs/outputs 存入 `ev.attributes` |
| `strix/cli.py` | 新增 `--dataflow-output` 参数，调用 exporter |

### 不变文件

parser.py, analyzer.py, exporters.py, op_events.py, domain.py, hardware.py, value_resolver.py, cost_model.py

### 依赖

零新增依赖。纯标准库（dataclasses, typing）。

## 实施计划

| # | SubTask | 交付物 | 依赖 |
|---|---------|--------|------|
| S1 | SSA 信息持久化 | simulator.py: 在 `_schedule_event` 中存储 ssa_inputs/ssa_outputs 到 ev.attributes | 无 |
| S2 | 数据模型 + 提取 | dataflow.py: DFNode/DFEdge/DataFlowGraph + extract_dataflow() | S1 |
| S3 | DOT 导出器 | dataflow_exporter.py: DataFlowDotExporter | S2 |
| S4 | CLI 集成 | cli.py: --dataflow-output flag | S3 |
| S5 | 集成测试 | 用 ir_dumps 中的 LLO 文件验证端到端 | S4 |

## 测试策略

1. DOT 文件语法正确（能被 `dot` 解析）
2. 节点数 = 叶/块 OpEvent 数
3. 边数 = 非空 SSA 依赖数
4. 跨 stream 边正确标记为红色虚线
