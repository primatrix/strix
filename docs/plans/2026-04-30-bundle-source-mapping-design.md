# Bundle-Source Mapping: LLO VLIW Bundles 与 Pallas 源码对应分析

## 概述

为 Strix 新增 `analyze-bundles` 子命令，解析 `*-final_bundles.txt` 文件，构建完整的 VLIW Bundle AST，并建立 Pallas 源码行到 Bundle slot 的倒排索引，支持查询「某行 Pallas 代码编译产生了哪些 VLIW 指令」。

## 背景

### 输入格式

`*-final_bundles.txt` 是 LLO 编译器最终输出的 VLIW bundle 列表。每行一个 bundle，格式为：

```
<address> <control_flags>: <nesting> { <instr1> ;; <instr2> ;; ... } <comments>
```

示例：
```
0x5d   :  { %s147 = sshll.u32 ... /* loc(":684:10 to :34") */  ;;  %s158 = sshll.u32 ... /* loc(":690:10 to :34") */ }
```

关键特征：
- 一个 bundle 内多条指令以 `;;` 分隔，每条可能有不同的 `loc()` 注解
- 同一个 Pallas 表达式的 LLO 指令可能散布在多个不连续的 bundle 中（VLIW 调度重排）
- `loc()` 中的路径是 TPU Pod 内路径，非本地路径

### 与现有 Strix 的关系

现有 Strix 解析 post-finalize-llo IR（单条指令序列），本功能解析 final_bundles（VLIW 打包后）。两者格式完全不同，作为独立模块实现。

## 数据模型

```python
# strix/bundle_domain.py

@dataclass
class SourceLoc:
    file: str           # 原始完整路径 (TPU Pod 路径)
    start_line: int
    start_col: int
    end_line: int       # 跨行时不同于 start_line
    end_col: int

@dataclass
class BundleInstruction:
    """Bundle 内的单条 VLIW slot 指令"""
    opcode: str                  # e.g. "sshll.u32", "dma.hbm_to_vmem"
    raw_text: str                # 完整原始文本
    outputs: List[str]           # SSA outputs, e.g. ["%s30_s17"]
    loc: Optional[SourceLoc]     # 来自 /* loc(...) */ 注解

@dataclass
class Bundle:
    """一个 VLIW Bundle (一个时钟周期)"""
    address: int                 # 十六进制地址
    control_flags: List[str]     # ["LH"], ["LB", "CT"], etc.
    nesting_depth: int           # > 的个数, 0 = top level
    instructions: List[BundleInstruction]
    comments: List[str]          # entry/exit bundle, region markers
    is_empty: bool               # {} NOP bundle

@dataclass
class BundleProgram:
    """解析后的完整 bundle 程序"""
    bundles: List[Bundle]
    source_index: Dict[SourceLoc, List[Tuple[int, int]]]
    # SourceLoc → [(bundle_address, slot_index), ...] 倒排索引
```

映射粒度为 **(bundle_address, slot_index)**，因为同一 bundle 内不同 slot 可能对应不同 Pallas 表达式。

## 解析逻辑

```python
# strix/bundle_parser.py

class BundleParser:
    def parse_file(self, path: str) -> BundleProgram:
        """入口：读取文件，跳过 header (控制标志说明)，逐行解析"""

    def _parse_bundle_line(self, line: str) -> Optional[Bundle]:
        """解析单行 bundle:
        1. 提取地址 (hex)
        2. 提取控制标志 (LH/LB/LE/PB/PF/CT)
        3. 计算嵌套深度 (> 的个数)
        4. 提取 { ... } 内容，按 ;; 分割为指令
        5. 提取 } 之后的注释
        """

    def _parse_instruction(self, text: str) -> BundleInstruction:
        """解析单条 slot 指令:
        1. 提取 outputs (= 左边的 %name)
        2. 提取 opcode
        3. 提取 /* loc(...) */ → SourceLoc
        4. 保留 raw_text
        """

    def _parse_loc(self, loc_str: str) -> SourceLoc:
        """解析两种 loc 格式:
        - loc("file":L1:C1 to :C2)      → 单行
        - loc("file":L1:C1 to L2:C2)    → 跨行
        """

    def _build_source_index(self, bundles: List[Bundle]) -> Dict:
        """遍历所有 bundle 的所有 slot，构建 SourceLoc → [(addr, slot)] 倒排索引"""
```

关键正则：
- Bundle 行: `^\s*(0x[0-9a-f]+)\s+([A-Z, ]*)\s*:\s*(>*)\s*\{(.*)\}\s*(.*)$`
- Loc 注解: `loc\("([^"]+)":(\d+):(\d+)\s+to\s+(?:(\d+):)?:?(\d+)\)`
- 指令分隔: `;;`

特殊情况处理：
- 空 bundle `{}`
- 多行 bundle（BoundsCheck 的 `shalt.err` 跨行注释，含 `hlo:` 行）
- 非 loc 注释（`/* materialized constant */`, region markers）

## 输出

### Console 输出

按源码行号排序，每个 Pallas 表达式一段：

```
=== Bundle-Source Mapping ===
File: kernel.py

  L587:14-42
    14 instrs / 8 bundles (0x11..0x1a, non-contiguous)
    sld×2  sshll.u32×1  sadd.s32×1  scmp×1  smin.u32×1  sand.u32×1  ssub.s32×1

  L665:12-669:13
    48 instrs / 40 bundles (0x08, 0x1b..0x5a)
    sor.u32×8  vsyncadd.remote×8  sshrl.u32×8  sand.u32×8  sshll.u32×8  sadd.s32×8

Summary: 87 source locations, 5289 bundles, 3682 annotated instrs
```

当提供 `--source-root` 时，额外展示对应 Pallas 源码行：

```
  L684:10-34
    > gate = pl.load(gate_ref, ...)
    14 instrs / 11 bundles (0x5d..0x68)
    ...
```

路径匹配策略：从 loc 路径末尾逐级向上，在 `--source-root` 下寻找唯一匹配文件。

### JSON 输出

```json
{
  "total_bundles": 5289,
  "annotated_instructions": 3682,
  "mappings": [
    {
      "loc": {
        "file": "/workspace/.../kernel.py",
        "start_line": 684, "start_col": 10,
        "end_line": 684, "end_col": 34
      },
      "slots": [
        {"bundle": "0x5d", "slot": 2},
        {"bundle": "0x62", "slot": 0}
      ],
      "opcodes": {"scalar_lea.hbm": 1, "scalar_lea.vmem": 2, "dma.hbm_to_vmem": 1}
    }
  ]
}
```

## CLI 接口

```bash
# 基本用法
python -m strix.cli analyze-bundles path/to/final_bundles.txt

# JSON 输出
python -m strix.cli analyze-bundles path/to/final_bundles.txt --json output.json

# 按行号过滤
python -m strix.cli analyze-bundles path/to/final_bundles.txt --line 684

# 关联本地源码
python -m strix.cli analyze-bundles path/to/final_bundles.txt \
  --source-root /local/path/to/project/
```

## 文件结构

### 新增文件

| 文件 | 职责 |
|------|------|
| `strix/bundle_domain.py` | 数据类: SourceLoc, BundleInstruction, Bundle, BundleProgram |
| `strix/bundle_parser.py` | BundleParser: 解析 final_bundles.txt |
| `strix/bundle_exporter.py` | BundleConsoleExporter + BundleJsonExporter |

### 修改文件

| 文件 | 改动 |
|------|------|
| `strix/cli.py` | 新增 `analyze-bundles` 子命令 |

### 不变文件

parser.py, simulator.py, analyzer.py, op_events.py, exporters.py, domain.py, hardware.py, value_resolver.py, cost_model.py

### 依赖

零新增依赖。纯标准库: re, dataclasses, json, argparse, pathlib。
