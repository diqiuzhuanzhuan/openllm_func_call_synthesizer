# 评测模式说明文档

## 概述

新增了两种评测模式：**合并评测** 和 **分开评测**，可以在配置文件中通过 `evaluate_separate` 参数控制。

## 配置方式

在 `config.yaml` 中配置：

```yaml
# 评测模式配置
evaluate_separate: false  # false=合并评测, true=分开评测
```

## 两种模式详解

### 模式 1: 合并评测 (evaluate_separate: false)

**特点**：
- 对所有数据进行统一评测
- 不区分 MCP 和 Uliya 数据
- 输出一份完整的评测报告

**适用场景**：
- 想了解整体模型性能
- 数据类型混合，不需要细分

**输出文件**：
- `{output_base}.xlsx` - 完整评测结果

**示例输出**：
```
================================================================================
【合并评测模式】对所有数据进行统一评测
================================================================================

============================================================
开始评测数据集: All (共 1234 条)
============================================================

【All】函数相等分布: ...
【All】参数相等分布: ...

=== 【All】评测结果简表 ===
function_same | 总数: 1234 | 正确: 1100 | 错误: 134 | 准确率: 89.14%
argument_same | 总数: 1234 | 正确: 1050 | 错误: 184 | 准确率: 85.09%

========== 【All】argument具体计算过程如下 ==========
...
```

---

### 模式 2: 分开评测 (evaluate_separate: true)

**特点**：
- 根据 `gt_intent` 自动分离 MCP 和 Uliya 数据
- 分别计算各自的准确率
- 输出多份评测报告（MCP、Uliya、Other、完整）

**适用场景**：
- 需要对比 MCP 和 Uliya 的性能差异
- 想了解模型在不同任务上的表现
- 分析特定类型数据的准确率

**数据分离规则**：
- **MCP 数据**: `gt_intent` 在 `mcp_intent_list` 中
- **Uliya 数据**: `gt_intent` 在 `uliya_intent_list` 中
- **Other 数据**: 不在以上两个列表中的数据

**输出文件**：
- `{output_base}_mcp.xlsx` - MCP 数据评测结果
- `{output_base}_uliya.xlsx` - Uliya 数据评测结果
- `{output_base}_other.xlsx` - Other 数据评测结果（如果有）
- `{output_base}.xlsx` - 完整评测结果（合并所有）

**示例输出**：
```
================================================================================
【分开评测模式】根据 intent 类型分别评测 MCP 和 Uliya 数据
================================================================================

数据分布: MCP=865 条, Uliya=369 条, Other=0 条

============================================================
开始评测数据集: MCP (共 865 条)
============================================================

【MCP】函数相等分布: ...
【MCP】参数相等分布: ...

=== 【MCP】评测结果简表 ===
function_same | 总数: 865 | 正确: 780 | 错误: 85 | 准确率: 90.17%
argument_same | 总数: 865 | 正确: 750 | 错误: 115 | 准确率: 86.71%

============================================================
开始评测数据集: Uliya (共 369 条)
============================================================

【Uliya】函数相等分布: ...
【Uliya】参数相等分布: ...

=== 【Uliya】评测结果简表 ===
function_same | 总数: 369 | 正确: 320 | 错误: 49 | 准确率: 86.72%
argument_same | 总数: 369 | 正确: 300 | 错误: 69 | 准确率: 81.30%

MCP 评测结果保存于: test_all_evaluate_mcp.xlsx
Uliya 评测结果保存于: test_all_evaluate_uliya.xlsx
完整评测结果保存于: test_all_evaluate.xlsx
```

---

## Intent 列表配置

确保在 `config.yaml` 中正确配置了 intent 列表：

```yaml
# MCP Intent 列表（9个）
mcp_intent_list:
  - "create_album"
  - "search_photos"
  - "get_album_list"
  - "music_play_control"
  - "music_search_control"
  - "music_settings_control"
  - "video_search_control"
  - "video_play_control"
  - "get_system_info"

# Uliya Intent 列表（4个）
uliya_intent_list:
  - "general_query"
  - "summary_document"
  - "search_document"
  - "translate"
```

## 使用示例

### 示例 1: 合并评测

```yaml
# config.yaml
evaluate_separate: false
steps:
  evaluate: true
```

运行：
```bash
python main.py --config configs/config.yaml
```

结果：生成 `test_all_evaluate.xlsx`

---

### 示例 2: 分开评测

```yaml
# config.yaml
evaluate_separate: true
steps:
  evaluate: true
```

运行：
```bash
python main.py --config configs/config.yaml
```

结果：生成 4 个文件
- `test_all_evaluate_mcp.xlsx`
- `test_all_evaluate_uliya.xlsx`
- `test_all_evaluate_other.xlsx` (如果有)
- `test_all_evaluate.xlsx`

---

## 评测指标说明

两种模式使用相同的评测指标：

### 1. 基础指标
- **function_same**: intent 是否完全匹配
- **argument_same**: slots 是否完全匹配

### 2. 细粒度指标 (argument_evaluation_result)
- **exact_match**: 完全一致
- **partial_match**: 部分字段匹配
- **subset**: 子集关系
- **has_extra_fields**: 有多余字段
- **no_match**: 完全不匹配

### 3. 综合准确率
- **(exact_match + partial_match) / total**
- **(exact_match + partial_match + subset) / total**

---

## 注意事项

1. **gt_intent 字段要求**：
   - 分开评测模式依赖 `gt_intent` 字段
   - 确保在推理阶段已经生成了 `gt_intent` 列
   - `gt_intent` 由 `preprocess_data()` 函数自动从 `output` 字段提取

2. **数据完整性**：
   - 如果某个 intent 不在任何列表中，会被归入 "Other" 类别
   - Other 类别也会单独评测（如果有数据）

3. **文件保存**：
   - 分开评测会生成多个文件，便于分别分析
   - 完整文件包含所有数据，保持原始顺序

4. **性能影响**：
   - 两种模式评测耗时基本相同
   - 分开评测只是多了数据分离和多次打印的开销

---

## 最佳实践

### 推荐工作流程

1. **开发阶段** - 使用合并评测
   ```yaml
   evaluate_separate: false
   ```
   快速了解整体性能

2. **分析阶段** - 使用分开评测
   ```yaml
   evaluate_separate: true
   ```
   深入分析各类型数据的表现

3. **对比实验** - 两种模式都运行
   - 先运行合并评测，了解整体
   - 再运行分开评测，找出问题

### 常见使用场景

**场景 1**: 模型在某类任务上表现差
```yaml
evaluate_separate: true
# 分开评测后可以看到是 MCP 还是 Uliya 类型的任务准确率低
```

**场景 2**: 对比不同 prompt 的效果
```yaml
mode: "split"  # 使用 split 模式确保使用正确的 prompt
evaluate_separate: true  # 分开评测看效果
```

**场景 3**: 快速验证模型整体能力
```yaml
evaluate_separate: false  # 合并评测即可
```

---

## 代码结构

```
evaluate_module(config, data)  # 主入口
├── 判断 evaluate_separate 参数
├── if True: 分开评测模式
│   ├── 根据 gt_intent 分离数据
│   ├── evaluate_single_dataset(data_mcp, "MCP")
│   ├── evaluate_single_dataset(data_uliya, "Uliya")
│   ├── evaluate_single_dataset(data_other, "Other")
│   └── 保存多个文件
└── if False: 合并评测模式
    ├── evaluate_single_dataset(data, "All")
    └── 保存一个文件
```

每个 `evaluate_single_dataset()` 都会计算完整的评测指标。
