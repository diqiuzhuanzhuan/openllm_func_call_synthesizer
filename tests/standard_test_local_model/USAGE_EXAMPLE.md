# 快速使用指南

## 1. 合并评测（默认）

所有数据统一评测，输出一份报告。

**配置 `config.yaml`**:
```yaml
evaluate_separate: false
```

**运行**:
```bash
python main.py --config configs/config.yaml
```

**输出文件**:
- `test_all_evaluate.xlsx` - 完整评测结果

---

## 2. 分开评测

根据 intent 类型分别评测 MCP 和 Uliya 数据。

**配置 `config.yaml`**:
```yaml
evaluate_separate: true

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

uliya_intent_list:
  - "general_query"
  - "summary_document"
  - "search_document"
  - "translate"
```

**运行**:
```bash
python main.py --config configs/config.yaml
```

**输出文件**:
- `test_all_evaluate_mcp.xlsx` - MCP 数据评测
- `test_all_evaluate_uliya.xlsx` - Uliya 数据评测
- `test_all_evaluate.xlsx` - 完整评测结果

---

## 3. 配合推理模式使用

### 混合模式 + 分开评测

```yaml
# 推理配置
mode: "hybrid"  # 先用 MCP，失败后用 Uliya

# 评测配置
evaluate_separate: true  # 分开看 MCP 和 Uliya 的表现
```

### 分割模式 + 分开评测

```yaml
# 推理配置
mode: "split"  # 根据 intent 选择对应 prompt

# 评测配置
evaluate_separate: true  # 分开看各自表现
```

---

## 4. 完整流程示例

```yaml
# config.yaml 完整配置
steps:
  inference: true      # 执行推理
  postprocess: true    # 数据后处理
  evaluate: true       # 执行评测

mode: "hybrid"
evaluate_separate: true

input_file: "path/to/input.xlsx"
output_file: "path/to/output.xlsx"
evaluate_output_file: "path/to/evaluate.xlsx"
```

运行后会生成：
1. 推理结果: `output.xlsx`
2. 后处理结果: `output_processed.xlsx`
3. MCP 评测: `evaluate_mcp.xlsx`
4. Uliya 评测: `evaluate_uliya.xlsx`
5. 完整评测: `evaluate.xlsx`

---

## 5. 典型输出对比

### 合并评测输出
```
【合并评测模式】对所有数据进行统一评测
开始评测数据集: All (共 1234 条)

【All】评测结果简表
function_same | 总数: 1234 | 正确: 1100 | 准确率: 89.14%
```

### 分开评测输出
```
【分开评测模式】根据 intent 类型分别评测 MCP 和 Uliya 数据
数据分布: MCP=865 条, Uliya=369 条

【MCP】评测结果简表
function_same | 总数: 865 | 正确: 780 | 准确率: 90.17%

【Uliya】评测结果简表
function_same | 总数: 369 | 正确: 320 | 准确率: 86.72%
```

可以清楚看到 MCP 和 Uliya 数据的性能差异！
