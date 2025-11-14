# 代码结构说明

## 重构后的模块化设计

### `llm_local_caller.py` - 本地模型推理

#### 核心函数
1. **`preprocess_data(df, config)`** - 统一数据预处理
   - 创建 `mcp_input`, `uliya_input`, `gt_intent` 三列
   - 被所有推理函数复用

2. **`batch_llm_predict(config)`** - 单进程推理主函数
   - 简洁清晰，只负责流程控制
   - 调用模式函数执行具体推理

3. **`batch_llm_predict_threaded(config)`** - 多进程推理主函数
   - 与单进程版本结构一致
   - 调用多进程模式函数

#### 辅助函数
- **`extract_intent(x)`** - 从 output 字段提取 intent
- **`_run_hybrid_mode(df, llm_once, config)`** - 单进程混合模式
- **`_run_split_mode(df, llm_once, config)`** - 单进程分割模式
- **`_run_hybrid_mode_mp(df, config, num_processes)`** - 多进程混合模式
- **`_run_split_mode_mp(df, config, num_processes)`** - 多进程分割模式

### `llm_api_caller.py` - API 推理

#### 核心函数
1. **`preprocess_data(df, config)`** - 统一数据预处理（与本地模型相同）

2. **`batch_llm_predict_api(config)`** - API 推理主函数
   - 结构与本地模型版本保持一致
   - 调用 API 模式函数执行具体推理

#### 辅助函数
- **`extract_intent(x)`** - 从 output 字段提取 intent
- **`_run_hybrid_mode_api(df, run_one, config)`** - API 混合模式
- **`_run_split_mode_api(df, run_one, config)`** - API 分割模式

## 代码优势

### 1. 消除冗余
- ✅ `extract_intent` 和 `preprocess_data` 提取为独立函数
- ✅ 所有推理函数复用相同的预处理逻辑
- ✅ 减少重复代码约 100+ 行

### 2. 清晰的职责分离
- **主函数**: 只负责流程编排（读取→预处理→推理→保存）
- **模式函数**: 专注于各自的推理逻辑（hybrid/split）
- **工具函数**: 提供通用功能（提取 intent、预处理数据）

### 3. 易于维护和扩展
- 如需添加新模式，只需新增 `_run_xxx_mode` 函数
- 修改预处理逻辑只需改 `preprocess_data` 一处
- 各函数职责单一，易于测试和调试

## 函数调用关系

```
batch_llm_predict(config)
├── preprocess_data(df, config)
│   └── extract_intent(x)
├── _run_hybrid_mode(df, llm_once, config)
│   └── llm_once(prompt)
└── _run_split_mode(df, llm_once, config)
    └── llm_once(prompt)

batch_llm_predict_threaded(config)
├── preprocess_data(df, config)
│   └── extract_intent(x)
├── _run_hybrid_mode_mp(df, config, num_processes)
│   └── llm_once_mp(prompt)  [多进程]
└── _run_split_mode_mp(df, config, num_processes)
    └── llm_once_mp(prompt)  [多进程]

batch_llm_predict_api(config)
├── preprocess_data(df, config)
│   └── extract_intent(x)
├── _run_hybrid_mode_api(df, run_one, config)
│   └── run_one(text, system_prompt)
└── _run_split_mode_api(df, run_one, config)
    └── run_one(text, system_prompt)
```

## 配置参数

所有函数共享相同的配置结构：
```yaml
# 必需参数
input_file: "path/to/input.xlsx"
output_file: "path/to/output.xlsx"
input_field: "input"
ground_truth: "output"
system_prompt_mcp_file: "path/to/mcp_prompt.txt"
system_prompt_uliya_file: "path/to/uliya_prompt.txt"

# 模式配置
mode: "hybrid"  # 或 "split"

# split 模式需要
mcp_intent_list: [...]
uliya_intent_list: [...]
```

## 使用建议

1. **混合模式 (hybrid)**: 适合未知数据分布，自动回退
2. **分割模式 (split)**: 适合已知 intent 分布，效率更高
3. 推荐先用小数据集测试混合模式，观察回退率
4. 根据回退率决定是否切换到分割模式
