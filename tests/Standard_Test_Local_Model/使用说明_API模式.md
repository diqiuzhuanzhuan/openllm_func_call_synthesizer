# API模式使用说明

## 功能概述

`evaluate_local_model.py` 已新增API调用功能，现在支持两种运行模式：

1. **本地模型模式**（原有功能）：直接加载本地模型到GPU进行推理
2. **API模式**（新增功能）：通过HTTP API调用远程模型服务进行推理

## 新增内容

### 1. 新增函数

#### `get_llm_response_api()`
单次API调用函数，用于发送请求并获取模型响应。

**参数：**
- `text`: 用户输入文本
- `system_prompt`: 系统提示词
- `api_url`: API服务地址
- `model_name`: 模型名称
- `top_p`: 采样参数（默认0.1）
- `temperature`: 温度参数（默认0.01）

**返回：**
- 成功：返回生成的文本
- 失败：返回None

#### `batch_llm_predict_api()`
批量API调用函数，读取Excel文件中的数据，批量调用API进行预测。

**功能：**
- 读取输入Excel文件
- 对每行数据调用API
- 解析返回结果
- 保存到输出Excel文件

### 2. 新增配置项

在YAML配置文件中新增以下配置项：

```yaml
# API模式开关
use_api: true  # true=使用API模式, false=使用本地模型模式

# API相关配置
api_url: "http://192.168.111.3:11434/api/chat"  # API服务地址
model_name: "mcp_intent_1016:q4_k_m"  # 模型名称
top_p: 0.1  # 可选，默认0.1
temperature: 0.01  # 可选，默认0.01
```

## 快速开始

### 步骤1：准备配置文件

复制示例配置文件：
```bash
cp config_api_example.yaml my_api_config.yaml
```

或者修改现有配置文件，添加以下内容：

```yaml
use_api: true
api_url: "http://your-api-server:port/api/chat"
model_name: "your-model-name"
```

### 步骤2：运行程序

```bash
python evaluate_local_model.py --config my_api_config.yaml
```

### 步骤3：查看结果

结果会保存到配置文件中指定的 `output_file` 路径。

## 详细配置说明

### 完整配置文件示例

```yaml
# ========== API模式配置 ==========
use_api: true
api_url: "http://192.168.111.3:11434/api/chat"
model_name: "mcp_intent_1016:q4_k_m"
top_p: 0.1
temperature: 0.01

# ========== 文件路径配置 ==========
input_file: "/path/to/input.xlsx"
output_file: "/path/to/output.xlsx"

# ========== 执行步骤控制 ==========
steps:
  inference: true      # 执行推理
  postprocess: true    # 数据后处理
  evaluate: true       # 执行评测
  evaluate_output_str: false

# ========== 字段配置 ==========
input_field: "input"
ground_truth: "output"
ground_truth_intent: "gt_intent"
ground_truth_slot: "gt_slots"
llm_intent: "llm_intent"
llm_slot: "llm_slots"

# ========== 系统提示词 ==========
system_prompt: >
  Your system prompt here...
```

## 使用场景对比

| 场景 | 建议模式 | 原因 |
|------|---------|------|
| 有强大GPU资源 | 本地模型 | 速度快，无网络依赖 |
| GPU资源有限 | API模式 | 不占用本地GPU |
| 需要调用远程服务 | API模式 | 统一使用远程模型 |
| 需要快速切换模型 | API模式 | 无需重新加载模型 |
| 需要最高性能 | 本地模型 | 无网络延迟 |

## 功能特点

### API模式优势
1. ✅ 无需本地GPU资源
2. ✅ 无需加载大模型到内存
3. ✅ 可以调用远程强大的模型
4. ✅ 配置简单，只需API地址
5. ✅ 多个客户端可以共享同一个模型服务

### 本地模式优势
1. ✅ 推理速度快（无网络延迟）
2. ✅ 数据不离开本地，更安全
3. ✅ 不依赖网络连接
4. ✅ 完全控制模型版本

## 测试API功能

提供了独立的测试脚本 `test_api_function.py`：

```bash
python test_api_function.py
```

该脚本会测试API调用的基本功能，输出示例：

```
==================================================
开始测试API调用功能
==================================================

API地址: http://192.168.111.3:11434/api/chat
模型名称: mcp_intent_1016:q4_k_m

【测试用例 1】
输入: 我想找周杰伦的歌
--------------------------------------------------
API调用成功，耗时: 2.35 秒
响应: {"intent": "music_play_control", "slots": {"title": "周杰伦"}}
--------------------------------------------------
```

## 常见问题

### Q1: API调用失败怎么办？
**A:** 检查以下几点：
1. API服务是否正常运行
2. API地址是否正确
3. 网络连接是否正常
4. 模型名称是否正确

### Q2: 如何在两种模式之间切换？
**A:** 只需修改配置文件中的 `use_api` 参数：
- `use_api: true` → API模式
- `use_api: false` → 本地模型模式

### Q3: API模式下还需要配置CUDA吗？
**A:** 不需要。API模式下不会使用本地GPU，因此无需配置 `cuda_device`。

### Q4: 两种模式可以使用相同的配置文件吗？
**A:** 可以。程序会根据 `use_api` 参数自动选择相应的配置项。

### Q5: API响应格式有要求吗？
**A:** 是的，API需要返回Ollama格式的响应：
```json
{
  "message": {
    "content": "模型生成的文本"
  }
}
```

## 技术细节

### API请求格式

程序发送的API请求格式如下：

```json
{
  "model": "model_name",
  "stream": false,
  "top_p": 0.1,
  "temperature": 0.01,
  "messages": [
    {
      "role": "system",
      "content": "系统提示词"
    },
    {
      "role": "user",
      "content": "用户输入"
    }
  ]
}
```

### 错误处理

- API请求失败：返回None并打印错误信息
- 解析失败：返回原始字符串
- 异常情况：返回空字典 `{}`

### 性能优化建议

1. **批量处理**：程序会逐行处理Excel数据，对于大批量数据建议：
   - 分批处理
   - 考虑并发调用（需自行修改代码）

2. **超时设置**：可以在 `get_llm_response_api()` 函数中添加timeout参数

3. **重试机制**：对于网络不稳定的情况，建议添加重试逻辑

## 更新日志

### v2.0 (当前版本)
- ✨ 新增API调用模式
- ✨ 新增 `get_llm_response_api()` 函数
- ✨ 新增 `batch_llm_predict_api()` 函数
- ✨ 添加API模式配置支持
- 📝 添加详细文档和示例
- 🧪 添加测试脚本

### v1.0
- ✅ 支持本地模型推理
- ✅ 支持数据后处理
- ✅ 支持评测功能

## 联系方式

如有问题或建议，请联系开发团队。

## 附录

### 相关文件
- `evaluate_local_model.py` - 主程序
- `call_LLM_api.py` - API调用参考代码
- `config_api_example.yaml` - API模式配置示例
- `test_api_function.py` - API功能测试脚本
- `API_MODE_README.md` - 英文说明文档

### 依赖包
```
pandas
requests
pyyaml
openpyxl  # Excel读写
transformers  # 仅本地模式需要
torch  # 仅本地模式需要
```

