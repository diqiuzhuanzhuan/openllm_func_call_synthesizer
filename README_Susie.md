# 模型评估：
##  examples/conf/config.yaml修改为：  
```
defaults:
  - synthesizer: evaluation
```
##  examples/conf/synthesizer/evaluation.yaml 需要修改内容： 
a. mcp_servers 地址是："http://192.168.111.11:8000/mcp"  
b. query_generation enable: False  
## function_call_generation 生成模型response    
* enable: True 执行次步骤  
* 输入数据： function_dataset 目录下的train.jsonl  
* max_num: -1   -1表示全部数据，先少量数据做测评
* name 模型名字需要修改
* **provider**：  
  * 如果是 ollama 模型，`backend: "litellm"`，示例：

    ```yaml
    provider:
      backend: "litellm"
      base_url: "http://192.168.111.11:11434"
    ```

  * 如果是 openai 模型，`backend: "openai"`。如果模型是部署的 vllm 服务，则 `base_url` 改成对应的 vllm 服务地址，例如：

    ```yaml
    provider:
      backend: "openai"
      base_url: "http://192.168.111.6:8000/v1"
    ```

## 注意：
先少量数据评估  
一般 "gpt-5-mini"效果比较好， 如果组内同时有人在用， max_tokens_per_minute要开小一些

# 环境相关
1. uv sync 同步环境 生成 .venv 文件
source ./venv/bin/activate 启动虚拟环境
2. openllm里openllm_func_call_synthesizer/.env 要换成自己的api_key， 但是注意 不要上传到git上。 
样例：
```
export OPENAI_API_KEY="sk-nhVbM4If "
export ANTHROPIC_API_KEY="sk-nhVbM4 "
export GEMINI_API_KEY="sk-nhVbM "

export OPENAI_API_BASE="https://api9.xhub.chat/v1"
export OPENAI_BASE_URL="https://api9.xhub.chat/v1"
export ANTHROPIC_API_BASE="https://api9.xhub.chat/v1"
export ANTHROPIC_BASE_URL="https://api9.xhub.chat/v1"
export GEMINI_API_BASE="https://api9.xhub.chat/v1"
export GEMINI_BASE_URL="https://api9.xhub.chat/v1" 
```
注意 文件名是 .env

不想传git的内容，可以加到/.gitignore文件里
3. 如果import 有波浪线。 
/openllm_func_call_synthesizer/.vscode/settings.json
```
{
  "files.exclude": {
    "src/openllm_func_call_synthesizer.egg-info": true,
    "**/__pycache__": true,
    "**/__pycache__.*": true
  },
  "python.envFile": "${workspaceFolder}/.env",
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true
}
```


# 通过种子query  扩展query 及  生成response
examples/function_docs.add_seed_query.example.json
function_docs.json样例
```
{
    "tools": [
      {
        "name": "search_photos",
        "description": "Search for photos or images",
        "input_schema": {
          "type": "object",
          "properties": {
            "keyword": {
              "type": "string",
              "description": "The search keyword for photos or images. It can be descriptive text or a file name, e.g., 'photos taken last August' or 'dog on the grass'."
            }
          },
          "required": [
            "keyword"
          ],
          "additionalProperties": false
        },
        "query": "Search for photos or images about Summer"
      }
    ]
}

```

# 生成最新的mcp工具定义
python /apps/function_call.py
结果保存到了根目录。 

# 常用git 命令
获取最新代码， 把自己代码更新 放到后面
git fetch； git rebase

git add .
git commit -m 'your comment'
git push