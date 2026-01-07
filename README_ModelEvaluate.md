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