本代码用于测评微调后的模型，在测试数据（test data）上的效果。  
具体功能为从本地路径加载模型，请求模型得到response， 数据后处理， 并给出完成测评报告。   
# 调用方法
python main.py --config ./config.yaml

在config.yaml里可以方便配置相关参数
可以选择只执行其中某些步骤

# 说明
为避免重复请求模型浪费时间， 每一个步骤都独立保存结果 
在config里需定义好input output文件 
可以在step里选择执行哪个步骤  

另外，数据后处理，可以根据自己需求 添加内容。 

# 使用说明
## steps: 控制执行哪些步骤 (true/false)
    steps:
        inference: true                 # 是否执行LLM推理
        postprocess: true             # 是否执行数据后处理（LLM+GT标准字段合成一步）
        evaluate: true                  # 是否执行评测
        evaluate_output_str: false     # 是否执行输出为str格式的评测
## 输出格式 支持json格式和str格式的 解析及测评
  evaluate: true                  # 是否执行评测
  evaluate_output_str: false

## 支持以api方式调用， 支持load本地模型方式调用
use_api: true # false

