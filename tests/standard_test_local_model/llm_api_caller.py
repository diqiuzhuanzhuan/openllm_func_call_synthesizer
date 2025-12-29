#!/usr/bin/env python

"""
LLM API调用模块
支持main函数，直接用config.yaml测评，也支持单条手动修改数据测试
"""

import json
import re
import time

import pandas as pd
import requests
from ollama import Client
from pandarallel import pandarallel
from parse_response_to_json import parse_react_to_json


def filter_think(text):
    try:
        rs = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return rs.strip()
    except Exception:
        return text.strip()


def get_llm_response_api(text, system_prompt, api_url, model_name, temperature=0.01):
    """
    使用API方式调用LLM
    """
    begin_time = time.time()

    payload = json.dumps(
        {
            "model": model_name,
            "stream": False,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        }
    )
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.request("POST", api_url, headers=headers, data=payload)
        if response.status_code == 200:
            generated_text = json.loads(response.text).get("message", {}).get("content", {})
            generated_text = filter_think(generated_text)
            end_time = time.time()
            print(f"[INFO] API调用成功，耗时: {end_time - begin_time:.2f} 秒")
            return generated_text
        else:
            print(f"[ERROR] API请求失败，状态码：{response.status_code}，错误信息：{response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] API调用异常: {e}")
        return None


def parse_llm_response(rsp):
    """
    尝试结构化解析LLM原始输出
    """
    if rsp is None:
        return {}
    print("[DEBUG] 最原始的结果：", rsp)
    try:
        parsed = parse_react_to_json(rsp)
    except Exception:
        parsed = None

    if isinstance(parsed, dict) and len(parsed) > 0:
        return parsed
    else:
        return rsp


def extract_intent(x):
    """从output字段中提取intent"""
    if pd.isna(x):
        return ""
    try:
        if isinstance(x, str):
            import ast
            import json

            try:
                parsed = json.loads(x)
            except Exception:
                parsed = ast.literal_eval(x)
        else:
            parsed = x
        if isinstance(parsed, dict):
            return parsed.get("intent", "")
    except Exception:
        return ""
    return ""


def preprocess_data(df, config):
    """
    统一数据预处理：创建 mcp_input, uliya_input, gt_intent 列
    Args:
        df: 输入的DataFrame
        config: 配置字典
    Returns:
        处理后的DataFrame
    """
    input_field = config["input_field"]
    ground_truth_field = config.get("ground_truth", "output")

    # 创建两个输入列（API模式不需要拼接system_prompt，在请求时拼接）
    df["mcp_input"] = df[input_field].astype(str)
    df["uliya_input"] = df[input_field].astype(str)

    # 提取 gt_intent
    df["gt_intent"] = df[ground_truth_field].apply(extract_intent)

    return df


def get_ollama_response_chat(system_prompt, text, url, model_name, tool_path, OPENAI_TOOLS_DATA):
    client = Client(host=url)

    model_name = model_name

    response = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.01},
        tools=OPENAI_TOOLS_DATA,
        # format=''
    )
    # # ✅ Ollama 正确取法
    msg = response["message"]

    # print('----------response["message"]-----------', msg)
    # 单独写一个函数来解析 tool_calls
    def parse_tool_calls(tool_calls):
        parsed_tools = []
        if tool_calls:
            for call in tool_calls:
                # 尝试当作对象解析
                function = getattr(call, "function", None)
                if function:
                    parsed_func = {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    }
                    parsed_tools.append(parsed_func)
                # 尝试当作 dict 解析（兼容性考虑）
                elif isinstance(call, dict) and "function" in call:
                    parsed_tools.append(
                        {
                            "name": call["function"].get("name"),
                            "arguments": call["function"].get("arguments"),
                        }
                    )
        return parsed_tools if parsed_tools else []

    result = {
        # "role": msg["role"],
        "content": filter_think(msg.get("content")),
        "tool_calls": parse_tool_calls(msg.get("tool_calls")),
    }
    print("===============result===============\n", result)
    if result["tool_calls"] and isinstance(result["tool_calls"], list) and len(result["tool_calls"]) > 0:
        return result["tool_calls"]
    else:
        return result["content"]


def extract_tool_call(content):
    # 使用正则表达式提取 <tool_call>...</tool_call> 之间的内容
    match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    if not match:
        return None
    raw_json = match.group(1).strip()
    try:
        # 解析json字符串
        return json.loads(raw_json)
    except Exception as e:
        print("解析tool_call内容为JSON失败:", e)
        return None


def get_llm_response(input_text, system_prompt, api_url, model_name, OPENAI_TOOLS_DATA):
    # 尝试加载为 Python list（如果原数据是字符串形式的 JSON list）
    if isinstance(OPENAI_TOOLS_DATA, str):
        tools = json.loads(OPENAI_TOOLS_DATA)
    else:
        tools = OPENAI_TOOLS_DATA

    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_text}],
        "temperature": 0.2,
        # "presence_penalty": 2,
        # "repetition_penalty": 2,
        # "top_p": 0.9,
        "stream": False,
        "tools": tools,  # OPENAI_TOOLS_DATA
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(response.text)

    resp = response.json()
    print("-------------resp--------------------\n", resp)
    message = resp["choices"][0]["message"]
    print("-----------------message--------------------\n", message)
    content = filter_think(message.get("content", ""))

    # 尝试解析 JSON，如果模型输出了 function call
    try:
        tool_call = json.loads(content)
    except Exception:
        tool_call = None

    return content, tool_call


def get_one_response(input_text, system_prompt, api_url, model_name, OPENAI_TOOLS_DATA):
    content, tool_call = get_llm_response(input_text, system_prompt, api_url, model_name, OPENAI_TOOLS_DATA)
    tool_call = extract_tool_call(content)
    print("模型原始输出：")
    print("content:", content)
    print("\n解析后的 JSON：")
    print("tool_call:", tool_call)

    if tool_call:
        return tool_call
    else:
        return content


def get_ollama_response_no_tool(config):
    with open(config.get("tool_path")) as f:
        OPENAI_TOOLS_DATA = json.load(f)

    # 读取数据并预处理
    df = pd.read_excel(config["input_file"])
    # df = df.iloc[0:10]
    df = preprocess_data(df, config)

    # 定义API调用函数
    api_url = config.get("api_url", "http://192.168.111.3:11434/api/chat")
    model_name = config.get("model_name")

    df["llm_response"] = df[config.get("input_field")].apply(
        lambda x: get_one_response(x, config.get("system_prompt"), api_url, model_name, OPENAI_TOOLS_DATA)
    )

    # 保存结果
    df.to_excel(config["output_file"], index=False)
    print("API LLM inference done, result saved:", config["output_file"])
    print("------end time (API)------", time.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def batch_ollama_api_function_call(config):
    """
    使用config.yaml批量调用LLM进行预测
    Args:
        config: 配置字典，需要包含以下字段:
            - api_url: API地址
            - model_name: 模型名称
            - input_file: 输入Excel文件路径
            - output_file: 输出Excel文件路径
            - system_prompt_mcp: MCP系统提示词
            - system_prompt_uliya: Uliya系统提示词
            - input_field: 输入字段名
            - ground_truth: Ground Truth字段名
            - mode: 模式选择 ('hybrid' 或 'split')
            - mcp_intent_list: MCP intent列表 (split模式需要)
            - uliya_intent_list: Uliya intent列表 (split模式需要)
    Returns:
        DataFrame
    """
    print("------begin time (API)------", time.strftime("%Y-%m-%d %H:%M:%S"))

    with open(config.get("tool_path")) as f:
        OPENAI_TOOLS_DATA = json.load(f)

    # 读取数据并预处理
    df = pd.read_excel(config["input_file"])
    # df = df.iloc[0:10]
    df = preprocess_data(df, config)

    # 获取模式配置
    mode = config.get("mode", "hybrid")
    print(f"Running in {mode} mode")

    system_prompt_fc = (
        "You are a helpful assistant. "
        "If a function is required, respond ONLY with tool_calls. "
        "Do not output any other text."
    )

    # 定义API调用函数
    api_url = config.get("api_url", "http://192.168.111.3:11434/api/chat")
    model_name = config.get("model_name")

    tool_path = config.get("tool_path")

    df["llm_response"] = df[config.get("input_field")].apply(
        lambda x: get_ollama_response_chat(system_prompt_fc, x, api_url, model_name, tool_path, OPENAI_TOOLS_DATA)
    )

    # 保存结果
    df.to_excel(config["output_file"], index=False)
    print("API LLM inference done, result saved:", config["output_file"])
    print("------end time (API)------", time.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def batch_llm_predict_api(config):
    """
    使用config.yaml批量调用LLM进行预测
    Args:
        config: 配置字典，需要包含以下字段:
            - api_url: API地址
            - model_name: 模型名称
            - input_file: 输入Excel文件路径
            - output_file: 输出Excel文件路径
            - system_prompt_mcp: MCP系统提示词
            - system_prompt_uliya: Uliya系统提示词
            - input_field: 输入字段名
            - ground_truth: Ground Truth字段名
            - mode: 模式选择 ('hybrid' 或 'split')
            - mcp_intent_list: MCP intent列表 (split模式需要)
            - uliya_intent_list: Uliya intent列表 (split模式需要)
    Returns:
        DataFrame
    """
    print("------begin time (API)------", time.strftime("%Y-%m-%d %H:%M:%S"))

    # 读取数据并预处理
    df = pd.read_excel(config["input_file"])
    # df = df.iloc[0:30]
    df = preprocess_data(df, config)

    # 获取模式配置
    mode = config.get("mode", "hybrid")
    print(f"Running in {mode} mode")

    # 初始化 pandarallel

    pandarallel.initialize()

    # 定义API调用函数
    api_url = config.get("api_url", "http://192.168.111.3:11434/api/chat")
    model_name = config.get("model_name")

    def run_one(text, system_prompt):
        return get_llm_response_api(
            text=text,
            system_prompt=system_prompt,
            api_url=api_url,
            model_name=model_name,
        )

    # 根据模式执行推理
    if mode == "hybrid":
        df = _run_hybrid_mode_api(df, run_one, config)
    elif mode == "split":
        df = _run_split_mode_api(df, run_one, config)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'hybrid' or 'split'.")

    # 保存结果
    df.to_excel(config["output_file"], index=False)
    print("API LLM inference done, result saved:", config["output_file"])
    print("------end time (API)------", time.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def _run_hybrid_mode_api(df, run_one, config):
    """API混合模式：先用 mcp 请求，如果没有结果，再用 uliya 请求"""
    print(
        "Using hybrid mode: trying mcp with system_prompt_mcp first, then uliya with system_prompt_uliya if no result"
    )

    system_prompt_mcp = config.get("system_prompt_mcp", "")
    system_prompt_uliya = config.get("system_prompt_uliya", "")

    # 第一轮：用 mcp
    df["llm_response_ori_mcp"] = df["mcp_input"].apply(lambda x: run_one(x, system_prompt_mcp))
    df["llm_response_mcp"] = df["llm_response_ori_mcp"].apply(parse_llm_response)

    # 检查哪些需要用 uliya
    def need_uliya_check(result):
        if isinstance(result, dict):
            if result.get("intent", "") in ["", "unknown"]:
                return True
        elif isinstance(result, str) and not result.strip():
            return True
        elif not result:
            return True
        return False

    df["need_uliya"] = df["llm_response_mcp"].apply(need_uliya_check)

    # 第二轮：对需要的行用 uliya
    uliya_rows = df[df["need_uliya"]]
    if len(uliya_rows) > 0:
        print(f"Re-running {len(uliya_rows)} rows with uliya_input and system_prompt_uliya")
        df.loc[df["need_uliya"], "llm_response_ori_uliya"] = uliya_rows["uliya_input"].apply(
            lambda x: run_one(x, system_prompt_uliya)
        )
        df.loc[df["need_uliya"], "llm_response_uliya"] = df.loc[df["need_uliya"], "llm_response_ori_uliya"].apply(
            parse_llm_response
        )

        # 合并结果：如果需要uliya则用uliya的结果，否则用mcp的结果
        df["llm_response_ori"] = df.apply(
            lambda row: row["llm_response_ori_uliya"]
            if row["need_uliya"] and pd.notna(row.get("llm_response_ori_uliya"))
            else row["llm_response_ori_mcp"],
            axis=1,
        )
        df["llm_response"] = df.apply(
            lambda row: row["llm_response_uliya"]
            if row["need_uliya"] and pd.notna(row.get("llm_response_uliya"))
            else row["llm_response_mcp"],
            axis=1,
        )
    else:
        df["llm_response_ori"] = df["llm_response_ori_mcp"]
        df["llm_response"] = df["llm_response_mcp"]

    return df


def _run_split_mode_api(df, run_one, config):
    """API分割模式：根据 gt_intent 选择对应的 prompt"""
    print("Using split mode: selecting prompt based on gt_intent")

    system_prompt_mcp = config.get("system_prompt_mcp", "")
    system_prompt_uliya = config.get("system_prompt_uliya", "")
    mcp_intent_list = config.get("mcp_intent_list", [])
    uliya_intent_list = config.get("uliya_intent_list", [])

    def select_system_prompt(intent):
        if intent in mcp_intent_list:
            return system_prompt_mcp
        elif intent in uliya_intent_list:
            return system_prompt_uliya
        else:
            # 如果不在任何列表中，默认使用 mcp
            print(f"Intent '{intent}' not in any list, using system_prompt_mcp by default")
            return system_prompt_mcp

    df["selected_system_prompt"] = df["gt_intent"].apply(select_system_prompt)
    df["selected_input"] = df.apply(
        lambda row: row["mcp_input"] if row["selected_system_prompt"] == system_prompt_mcp else row["uliya_input"],
        axis=1,
    )

    # 使用选定的 system_prompt 进行推理
    df["llm_response_ori"] = df.apply(lambda row: run_one(row["selected_input"], row["selected_system_prompt"]), axis=1)
    df["llm_response"] = df["llm_response_ori"].apply(parse_llm_response)

    return df


def main():
    # 单条测试
    system_prompt_mcp = """You are an intent recognition and slot extraction assistant.
Your tasks are:

1. Identify the user’s intent (`intent`);
2. Extract the corresponding slots (`slots`) from the user’s input.

Please strictly follow the output requirements below:

* The output must always use JSON format:

```
{
  "intent": "<intent_name>",
  "slots": {
    "<slot1>": "<value>",
    "<slot2>": "<value>"
  }
}
```

* If a slot is not mentioned in the user’s input, omit it. Do not output empty strings or null values.
* If the intent cannot be recognized, output:

```
{
  "intent": "unknown",
  "slots": {}
}
```

---

## Intent and Slot Definitions

1. **create_album**: Create a photo album

   * Slots:

     * `album_name`: the name of the album
     * `album_type`: the type of album. Default value: `normal` (regular album)

2. **search_photos**: Search for photos

   * Slots:

     * `keywords`: a description of the photo, e.g., \
         "photos taken last December", "photos about soccer", “photos at the beach,” “photos from the amusement park”

3. **get_album_list**: Retrieve albums

   * Slots:

     * `album_type`: the type of album. Possible values:

       * `normal`: regular album
       * `face`: people album
       * `baby`: baby album
       * `condition`: conditional album (e.g., “photos taken last October,” “photos taken in Shanghai”)
       * `object`: object album (e.g., “cat album,” “dog album”)

4. **music_play_control**: Music playback

   * Slots:

     * `title`: the name of a song, album, artist, or playlist
     * `source`: music source. Possible values:

       * `recent`: recently played
       * `favorites`: favorites
     * `play_mode`: playback mode. Possible values:

       * `normal`: sequential
       * `random`: shuffle
       * `single`: repeat single track
       * `loop`: repeat all tracks

5. **music_settings_control**: Music player settings

   * Slots:

     * `auto_stop_time`: the auto-stop time, e.g., 30, 1

6. **video_search_control**: Search for videos

   * Slots:

     * `title`: video description，e.g., video name, video style, or movie star
     * `type`: video type. Possible values:

       * `tv`: TV series/dramas
       * `movie`: films/blockbusters
       * `collection`: movie series/collections

7. **video_play_control**: Play video content

   * Slots:

     * `title`: video description， e.g., video name, video style, or movie star
     * `type`: video type. Possible values:

       * `tv`: TV series/dramas
       * `movie`: films/blockbusters
       * `collection`: movie series/collections

8. **get_system_info**: Get system or device information

   * Slots:

     * `system_type`: category of system or device information. Possible values:

       * `system`: system info
       * `device`: device info
       * `storage`: storage info
       * `network`: network info
       * `uglink`: UGREEN Link related info

"""

    system_prompt_5intent = """
You are an NAS intent classifier. \
    You need to accurately categorize the user's input into one of the following five intent categories. \
    **Only output the category name**, and reply in English.

### Intent Categories and Definitions
1. **general_knowledge_query**: Questions about general knowledge or topics not related to NAS.
   *Example*: "What is the capital of France?"
2. **search_photos**: Search for photos or images based on content, tags, or keywords. \
    Typically includes words like “picture”, “photo”, or “image”.
   *Example*: "Find photos from my beach vacation."
3. **summary_document**: Summarize the content of a document or report.
   *Example*: "Summarize the quarterly report for me."
4. **search_document**: Locate specific documents or files based on keywords. \
    Often includes terms like “document”, “file”, or “report”.
   *Example*: "Find the 2023 financial report."
5. **translate**: Translate text or documents into a specified language.
   *Example*: "Translate the user manual into Spanish."

### Examples
input: "Which country makes Casio watches?" → general_knowledge_query
input: "Translate 'Hello' into English." → translate_text
input: "Search for photos of the sky." → search_photos
input: "Find all files labeled 'update'." → search_document
input: "Summarize the main idea of this document." → summary_document

User input:

"""
    print(f"five intent: \n{system_prompt_5intent}")
    system_prompt = system_prompt_mcp  # system_prompt_5intent  # fallback

    API_URL = "http://192.168.111.3:11434/api/chat"
    model_name = "mcp_intent_1016:f16"
    input_ls = ["播放犯罪题材剧集", "请帮我找到《哈利波特》系列。"]
    for input_text in input_ls:
        try:
            rsp = get_llm_response_api(
                text=input_text, system_prompt=system_prompt, api_url=API_URL, model_name=model_name
            )
            print("单条调用结果：", rsp)
        except Exception as e:
            print("单条API调用异常：", e)


if __name__ == "__main__":
    main()
