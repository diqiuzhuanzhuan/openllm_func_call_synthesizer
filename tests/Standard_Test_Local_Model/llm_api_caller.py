#!/usr/bin/env python
# coding: utf-8

"""
LLM API调用模块
支持main函数，直接用config.yaml测评，也支持单条手动修改数据测试
"""

import json
import time
import requests
import pandas as pd
import re
import yaml
from parse_response_to_json import parse_react_to_json

def filter_think(text):
    try:
        rs = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return rs.strip()
    except Exception:
        return text.strip()

def get_llm_response_api(
    text,
    system_prompt,
    api_url,
    model_name,
    temperature=0.01
    ):
    """
    使用API方式调用LLM
    """
    begin_time = time.time()

    payload = json.dumps({
        "model": model_name,
        "stream": False,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", api_url, headers=headers, data=payload)
        if response.status_code == 200:
            generated_text = json.loads(response.text).get("message", {}).get('content', {})
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

def batch_llm_predict_api(config):
    """
    使用config.yaml批量调用LLM进行预测
    Args:
        config: 配置字典
    Returns:
        DataFrame
    """
    print('------begin time (API)------', time.strftime("%Y-%m-%d %H:%M:%S"))
    api_url = config.get('api_url', 'http://192.168.111.3:11434/api/chat')
    model_name = config.get('model_name')
    system_prompt = config.get('system_prompt', '')
    input_file = config['input_file']
    output_file = config['output_file']
    input_field = config['input_field']

    # 读取数据
    df = pd.read_excel(input_file)
    df['all_prompt_to_test'] = df[input_field].astype(str)

    def run_one(text):
        rsp = get_llm_response_api(
            text=text,
            system_prompt=system_prompt,
            api_url=api_url,
            model_name=model_name,
        )
        return rsp

    df['llm_response_ori'] = df['all_prompt_to_test'].apply(run_one)

    df['llm_response'] = df['llm_response_ori'].apply(parse_llm_response)

    df.to_excel(output_file, index=False)
    print("API LLM inference done, result saved:", output_file)
    print('------end time (API)------', time.strftime("%Y-%m-%d %H:%M:%S"))
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

     * `keywords`: a description of the photo, e.g., "photos taken last December", "photos about soccer", “photos at the beach,” “photos from the amusement park”

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
You are an NAS intent classifier. You need to accurately categorize the user's input into one of the following five intent categories. **Only output the category name**, and reply in English.

### Intent Categories and Definitions
1. **general_knowledge_query**: Questions about general knowledge or topics not related to NAS.  
   *Example*: "What is the capital of France?"
2. **search_photos**: Search for photos or images based on content, tags, or keywords. Typically includes words like “picture”, “photo”, or “image”.  
   *Example*: "Find photos from my beach vacation."
3. **summary_document**: Summarize the content of a document or report.  
   *Example*: "Summarize the quarterly report for me."
4. **search_document**: Locate specific documents or files based on keywords. Often includes terms like “document”, “file”, or “report”.  
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
    system_prompt = system_prompt_mcp  #system_prompt_5intent  # fallback

    API_URL = "http://192.168.111.3:11434/api/chat"
    model_name = "mcp_intent_1016:f16"
    input_ls = ["播放犯罪题材剧集", "请帮我找到《哈利波特》系列。"]
    for input_text in input_ls:
        try:
            rsp = get_llm_response_api(
                text=input_text,
                system_prompt=system_prompt,
                api_url=API_URL,
                model_name=model_name)
            print("单条调用结果：", rsp)
        except Exception as e:
            print("单条API调用异常：", e)

if __name__ == "__main__":
    main()