#!/usr/bin/env python
# coding: utf-8

# 本代码 读取 rerank之后的结果， 处理数据为 背景知识+问题  
# 调用chat模型，生成回复  

# # 数据预处理

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pandas as pd
import requests
import json
import time
from typing import List, Dict
from openai import OpenAI
import re
import pandas as pd
# from LLM_result2excel import json_to_excel_answers
import time

API_URL = "http://192.168.111.3:11434/api/chat"
API_URL_generate = "http://192.168.111.3:11434/api/generate"

MODEL_NAME = "mcp_intent_1016:q4_k_m"  #"mcp_intent_1016:f16"   # 模型名称（若服务需要）

def get_search_results_text(search_results):
    all_text = []
    search_results1 = eval(search_results) if type(search_results) == str else search_results
    if type(search_results1) == list:
        all_text = [rs['text'] for rs in eval(search_results)]
        return all_text
    else:
        return []
    
def filter_think(text):
    try:
        rs = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return rs.strip()
    except:
        return text.strip()

def get_llm_response_chat(text, system_prompt):
    beginn = time.time()

    system_prompt = system_prompt

    payload = json.dumps({
    "model": MODEL_NAME,
    "stream": False,
    "top_p": 0.1, 
    "temperature": 0.01,
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
    # 发送 POST 请求
    response = requests.request("POST", API_URL, headers=headers, data=payload)
    # print(response.text)

    # 处理响应
    if response.status_code == 200:
        #print("生成结果：",response.text)

        generated_text = json.loads(response.text).get("message",{}).get('content',{})

        print("生成结果：\n", generated_text)
        endd = time.time()
        print(f"----use time----- {endd-beginn} 秒")
        return generated_text
    else:
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")
        return None 


def get_llm_response_generate(text, API_URL_generate):
    beginn = time.time()

    payload = json.dumps({
    "model": MODEL_NAME,
    "prompt": text,
    "stream": False,
    "options": {
    # "num_ctx": 32768,
    # "num_predict": 16384
    "num_ctx": 4096,
    "num_predict": 4096
    }
    })
    headers = {
    'Content-Type': 'application/json'
    }
    # 发送 POST 请求
    response = requests.request("POST", API_URL_generate, headers=headers, data=payload)
    print(response.text)

    # 处理响应
    if response.status_code == 200:
        #print("生成结果：",response.text)

        generated_text = json.loads(response.text).get("response",{})

        print("生成结果：\n", generated_text)
        endd = time.time()
        print(f"----use time----- {endd-beginn} 秒")
        return generated_text
    else:
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")
        return None 

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

input_ls_zh = ["我想找周杰伦的歌", "我想找电影阿凡达", "我想听神话这首歌"]

input_ls_zh_v2 = ["西红柿炒鸡蛋有几种做法？", "帮我总结这个文档", "帮我找下python相关文件"]
# system_prompt_mcp    system_prompt_5intent

for input in input_ls_zh:
    get_llm_response_chat(input, system_prompt_mcp)

# for input in input_ls_zh_v2:
#     print('-----input-----', input)
#     get_llm_response_chat(input, system_prompt_5intent)