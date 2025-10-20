#!/usr/bin/env python
# coding: utf-8

"""
API功能测试脚本
用于测试新添加的API调用功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_local_model import get_llm_response_api

# 测试配置
API_URL = "http://192.168.111.3:11434/api/chat"
MODEL_NAME = "mcp_intent_1016:q4_k_m"

SYSTEM_PROMPT = """You are an intent recognition and slot extraction assistant.
Your tasks are:

1. Identify the user's intent (`intent`);
2. Extract the corresponding slots (`slots`) from the user's input.

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

* If a slot is not mentioned in the user's input, omit it. Do not output empty strings or null values.
* If the intent cannot be recognized, output:

```
{
  "intent": "unknown",
  "slots": {}
}
```

---

## Intent and Slot Definitions

1. **music_play_control**: Music playback
   * Slots:
     * `title`: the name of a song, album, artist, or playlist

2. **video_play_control**: Play video content
   * Slots:
     * `title`: video description

3. **search_photos**: Search for photos
   * Slots:
     * `keywords`: a description of the photo
"""

# 测试用例
test_cases = [
    "我想找周杰伦的歌",
    "我想找电影阿凡达",
    "我想听神话这首歌"
]

def test_api_function():
    """测试API调用功能"""
    print("=" * 50)
    print("开始测试API调用功能")
    print("=" * 50)
    print(f"\nAPI地址: {API_URL}")
    print(f"模型名称: {MODEL_NAME}")
    print()
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n【测试用例 {i}】")
        print(f"输入: {test_input}")
        print("-" * 50)
        
        response = get_llm_response_api(
            text=test_input,
            system_prompt=SYSTEM_PROMPT,
            api_url=API_URL,
            model_name=MODEL_NAME,
            top_p=0.1,
            temperature=0.01
        )
        
        if response:
            print(f"响应: {response}")
        else:
            print("API调用失败或返回为空")
        
        print("-" * 50)
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_api_function()

