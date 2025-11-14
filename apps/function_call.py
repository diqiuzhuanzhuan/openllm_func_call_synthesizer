# UGREEN License
#
# Copyright (c) 2025 UGREEN. All Rights Reserved.
#
# This software and associated documentation files (the "Software") are
# the proprietary information of UGREEN.
#
# The Software is provided solely for internal use within UGREEN
# and may not be copied, modified, distributed, or disclosed to any
# third party without prior written consent from UGREEN.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.


from openllm_func_call_synthesizer.utils import get_mcp_tools, convert_to_openai_tools
from litellm import completion, Chat


import asyncio

def main():
    loop = asyncio.get_event_loop()
    cfg = {"transport": "http://192.168.111.9:12000/mcp"}
    mcp_tools = loop.run_until_complete(get_mcp_tools(cfg))
    openai_format_tools = convert_to_openai_tools(mcp_tools)

    from openai import OpenAI


    client = OpenAI(base_url="http://192.168.111.3:8000/v1", api_key="dummy")

    messages = [
        {"role": "user", "content": "播放周杰伦的七里香"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="qwen",
        messages=messages,
        tools=openai_format_tools['tools']  # ⚡ 告诉模型有哪些工具可调用
    )
    print(response)
        

if __name__ == "__main__":
    main()
