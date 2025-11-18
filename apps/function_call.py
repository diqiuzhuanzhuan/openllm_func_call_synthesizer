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


import asyncio

from dotenv import load_dotenv

from openllm_func_call_synthesizer.utils import convert_to_mcp_tools, convert_to_openai_tools, get_mcp_tools
from openllm_func_call_synthesizer.utils.utils import write_json_file

load_dotenv()


def main():
    loop = asyncio.get_event_loop()
    cfg = {"transport": "http://192.168.111.9:12000/mcp"}
    mcp_tools = loop.run_until_complete(get_mcp_tools(cfg))
    openai_format_tools = convert_to_openai_tools(mcp_tools)
    mcp_format_tools = convert_to_mcp_tools(openai_format_tools["tools"])
    write_json_file(mcp_format_tools["tools"], "mcp_tools.json")
    from rich import pretty

    pretty.pprint(openai_format_tools)

    from openai import OpenAI

    client = OpenAI(base_url="http://192.168.111.6:8000/v1", api_key="dummy")
    # client = OpenAI()
    while True:
        user_input = input("请输入你的问题：")
        if user_input == "exit":
            break

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]

        if not openai_format_tools:
            openai_format_tools = {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "获取城市天气",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ]
            }

        response = client.chat.completions.create(
            model="qwen",
            messages=messages,
            tools=openai_format_tools["tools"],  # ⚡ 告诉模型有哪些工具可调用
        )
        print(response)


if __name__ == "__main__":
    main()
