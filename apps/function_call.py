# MIT License
#
# Copyright (c) 2025, Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import asyncio

from dotenv import load_dotenv

from openllm_func_call_synthesizer.utils import convert_to_mcp_tools, convert_to_openai_tools, get_mcp_tools
from openllm_func_call_synthesizer.utils.utils import write_json_file

load_dotenv()


def main():
    loop = asyncio.get_event_loop()
    cfg = {"transport": "http://192.168.56.200:12000/mcp"}
    cfg = {"transport": "http://192.168.111.9:12000/mcp"}
    mcp_tools = loop.run_until_complete(get_mcp_tools(cfg))
    openai_format_tools = convert_to_openai_tools(mcp_tools)
    write_json_file(openai_format_tools["tools"], "openai_tools.json")
    mcp_format_tools = convert_to_mcp_tools(openai_format_tools["tools"])
    write_json_file(mcp_format_tools["tools"], "mcp_tools.json")
    from rich import pretty

    pretty.pprint(openai_format_tools)

    from openai import OpenAI

    # client = OpenAI(base_url="http://192.168.111.6:8000/v1", api_key="dummy")
    client = OpenAI(base_url="http://192.168.111.4:8010/v1", api_key="dummy")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    # client = OpenAI()
    while True:
        user_input = input("请输入你的问题：")
        if user_input == "exit":
            break

        new_message = [
            {"role": "user", "content": user_input},
        ]
        messages.extend(new_message)

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
            model="qwen3_1.7b_mix",
            messages=messages,
            tools=openai_format_tools["tools"],  # ⚡ 告诉模型有哪些工具可调用
        )
        print(response.choices[0].message)
        messages.append(
            {
                "role": response.choices[0].message.role,
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
            }
        )


if __name__ == "__main__":
    main()
