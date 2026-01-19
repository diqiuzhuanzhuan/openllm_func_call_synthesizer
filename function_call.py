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
from transformers import AutoTokenizer
import ollama
# 使用本地模型路径，确保断网也能运行，且模板匹配
# 路径取自您的 vllm_start.sh
tokenizer = AutoTokenizer.from_pretrained("/data/work/public/SusieSu/workspace/LLaMA-Factory-main/saves/qwen3_1.7b_0115_function_call_batch8/sft/checkpoint-600", trust_remote_code=True)
import sys
sys.path.append("/data/work/CHenXuFei/openllm_func_call_synthesizer/src")

from openllm_func_call_synthesizer.utils import convert_to_mcp_tools, convert_to_openai_tools, get_mcp_tools
from openllm_func_call_synthesizer.utils.utils import write_json_file

load_dotenv()


def main():
    loop = asyncio.get_event_loop()
    cfg = {"transport": "http://192.168.56.200:12000/mcp"}
    cfg = {"transport": "http://192.168.111.11:8000/mcp"}
    mcp_tools = loop.run_until_complete(get_mcp_tools(cfg))
    openai_format_tools = convert_to_openai_tools(mcp_tools)
    write_json_file(openai_format_tools["tools"], "./openai_tools.json")
    mcp_format_tools = convert_to_mcp_tools(openai_format_tools["tools"])
    write_json_file(mcp_format_tools["tools"], "./mcp_tools.json")
    from rich import pretty

    pretty.pprint(openai_format_tools)

    from openai import OpenAI
    # client = OpenAI(base_url="http://192.168.111.6:8000/v1", api_key="dummy")
    client = OpenAI(base_url="http://192.168.111.11:11434/v1", api_key="dummy")

    client = ollama.Client(host="http://192.168.111.11:11434")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},  
    ]
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

        # 使用 ollama.complete 调用模型
        try:
            prompt = tokenizer.apply_chat_template(messages, tools=openai_format_tools["tools"], tokenize=False, add_generation_prompt=True)
            print(prompt)
            response = client.generate(
                model="function_call_old_format_0116-q4_K_M",
                prompt=prompt, 
                think=False,
                options={
                "temperature": 0.1,
                "repeat_penalty": 1.0
                }
            )
            print("Assistant: ",  response.response)
        except Exception as e:
            print(f"ollama.complete 调用失败: {e}")
            # 回退到 OpenAI 客


if __name__ == "__main__":
    main()

