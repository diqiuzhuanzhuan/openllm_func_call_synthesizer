import json
import re

from ollama import Client

system_prompt = (
    "You are a helpful assistant. "
    "You are given a query and a function call. "
    "You need to determine if the function call is correct for the query."
)

with open("/data/work/SusieSu/project/openllm_func_call_synthesizer/openai_tools.json") as f:
    OPENAI_TOOLS = json.load(f)

# client = OpenAI(base_url="http://192.168.111.3:8019/v1", api_key="dummy")

client = Client(host="http://192.168.111.3:11434")
model_name = "function_call_1216-q4_K_M"
# model_name = "qwen3:1.7b"
# model_name = "function_call_test"
# filter_think 用到了 re


def filter_think(text):
    try:
        rs = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return rs.strip()
    except Exception:
        return text.strip()


def get_client_ollama_response_generate(text):
    response = client.generate(
        model=model_name,
        prompt=text,
        options={"temperature": 0.01},
    )
    print("----------response-----------")
    print(response["response"])
    return response["response"]
def get_client_ollama_response_chat(system_prompt, text):
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.01},
        tools=OPENAI_TOOLS,
        format="",
    )
    # # ✅  Ollama 正确取法
    msg = response["message"]
    print('----------response["message"]-----------', msg)
    # 处理content中的tool_call结构，提取出来放到tool_calls字段，content去掉<tool_call>部分
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls")
    tool_calls_list = None

    # 提取content中的<tool_call>部分，并解析成json
    tool_call_match = re.search(r"<tool_call>\s*(.+?)\s*</tool_call>", content, flags=re.DOTALL)
    if tool_call_match:
        tool_call_block = tool_call_match.group(1)
        # 支持多行多个json
        tool_calls_list = []
        for line in tool_call_block.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    tool_calls_list.append(json.loads(line))
                except Exception:
                    # 出错就按原样字符串放回去
                    tool_calls_list.append(line)
        # 剔除<tool_call>部分
        filtered_content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
    else:
        filtered_content = filter_think(content)

    result = {
        "role": msg["role"],
        "content": filtered_content,
        "tool_calls": tool_calls_list if tool_calls_list is not None else msg.get("tool_calls"),
    }
    print('===============result===============\n', result)

    return result


# input = "你好"
input = "帮我创建个名字叫小狗的相册"
get_client_ollama_response_chat(system_prompt, input)