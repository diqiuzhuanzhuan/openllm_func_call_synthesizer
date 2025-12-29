import json
import re

from ollama import Client

client = Client(host="http://localhost:11434")
model_name = "function_call_1216-q4_K_M"

system_prompt_fc = (
    "You are a helpful assistant. "
    "If a function is required, respond ONLY with tool_calls. "
    "Do not output any other text."
)

with open("/data0/work/SusieSu/project/openllm_func_call_synthesizer/openai_tools.json") as f:
    OPENAI_TOOLS_DATA = json.load(f)


def filter_think(text):
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_ollama_response_generate(text):
    # response = client.chat.(
    response = client.generate(
        model=model_name,
        prompt=text,
        options={"temperature": 0.01},
    )
    print(response["response"])
    return response["response"]


def get_ollama_response_chat(system_prompt, text):
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
    print('====================response["message"]====================\n', msg)

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
        return parsed_tools if parsed_tools else None

    result = {
        # "role": msg["role"],
        "content": filter_think(msg.get("content")),
        "tool_calls": parse_tool_calls(msg.get("tool_calls")),
    }
    return result


if __name__ == "__main__":
    with open("/data0/work/SusieSu/project/openllm_func_call_synthesizer/openai_tools.json") as f:
        OPENAI_TOOLS_DATA = json.load(f)
    GENERATE_TOOLS = [aa["function"] for aa in OPENAI_TOOLS_DATA]

    # print('GENERATE_TOOLS------------', GENERATE_TOOLS)

    ss = "\n".join(str(func) for func in GENERATE_TOOLS) + "\n"
    # print('------------ss------------')
    # print(ss)

    # 测试
    input = "北京今天天气在呢么样？"
    # input = "帮我创建一个叫小狗的相册"
    # input = "我想播放周杰伦的稻香并且创建一个小狗的相册"
    # input = "帮我找点花朵的照片吧"
    # input = "找一个名叫宝宝的相册"
    # input = "What is the weather like in Shanghai ? And Beijing ?"
    # input = "你好"

    # test chat mode
    rs = get_ollama_response_chat(system_prompt_fc, input)

    # # test generate mode
    generate_prompt = """<|im_start|>system
    You are a helpful assistant

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {{GENERATE_TOOLS}}
     </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call>
    XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call><|im_end|>
    <|im_start|>user
    {{input_text}}<|im_end|>
    <|im_start|>assistant
    """
    final_prompt = generate_prompt.replace("{{input_text}}", input).replace("{{GENERATE_TOOLS}}", ss)

    rs = get_ollama_response_generate(final_prompt)

    # print(' ==================== rs ========================= \n', rs)
