import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
import json
import re

from openai import OpenAI

system_prompt = (
    "You are a helpful assistant. "
    "You are given a query and a function call. "
    "You need to determine if the function call is correct for the query."
)

with open("/data0/work/SusieSu/project/openllm_func_call_synthesizer/openai_tools.json") as f:
    openai_format_tools = json.load(f)

client = OpenAI(base_url="http://192.168.111.3:8019/v1", api_key="dummy")
# model_name = "qwen3_1.7b_1215_function_call"
model_name = "qwen3_1.7b_1215_function_call"


def filter_think(text):
    try:
        rs = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return rs.strip()
    except Exception:
        return text.strip()


def get_one_vllm_response(system_prompt, text):
    """
    调用 vLLM 服务，返回 choices 的解析结果列表
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        tools=openai_format_tools,  # 这里应是 openai_format_tools["tools"]，不是 openai_format_tools
        tool_choice="auto",  # auto => 模型可能决定用工具或不使用
        stream=False,
        temperature=0.01,
    )

    res_list = []
    print(
        "==========================response.choices[0].message==========================\n", response.choices[0].message
    )
    for choice in response.choices:
        rs = {
            "role": choice.message.role,
            "content": filter_think(choice.message.content),
            "tool_calls": getattr(choice.message, "tool_calls", None),
        }
        # print(rs)
        res_list.append(rs)
    return res_list


def parse_vllm_multi_response(rs_list):
    """
    解析 get_one_vllm_response 返回的 rs_list，每一项输出 function call结构或 content。
    如果tool_calls非空，则取 function call 信息；否则直接返回content内容。
    """
    parsed_list = []
    for rs in rs_list:
        tool_calls = rs.get("tool_calls", None)
        if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
            function_call = tool_calls[0].function
            function_args = function_call.arguments  # 参数内容（通常是json字符串）
            function_name = function_call.name
            try:
                function_args_dict = eval(function_args)
            except Exception:
                function_args_dict = function_args
            format_rs = {"name": function_name, "arguments": function_args_dict}
            parsed_list.append(format_rs)
        else:
            parsed_list.append(rs.get("content", None))
    return parsed_list


def get_vllm_response(x, system_prompt):
    try:
        rs_list = get_one_vllm_response(system_prompt, x)
        parsed_rs_list = parse_vllm_multi_response(rs_list)
        print("------parsed_rs_list------\n", parsed_rs_list[0])
        return parsed_rs_list[0]
    except Exception as e:
        print(e)
        return {"unknown": {}}


if __name__ == "__main__":
    system_prompt = (
        "You are a helpful assistant. "
        "You are given a query and a function call. "
        "You need to determine if the function call is correct for the query."
    )
    # # input = "我喜欢蔡健雅，你喜欢不啊"
    # # input = "我想找周杰伦的稻香"
    # input = "请播放一些古典音乐。"
    # # input = "给我翻译下这个句子好不好？"
    # rs = get_response(system_prompt, input, MODEL, TOKENIZER)
    # rs1 = parse_fuction_call_response(rs)
    # print('-------------------------\n',rs1,type(rs1))

    # df = pd.read_excel("/data0/work/SusieSu/project/openllm_datas_and_temp_codes/DPO_data/1208/test_all.xlsx")
    # print("df.shape, df.columns", df.shape, df.columns)

    # 调用示例
    input = "我想做一个家庭聚会的照片集。"
    vllm_result = get_one_vllm_response(system_prompt, input)

    # rs = get_vllm_response(input, system_prompt)
    print("================vllm_result================\n", vllm_result)

    # print("-------------\n", rs_list)

    # parsed_rs_list = parse_vllm_multi_response(rs_list)
    # print("------parsed_rs_list------\n")
    # print(parsed_rs_list)
