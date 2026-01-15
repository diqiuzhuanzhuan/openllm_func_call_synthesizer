import ast
import json
import random
import re
import string
import uuid

import pandas as pd
key_input_column = "query"
key_output_column = "function_call" #function_call


rs_key_intent_name = "name" # name
rs_key_slot_name = "arguments" # arguments

def test_one_response(messages):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

    from transformers import AutoTokenizer

    # root = "/data0/work/SusieSu/project/openllm_func_call_synthesizer/\
    #    src/openllm_func_call_synthesizer/data_process/mcp_dataprocess_1025/train_datas_1025"
    model_name = (
        "/data0/work/SusieSu/project/workspace/llama/LLaMA-Factory/saves/qwen3_1.7b_1030_intent_mcp/sft/checkpoint-240"
    )

    # load the tokenizer and the models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    text = tokenizer.apply_chat_template(
        messages["messages"],
        tools=messages["functions"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    print(text)

import ast
import re

with open("/data/work/CHenXuFei/openai_tools.json") as f:
    TOOLS_SCHEMA = json.load(f)

# LLaMA-Factory / apply_chat_template 通常希望 tools 为数组；
# 这里保留原行为（写入字符串）以兼容你现有数据管线。
FUNCTIONS = json.dumps(TOOLS_SCHEMA)


def _build_tool_params_map(tools_schema):
    m = {}
    if not isinstance(tools_schema, list):
        return m
    for t in tools_schema:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        params = fn.get("parameters")
        if isinstance(name, str):
            m[name] = params if isinstance(params, dict) else {}
    return m


TOOL_PARAMS_MAP = _build_tool_params_map(TOOLS_SCHEMA)


_PLAY_MODE_MAP = {
    "normal": "normal",
    "default": "normal",
    "shuffle": "shuffle",
    "random": "shuffle",
    "repeat one": "repeat_one",
    "repeat 1": "repeat_one",
    "repeat all": "repeat_all",
    "repeat": "repeat_all",
    "loop": "repeat_all",
    "single": "repeat_one",
    "one": "repeat_one",
    "all": "repeat_all",
}


def _coerce_number(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    if isinstance(v, str):
        m = re.search(r"(-?\d+(?:\.\d+)?)", v)
        if not m:
            return None
        s = m.group(1)
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return None
    return None


def _normalize_arguments(name: str, arguments):
    """Normalize/repair arguments to better match openai_tools.json schema."""
    # arguments should be a dict; attempt parsing if it's a string
    args = arguments
    if args is None:
        args = {}
    if isinstance(args, str):
        s = args.strip()
        if not s:
            args = {}
        else:
            try:
                args = json.loads(s)
            except Exception:
                try:
                    args = ast.literal_eval(s)
                except Exception:
                    # can't parse -> treat as empty
                    args = {}

    if not isinstance(args, dict):
        return {}

    # --- targeted repairs discovered in 0114 ---
    if name == "music_play_control":
        if "play_mode" in args and isinstance(args["play_mode"], str):
            s = args["play_mode"].strip().lower().replace("-", " ").replace("_", " ")
            s = re.sub(r"\s+", " ", s)
            fixed = _PLAY_MODE_MAP.get(s)
            if fixed is None:
                # optional field; drop if invalid
                args.pop("play_mode", None)
            else:
                args["play_mode"] = fixed

    if name == "music_search_control":
        # schema requires keyword; some data used title
        if "keyword" not in args and "title" in args:
            args["keyword"] = args.pop("title")

    if name == "video_play_control":
        # schema doesn't have play_mode
        if "play_mode" in args:
            args.pop("play_mode", None)

    if name == "music_settings_control":
        # schema requires number
        if "auto_stop_time" in args:
            coerced = _coerce_number(args.get("auto_stop_time"))
            if coerced is not None:
                args["auto_stop_time"] = coerced
            # if can't coerce, keep original (let downstream validator decide)

    return args


def parse_function_call(fc_raw):
    """
    解析function_call字段为tool_calls格式，如果是有效JSON字符串/字典，自动适配结构。
    支持简单字符串/单条/多条function_call。
    """
    import ast

    if pd.isnull(fc_raw):
        return []

    # 先尝试直接作为JSON
    fc = fc_raw
    if isinstance(fc, str):
        fc = fc.strip()
        if not fc:
            return []
        # 尝试json或python字面量
        try:
            res = json.loads(fc)
            fc = res
        except Exception:
            try:
                # 支持可能是类似"[{...}]"的字符串
                res = ast.literal_eval(fc)
                fc = res
            except Exception:
                # 可能只是一个"{"name":...}"，尝试包裹成list
                if fc.startswith("{") and fc.endswith("}"):
                    try:
                        res = [json.loads(fc)]
                        fc = res
                    except Exception:
                        return []
                else:
                    return []

    # 现在fc应该是dict/list
    tool_calls = []
    if isinstance(fc, dict):
        fc = [fc]
    
    if isinstance(fc, list):
        for idx, tool in enumerate(fc, 1):
            # 如果是chatGPT格式有外层function字段
            if "function" in tool:
                name = tool["function"].get(rs_key_intent_name)
                arguments = tool["function"].get(rs_key_slot_name)
            else:
                name = tool.get(rs_key_intent_name)
                arguments = tool.get(rs_key_slot_name)

            if isinstance(name, str):
                arguments = _normalize_arguments(name, arguments)
            # 生成随机ID，格式如 call_eVyb2my3Ocade28Bi6RHJPdC
            random_id = 'call_' + ''.join(random.choices(string.ascii_letters + string.digits, k=24))
            # 按照图片格式：function内部arguments为对象，字段顺序为 function, id, type
            tc = {
                "function": {"arguments": arguments, "name": name},
                "id": random_id,
                "type": "function",
            }
            tool_calls.append(tc)
    if not tool_calls:
        return []
    return tool_calls


def get_answer_content(answer):
    """
    从answer字段提取content内容
    支持格式: {"content": "xxx", "role": "assistant"} 或 {'content': 'xxx', 'role': 'assistant'}
    """
    content_val = ""
    try:
        if pd.isnull(answer) or answer == "":
            return ""
        
        ans_obj = answer
        if isinstance(answer, str):
            answer = answer.strip()
            # 先尝试用json.loads解析（双引号格式）
            try:
                ans_obj = json.loads(answer)
            except Exception:
                # 再尝试用ast.literal_eval解析（单引号格式）
                try:
                    ans_obj = ast.literal_eval(answer)
                except Exception:
                    # 都失败了，用正则提取content字段（支持单引号和双引号）
                    m = re.search(r'["\']content["\']\s*:\s*["\'](.+?)["\']', answer)
                    if m:
                        content_val = m.group(1)
                    return content_val
        
        # 如果成功解析为dict，直接获取content
        if isinstance(ans_obj, dict):
            content_val = ans_obj.get("content", "") or ""
        
    except Exception as e:
        print("get_answer_content Exception:", e)
        content_val = ""

    return content_val


def make_message_row_simple(row):
    """
    构造给定行的 chat message 格式，支持function_call转为tool_calls
    按图片格式：字段顺序为 content, role, tool_calls
    """
    messages = [
        {"content": "You are a helpful assistant.", "role": "system", "tool_calls": []},
        {"content": str(row[key_input_column]) if pd.notnull(row.get(key_input_column)) else "", "role": "user", "tool_calls": []},
    ]

    has_fc = pd.notnull(row.get(key_output_column)) and str(row[key_output_column]).strip() != ""
    
    tool_calls = []
    if has_fc:
        # 用function_call生成tool_calls
        tool_calls = parse_function_call(row[key_output_column])
        print('----tool_calls-----', tool_calls)

    if tool_calls:
        assistant_msg = {"content": "", "role": "assistant", "tool_calls": tool_calls}
    elif pd.notnull(row.get("answer")):
        # 没有function_call，但有answer → 读取answer的content
        answer_content = get_answer_content(row.get("answer"))
        assistant_msg = {"content": answer_content, "role": "assistant", "tool_calls": []}
    else:
        assistant_msg = {"content": "", "role": "assistant", "tool_calls": []}

    messages.append(assistant_msg)

    return {"messages": messages, "tools": FUNCTIONS}


if __name__ == "__main__":
    # df = pd.read_csv(
    #     "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/function_call_critic_1112_v2/output.csv"
    # )
    # 读取合并去重后的数据
    df = pd.read_excel('/data/work/CHenXuFei/data/function_call_data/train_data_fc_0114_fixed/raw_function_call_data_0114_processed.xlsx')
    print(df.shape, df.columns)
    # print(df["score"].value_counts().sort_index(ascending=False))
    

    df_none = df[df[key_output_column].isnull() | (df[key_output_column].astype(str).str.strip() == "")]
    print("df_none.shape", df_none.shape)

    # 应用到df，加到新的一列里
    # df = df.drop_duplicates(subset=["query"])
    df = df.drop_duplicates(subset=[key_input_column])
    print("dedup df.shape", df.shape)

    # df = df[df["score"] > 4]
    # print("score > 4 df.shape", df.shape)

    df["lora_input"] = df.apply(make_message_row_simple, axis=1)

    for i in range(2):
        print(json.dumps(df.iloc[i]["lora_input"], ensure_ascii=False, indent=2))
    
    # 保存到原文件（覆盖，添加lora_input列）
    df.to_excel(
        "/data/work/CHenXuFei/data/function_call_data/train_data_fc_0114_fixed/raw_function_call_data_0114_processed_with_messages.xlsx",
        index=False
    )
    print(f"已保存，共 {len(df)} 条数据")
    

