import json

import pandas as pd


def test_one_response(messages):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = "/data0/work/SusieSu/project/openllm_func_call_synthesizer/src/openllm_func_call_synthesizer/data_process/mcp_dataprocess_1025/train_datas_1025"
    model_name = (
        "/data0/work/SusieSu/project/workspace/llama/LLaMA-Factory/saves/qwen3_1.7b_1030_intent_mcp/sft/checkpoint-240"
    )

    # load the tokenizer and the models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

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

with open("/data0/work/SusieSu/project/uliya/mcp/function_docs.json") as f:
    fun_ = json.load(f)

FUNCTIONS = fun_["tools"]
FUNCTIONS = json.dumps(FUNCTIONS)


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
                name = tool["function"].get("name")
                arguments = tool["function"].get("arguments")
            else:
                name = tool.get("name")
                arguments = tool.get("arguments")
            tc = {
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
            tool_calls.append(tc)
    if not tool_calls:
        return []
    return tool_calls


def get_answer_content(answer):
    try:
        ans_obj = answer
        if isinstance(answer, str):
            # 先尝试用ast.literal_eval解析
            try:
                ans_obj = ast.literal_eval(answer)
            except Exception:
                # print("ast.literal_eval error:", e)
                # 如果失败但answer像字典字符串，则用正则取content字段
                m = re.search(r"'content'\s*:\s*([\"'])(.*?)\1", answer)
                if m:
                    content_val = m.group(2)
                else:
                    content_val = ""
                ans_obj = answer  # 继续保持为字符串，以兼容下方结构
        if content_val == "":
            if isinstance(ans_obj, dict):
                content_val = ans_obj.get("content")
            elif isinstance(ans_obj, str):
                # 如果还没有找到，尝试再次用正则
                m = re.search(r"'content'\s*:\s*([\"'])(.*?)\1", ans_obj)
                if m:
                    content_val = m.group(2)
                else:
                    print("ans_obj is not dict, it is:", type(ans_obj))
                    content_val = ""
            else:
                print("ans_obj is not dict, it is:", type(ans_obj))
                content_val = ""
    except Exception as e:
        print("Exception occurred:", e)
        content_val = ""

    return content_val


def make_message_row_simple(row):
    """
    构造给定行的 chat message 格式，支持function_call转为tool_calls
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can call tools.", "loss_weight": 0.0},
        {"role": "user", "content": str(row["query"]) if pd.notnull(row.get("query")) else "", "loss_weight": 0.0},
    ]

    has_fc = pd.notnull(row.get("function_call")) and str(row["function_call"]).strip() != ""

    if has_fc:
        # 用function_call生成tool_calls
        tool_calls = parse_function_call(row["function_call"])
        assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls, "loss_weight": 1.0}
    else:
        answer_content = get_answer_content(row.get("answer"))
        assistant_msg = {"role": "assistant", "content": answer_content, "tool_calls": [], "loss_weight": 1.0}

    messages.append(assistant_msg)

    return {"messages": messages, "functions": FUNCTIONS}


if __name__ == "__main__":
    df = pd.read_csv(
        "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/function_call_critic_1112_v2/output.csv"
    )
    print(df.shape, df.columns)
    print(df["score"].value_counts().sort_index(ascending=False))
    df_none = df[df["function_call"].isnull() | (df["function_call"].astype(str).str.strip() == "")]
    print("df_none.shape", df_none.shape)

    # 应用到df，加到新的一列里
    df = df.drop_duplicates(subset=["query"])
    print("dedup df.shape", df.shape)

    df = df[df["score"] > 4]
    print("score > 4 df.shape", df.shape)

    df["lora_input"] = df.apply(make_message_row_simple, axis=1)

    for i in range(2):
        print(json.dumps(df.iloc[i]["lora_input"], ensure_ascii=False, indent=2))

    df.to_excel(
        "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/function_call_for_train_1112_v2/function_call_for_train_1112.xlsx"
    )
