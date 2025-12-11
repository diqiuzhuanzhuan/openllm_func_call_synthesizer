import json
import re

import pandas as pd

from openllm_func_call_synthesizer.utils import check_intent_arguments


def read_jsonl(file_path):
    """通用的读取jsonl文件为列表字典格式的数据"""
    data = []
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    if len(data) > 1:
        if isinstance(data[0], dict):
            print("data[0].keys()", data[0].keys())
    return data


def read_jsonl_to_df(file_path):
    data = []
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():  # 跳过空行
                data.append(json.loads(line))
    if len(data) > 1:
        if isinstance(data[0], dict):
            print("data[0].keys()", data[0].keys())
    df = pd.DataFrame(data)
    print(df.shape, df.columns)
    return df


def get_intent_from_function_call(function_call):
    try:
        return eval(function_call)[0]["name"], eval(function_call)[0]["arguments"]
    except Exception:
        return "unknown", {}


def detect_language(input_text):
    # 确保输入是字符串
    if isinstance(input_text, str):
        s = input_text.strip()
        if re.search(r"[üöäß]|[ÄÖÜ]", s) or re.search(
            r"\b(der|die|das|und|ein|nicht|ist|zu|in|den|mit|auf|für|sie|es|dem|von)\b", s, re.I
        ):
            return "ger"
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", s):
            return "jap"
        if re.search(r"[a-zA-Z]", s):
            ascii_ratio = sum(1 for c in s if ord(c) < 128) / max(1, len(s))
            if ascii_ratio > 0.7:
                return "en"
    return "zh"


def count_score_and_proportion(df):
    # 按分数降序排列
    score_counts = (
        df["score"]
        .value_counts(ascending=False)
        .to_frame()
        .reset_index()
        .rename(columns={"index": "score", "score": "count"})
    )
    print(score_counts)

    # 统计 >8分的占比
    above_8 = df[df["score"] > 8].shape[0]
    total = df.shape[0]
    proportion_above_8 = above_8 / total if total > 0 else 0
    print(f"得分大于8分的占比: {proportion_above_8:.2%}")

    # 3. 应用到所有数据，输出不合格条目
    df["argument_check_error"] = df.apply(check_intent_arguments, axis=1)
    errors = df[df["argument_check_error"] != ""]
    print("存在问题的数据条数:", errors.shape[0])
    print(errors[["query", "name", "arguments", "argument_check_error"]].head(10))


df = read_jsonl_to_df(
    "/data0/work/SusieSu/project/openllm_datas_and_temp_codes/function_query_1210/function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07/train.jsonl"
)

count_score_and_proportion(df)

df1 = df[df["score"] > 8]
print("score > 8 df1.shape", df1.shape)

df1[["name", "arguments"]] = df1["function_call"].apply(lambda x: pd.Series(get_intent_from_function_call(x)))
df1["language"] = df1["query"].apply(detect_language)
df1["source"] = "mcp_intent"

# 3. 应用到所有数据，输出不合格条目
df["argument_check_error"] = df.apply(check_intent_arguments, axis=1)
errors = df[df["argument_check_error"] != ""]
print("存在问题的数据条数:", errors.shape[0])
print(errors[["query", "name", "arguments", "argument_check_error"]].head(10))

df22 = df[df["argument_check_error"] == ""]
df22 = df22[["query", "function_call", "answer", "score", "name", "arguments", "language", "source"]]
df22.to_excel(
    "/data0/work/SusieSu/project/openllm_datas_and_temp_codes/function_query_1210/origin_intent_train_data_filtered/more_than_8_data_slim_less.xlsx"
)
