import json
import re

import pandas as pd


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


def merge_excel(path1, path2, on_col, how="left"):
    df1 = pd.read_excel(path1)
    print("df1.shape, df1.columns", df1.shape, df1.columns)
    df2 = pd.read_excel(path2)
    print("df2.shape, df2.columns", df2.shape, df2.columns)
    df = pd.merge(df1, df2, on=on_col, how=how)
    print("df.shape, df.columns", df.shape, df.columns)
    return df


def concat_excel_file(path1, path2):
    df1 = pd.read_excel(path1)
    print("df1.shape, df1.columns", df1.shape, df1.columns)
    df2 = pd.read_excel(path2)
    print("df2.shape, df2.columns", df2.shape, df2.columns)
    df = pd.concat([df1, df2], ignore_index=True)
    print("df.shape, df.columns", df.shape, df.columns)
    return df


def concat_df(df1, df2):
    print("df1.shape, df1.columns", df1.shape, df1.columns)
    print("df2.shape, df2.columns", df2.shape, df2.columns)
    concat_df = pd.concat([df1, df2], ignore_index=True)
    print("concat_df.shape, concat_df.columns", concat_df.shape, concat_df.columns)
    return concat_df


def read_excel(path):
    df = pd.read_excel(path)
    print("df.shape, df.columns", df.shape, df.columns)
    return df


def dedup(df, subset_cols):
    # 标记重复项，而不是直接去重
    df["is_duplicate"] = df.duplicated(subset=subset_cols, keep=False)
    df_de = df.drop_duplicates(subset=subset_cols)
    return df_de, df


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


def value_counts(df, col_name):
    df_vc = (
        df[col_name]
        .value_counts(ascending=False)
        .to_frame()
        .reset_index()
        .rename(columns={"index": col_name, col_name: col_name})
    )
    tmp = list(set(df_vc[col_name].to_list()))
    print("---------------------------------------------")
    print(df_vc)
    print("------------value_counts unique values-------------------", col_name, len(tmp), tmp)
    print("------------value_counts shape, columns-------------------", df_vc.shape, df_vc.columns)
    return df_vc


def value_counts_by_language(df, col_name):
    # 按语种统计
    df_lan = df.groupby("language")[col_name].value_counts().to_frame("count")
    print(df_lan.to_string())

    return df_lan
