#!/usr/bin/env python

import json

import pandas as pd


def change_decode(ll):
    # 解码并重新编码为正确格式
    for item in ll:
        function_call_data = json.loads(item["function_call"])
        item["function_call"] = json.dumps(function_call_data, ensure_ascii=False)
    return ll


def simple_vote_function(parsed_dicts, weights):
    """简化的投票函数，避免复杂的递归问题，空字典也参与投票"""
    if not parsed_dicts:
        return {}

    result = {}

    # 获取所有键，包括空字典的情况
    all_keys = set()
    # print('----------parsed_dicts-----', parsed_dicts)
    for d in parsed_dicts:
        # print('----------d-----', d)
        if d:  # 如果不是空字典
            all_keys.update(d.keys())

    # 如果所有字典都是空的，或者没有共同键，需要特殊处理
    if not all_keys:
        # 所有字典都是空的情况，投票选择空字典
        empty_weight = sum(weights[i] for i, d in enumerate(parsed_dicts) if not d)
        non_empty_weight = sum(weights[i] for i, d in enumerate(parsed_dicts) if d)
        return {} if empty_weight >= non_empty_weight else parsed_dicts[0]

    # 对每个键投票
    for key in all_keys:
        # 收集所有有该键的值和对应权重
        values_with_weights = []
        # 收集没有该键的情况（视为None）和对应权重
        none_weight = 0

        for i, d in enumerate(parsed_dicts):
            if key in d:
                values_with_weights.append((d[key], weights[i]))
            else:
                # 空字典或没有该键的情况
                none_weight += weights[i]

        if isinstance(values_with_weights[0][0], dict) if values_with_weights else False:
            # 嵌套字典：递归处理
            nested_values = [item[0] for item in values_with_weights]
            nested_weights = [item[1] for item in values_with_weights]
            result[key] = simple_vote_function(nested_values, nested_weights)
        else:
            # 简单值：投票选择
            counter = {}
            for val, weight in values_with_weights:
                counter[val] = counter.get(val, 0) + weight

            # 如果有空字典的权重，也要考虑进去（表示该键不存在）
            if none_weight > 0:
                counter[None] = none_weight

            # 选择权重最高的值
            max_val = max(counter, key=counter.get)
            if max_val is not None:
                result[key] = max_val
            # 如果None权重最高，则该键不包含在结果中

    return result


def vote_for_model_results(model_results: list[dict]) -> dict:
    """
    对多个模型的结果进行加权投票
    参数:
        model_results: 包含model和function_call的字典列表
                      格式: [{'model': 'model_name', 'function_call': 'json_string'}, ...]
    返回:
        投票后的最终function_call字典
    """
    # 模型权重映射
    model_weights = {
        "claude-sonnet-4": 1.5,
        "gemini-2.5": 1,
        "gpt-4o": 2,
        # 'qwen3-4b-2507': 0.5
    }

    print(f"输入的模型结果数量: {len(model_results)}")

    # 解析function_call JSON字符串并收集权重
    parsed_results = []
    weights = []

    for item in model_results:
        model_name = item["model"]
        function_call_str = item["function_call"]

        print(f"处理模型: {model_name}")

        try:
            # 解析JSON字符串
            function_call_dict = json.loads(function_call_str)
            parsed_results.append(function_call_dict)

            # 获取对应权重
            weight = model_weights.get(model_name, 1.0)  # 默认权重1.0
            weights.append(weight)

            print(f"  权重: {weight}")
            print(f"  Function Call: {function_call_dict}")

        except json.JSONDecodeError as e:
            print(f"  跳过无效的JSON: {function_call_str}")
            print(f"  错误: {e}")
            continue

    if not parsed_results:
        print("没有有效的结果可以投票")
        return {}

    print(f"\n有效结果数量: {len(parsed_results)}")
    print(f"对应权重: {weights}")

    # 使用现有的投票函数
    print("-------------调试信息：检查vote_for_nested_dict函数的输入-------------")
    print(parsed_results, type(parsed_results))
    # voted_result = vote_for_nested_dict(parsed_results, weights)
    # print('--------------------parsed_results--------------------', parsed_results)
    # print('--------------------weights--------------------', weights)
    voted_result = simple_vote_function(parsed_results, weights)
    # print('--------------------voted_result--------------------', voted_result)

    return voted_result


def batch_vote_for_queries(df_grouped_data):
    """
    批量处理多个query的投票

    参数:
        df_grouped_data: DataFrame，包含query和对应的model_function_call列表

    返回:
        DataFrame，包含每个query的投票结果
    """
    results = []

    for _, row in df_grouped_data.iterrows():
        try:
            query = row["query"]
            model_function_calls = row["model_function_call"]

            # print(f"\n{'='*50}")
            # print(f"处理Query: {query}")
            # print(f"模型数量: {len(model_function_calls)}")

            # 对当前query进行投票
            voted_result = vote_for_model_results(model_function_calls)

            results.append(
                {"query": query, "model_function_calls": model_function_calls, "voted_function_call": voted_result}
            )
        except Exception:
            print("--------------------error--------------------", row)
            results.append({"query": query, "model_function_calls": model_function_calls, "voted_function_call": ""})

    return pd.DataFrame(results)


test_data = [
    {"model": "claude-sonnet-4", "function_call": "{}"},
    {"model": "gemini-2.5", "function_call": '{"name": "music_play_control", "arguments": {"play_mode": "normal"}}'},
    {"model": "gpt-4o", "function_call": '{"name": "music_play_control", "arguments": {}}'},
    {
        "model": "qwen3-4b-2507",
        "function_call": '{"name": "music_play_control", \
            "arguments": {"title": "Happy Birthday", "type": "song", "source": "recent", "play_mode": "normal"}}',
    },
]

print("=== 测试相同结果的投票 ===")
result1 = vote_for_model_results(test_data)
print(f"\n最终投票结果: {result1}")


df = pd.read_excel(
    "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/data/function_call_0919/query_and_function_call.xlsx"
)
df = df[["query", "model_function_call"]]

df["model_function_call"] = df["model_function_call"].apply(lambda x: eval(x))
df["model_function_call"] = df["model_function_call"].apply(lambda x: change_decode(x))

df_result = batch_vote_for_queries(df)

df_raw = pd.read_excel("/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/data/merge_data/merge_data.xlsx")
print(df_raw.columns, df_raw.shape)
print(df_result.columns, df_result.shape)

df_result = df_result.drop_duplicates(subset=["query"])
print(df_result.columns, df_result.shape)
df_raw = df_raw.drop_duplicates(subset=["query"])
print(df_raw.columns, df_raw.shape)
df_raw = df_raw[["query", "dimension", "language", "function_call_model", "model_function_call"]]
df_merge = pd.merge(df_result, df_raw, on=["query"], how="left")
print(df_merge.columns, df_merge.shape)

df_merge = df_merge[
    [
        "query",
        "model_function_calls",
        "voted_function_call",
        "language",
        "dimension",
        "function_call_model",
        "model_function_call",
    ]
]
df_merge.to_excel(
    "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/data/function_call_0919/voted_function_call_rs_0919_3models.xlsx"
)
