#!/usr/bin/env python

import json

import pandas as pd


def simple_vote_function(parsed_dicts, weights):
    """简化的投票函数，避免复杂的递归问题"""
    if not parsed_dicts:
        return {}

    result = {}

    # 获取所有键
    all_keys = set()
    for d in parsed_dicts:
        all_keys.update(d.keys())

    # 对每个键投票
    for key in all_keys:
        values = [d[key] for d in parsed_dicts if key in d]
        key_weights = [w for d, w in zip(parsed_dicts, weights, strict=False) if key in d]

        if isinstance(values[0], dict):
            # 嵌套字典：递归处理
            result[key] = simple_vote_function(values, key_weights)
        else:
            # 简单值：投票选择
            counter = {}
            for val, weight in zip(values, key_weights, strict=False):
                counter[val] = counter.get(val, 0) + weight
            result[key] = max(counter, key=counter.get)

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
    model_weights = {"claude-sonnet-4": 2, "gemini-2.5": 1, "gpt-4o": 2, "qwen3-4b-2507": 0.5}

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
    print("--------------------parsed_results--------------------", parsed_results)
    print("--------------------weights--------------------", weights)
    voted_result = simple_vote_function(parsed_results, weights)
    print("--------------------voted_result--------------------", voted_result)

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

    return pd.DataFrame(results)


# 测试数据
test_data = [
    {
        "model": "claude-sonnet-4",
        "function_call": '{"name": "music_settings_control", "arguments": {"auto_stop_time": 15}}',
    },
    {"model": "gemini-2.5", "function_call": '{"name": "music_settings", "arguments": {"auto_stop_time": 15}}'},
    {"model": "gpt-4o", "function_call": '{"name": "music_settings_control", "arguments": {"auto_stop_time": 15}}'},
    {
        "model": "qwen3-4b-2507",
        "function_call": '{"name": "music_settings_control", "arguments": {"auto_stop_time": 15}}',
    },
]

print("=== 测试相同结果的投票 ===")
result1 = vote_for_model_results(test_data)
print(f"\n最终投票结果: {result1}")

# sample_grouped_data = pd.DataFrame([
#     {
#         'query': '设置音乐自动停止',
#         'model_function_call': [
#             {'model': 'claude-sonnet-4',
#               'function_call':
# '{"name": "music_settings_control", "arguments": {"auto_stop_time": 15}}'},
#             {'model': 'gpt-4o',
# 'function_call': '{"name": "music_settings_control", "arguments": {"auto_stop_time": 15}}'},
#             {'model': 'gemini-2.5', 'function_call':
# '{"name": "music_settings_control", "arguments": {"auto_stop_time": 20}}'}
#         ]
#     },
#     {
#         'query': '播放视频',
#         'model_function_call': [
#             {
# 'model': 'claude-sonnet-4',
# 'function_call':
# '{"name": "video_play", "arguments": {"quality": "4K"}}'},
#             {'model': 'gpt-4o',
# 'function_call': '{"name": "video_play", "arguments": {"quality": "HD"}}'},
#             {'model': 'gemini-2.5',
# 'function_call': '{"name": "video_play", "arguments": {"quality": "4K"}}'}
#         ]
#     }
# ])

# 批量处理
# batch_results = batch_vote_for_queries(sample_grouped_data)

# print(f"\n最终批量处理结果:")
# for _, row in batch_results.iterrows():
#     print(f"\nQuery: {row['query']}")
#     print(f"投票结果: {row['voted_function_call']}")


df = pd.read_excel(
    "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/data/merge_data/query_and_function_call.xlsx"
)
df = df[["query", "model_function_call"]]

df["model_function_call"] = df["model_function_call"].apply(lambda x: eval(x))

df_result = batch_vote_for_queries(df)
