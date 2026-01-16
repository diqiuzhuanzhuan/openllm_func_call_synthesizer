import json
import os
import sys
from random import shuffle

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("/data/work/CHenXuFei/openllm_func_call_synthesizer/src/openllm_func_call_synthesizer/utils")

from data_process_utils import (
    concat_df,
    concat_excel_file,
    count_score_and_proportion,
    detect_language,
    read_excel,
    read_jsonl,
    value_counts,
    value_counts_by_language,
)
from utils import check_intent_arguments


def get_intent_from_function_call(function_call):
    try:
        tmp = json.loads(function_call)
        return tmp[0].get("name", "unknown"), tmp[0].get("arguments", {})
    except Exception:
        return "unknown", {}


def process_function_call_data(root, source):
    df = pd.read_csv(os.path.join(root, "output.csv")).fillna("[]")
    print("input data shape, columns", df.shape, df.columns)

    # 1.预处理， 统计分布， 提取name 和 aruguments
    count_score_and_proportion(df)

    df = df[df["score"] > 8]
    print("score > 8 df.shape", df.shape)

    df[["name", "arguments"]] = df["function_call"].apply(lambda x: pd.Series(get_intent_from_function_call(x)))
    if "language" not in df.columns:
        df["language"] = df["query"].apply(detect_language)

    df["source"] = source
    print("-----------大于8分的name分布-------------------")
    value_counts(df, "name")

    print("df.shape", df.shape, df.columns)
    print("-----------原始文件  >8分的 分布-------------------")
    value_counts_by_language(df, "name")

    # 2.数据必须字段检查
    df["argument_check_error"] = df.apply(check_intent_arguments, axis=1)
    errors = df[df["argument_check_error"] != ""]
    print("存在问题的数据条数:", errors.shape[0], errors.columns)
    # print(errors[["query", "name", "arguments", "argument_check_error"]].head(10))

    df2 = df[df["argument_check_error"] == ""]
    df2 = df2[
        ["query", "function_call", "answer", "score", "name", "arguments", "language", "source", "argument_check_error"]
    ]
    df2.to_excel(os.path.join(root, "train_processed.xlsx"))
    # 只保存必要字段， 否则文件太大
    errors.to_excel(os.path.join(root, "train_errors.xlsx"))
    # 可以只存关键字段  [['query', 'dimension', 'language',  'provider', 'model',
    #    'function_call', 'answer', 'score', 'name', 'arguments',
    #    'source', 'argument_check_error']]
    print("-----------去除问题数据后   >8分的 分布-------------------")
    value_counts_by_language(df2, "name")

    print("完成数据处理，处理前数据条数:", df.shape[0], "处理后数据条数:", df2.shape[0])

    return df2


def split_fc_data(input_file, output_root):
    fc_data = read_jsonl(input_file)

    # 加载后先shuffle一次
    shuffle(fc_data)
    # 先切分出train和剩余部分（train: 0.8, 其余: 0.2）
    fc_train, fc_rest = train_test_split(fc_data, test_size=0.2, random_state=42, shuffle=True)
    # 再从剩余部分均分dev与test（各0.1/总体）
    fc_dev, fc_test = train_test_split(fc_rest, test_size=0.5, random_state=42, shuffle=True)

    # 保存切分结果
    with open(os.path.join(output_root, "mcp_train_fc.json"), "w", encoding="utf-8") as f_train:
        json.dump(fc_train, f_train, ensure_ascii=False, indent=2)
    with open(os.path.join(output_root, "mcp_dev_fc.json"), "w", encoding="utf-8") as f_dev:
        json.dump(fc_dev, f_dev, ensure_ascii=False, indent=2)
    with open(os.path.join(output_root, "mcp_test_fc.json"), "w", encoding="utf-8") as f_test:
        json.dump(fc_test, f_test, ensure_ascii=False, indent=2)
    print(f"已切分 train:{len(fc_train)}, dev:{len(fc_dev)}, test:{len(fc_test)}")


if __name__ == "__main__":
    big_root = "/data0/work/SusieSu/project/openllm_datas_and_temp_codes/function_query_1210/Function_Call_2qi_LoongMa/"
    # 路径太长可以定义短变量名，然后用os.path.join组装，或者将根目录抽取出来重用
    DATA_ROOT = "/data/work/CHenXuFei/data/function_call_data"
    FC_DIR = "function_query_1210/Function_Call_2qi_LoongMa"
    root1 = os.path.join(DATA_ROOT, FC_DIR, "function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07")
    root2 = os.path.join(DATA_ROOT, FC_DIR, "susie_function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07")

    data_process = False
    fc_data_merge = False
    concat_mcp_fc = False
    convert_to_jsonl = True
    split_fc_train_dev_test = True  # 这个之前要先转换好格式

    if data_process:
        df = process_function_call_data(root1, "function_call_loongma")
        df2 = process_function_call_data(root2, "function_call_loongma_susie")

    if fc_data_merge:
        # 数据合并
        df_concat = concat_df(df, df2)
        df_concat.to_excel(os.path.join(big_root, "function_call_raw_data_1212.xlsx"))
        # 数据分布统计
        value_counts_by_language(df_concat, "name")
        value_counts(df_concat, "name")
        value_counts(df_concat, "source")
        value_counts(df_concat, "language")

    if concat_mcp_fc:
        # 合并fc数据和原来intent_mcp数据
        mcp_data_root = "function_query_1210/Origin_MCP_Intent_train_data_filtered"
        fc_data_root = "function_query_1210/Function_Call_2qi_LoongMa"

        mcp_intent_file = os.path.join(
            DATA_ROOT, "function_query_1210/Origin_MCP_Intent_train_data_filtered/mcp_intent_raw_data_1212.xlsx"
        )
        fc_file = os.path.join(
            DATA_ROOT, "function_query_1210/Function_Call_2qi_LoongMa/function_call_raw_data_1212.xlsx"
        )

        df_all = concat_excel_file(mcp_intent_file, fc_file)

        df_all.to_excel(
            "/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1212_functino_call/mcp_fc_data_all_1212.xlsx"
        )

        value_counts_by_language(df_all, "name")
        value_counts(df_all, "name")
        value_counts(df_all, "source")
        value_counts(df_all, "language")

    if convert_to_jsonl:
        # 合并后的数据 转成train.jsonl 格式
        # 读取已经格式化好的数据（包含lora_input列）
        df = read_excel(
            "/data/work/CHenXuFei/data/function_call_data/train_data_fc_0114_fixed/raw_function_call_data_0114_processed_with_messages.xlsx"
            
        )
        df = df.where(pd.notnull(df), None)
        
        # 检查是否有lora_input列（由generate_function_call_format_data_new.py生成）
        if "lora_input" not in df.columns:
            print("警告: 没有找到lora_input列，请先运行generate_function_call_format_data_new.py生成格式化数据")
            sys.exit(1)

        import ast
        # 把df写到jsonl文件里，使用lora_input列
        count = 0
        with open(
            os.path.join(DATA_ROOT, "train_data_fc_0114_fixed/train.jsonl"),
            "w",
            encoding="utf-8",
        ) as fout:
            for _, row in df.iterrows():
                lora_input = row.get("lora_input")
                if lora_input and pd.notnull(lora_input):
                    # 如果是字符串需要解析（用ast.literal_eval因为是Python dict格式）
                    if isinstance(lora_input, str):
                        try:
                            lora_input = ast.literal_eval(lora_input)
                        except Exception as e:
                            print(f"解析失败: {e}")
                            continue
                    # 写入已格式化的数据
                    fout.write(json.dumps(lora_input, ensure_ascii=False) + "\n")
                    count += 1
        print(f"已生成jsonl文件，共 {count} 条数据")
        
    if split_fc_train_dev_test:
        critic_file = "train_data_fc_0114_fixed"

        input_file = os.path.join(DATA_ROOT, critic_file, "train.jsonl")

        output_root = os.path.join(DATA_ROOT, "train_data_fc_0114_fixed")
        split_fc_data(input_file, output_root)
