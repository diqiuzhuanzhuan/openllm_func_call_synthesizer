# function call数据，test data是json格式，
# 用于openllm测试，需要jsonl格式 只保留query
# 用StandartTest测评， 需要excel格式

import json

import pandas as pd


def change_json_to_jsonl(json_file_path, output_jsonl_path):
    # 读取JSON文件

    with open(json_file_path, encoding="utf-8") as f:
        data = json.load(f)

    # 将每条数据中 role 为 user 的 content 提取出来，并写入jsonl文件

    with open(output_jsonl_path, "w", encoding="utf-8") as fout:
        for item in data:
            # item 应该有一个 'messages' 字段，是 list
            user_content = ""
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            out_line = {"query": user_content}
            fout.write(json.dumps(out_line, ensure_ascii=False) + "\n")

    # 可选：预览前3个样本
    with open(output_jsonl_path, encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            print(line.strip())
            if i >= 2:
                break


def change_json_to_excel(json_file_path, output_excel_path):
    # 读取JSON文件
    with open(json_file_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        # 获取 user 的 query
        query = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        # 获取 assistant 的 tool_calls
        assistant_msg = None
        for msg in item.get("messages", []):
            if msg.get("role") == "assistant":
                assistant_msg = msg
                break

        function_col = ""
        if assistant_msg and assistant_msg.get("tool_calls"):
            tc = assistant_msg["tool_calls"]
            # 有可能是 list
            if isinstance(tc, list) and len(tc) > 0:
                # 只取第一个，如果有多个，你可以for循环
                tool = tc[0]
                func_info = tool.get("function", {})
                arguments = func_info.get("arguments", {})
                name = func_info.get("name", "")
                # arguments 可能是json字符串
                # 尝试解析
                if isinstance(arguments, str):
                    try:
                        arguments_dict = json.loads(arguments)
                    except Exception:
                        arguments_dict = arguments
                else:
                    arguments_dict = arguments
                # function列为整个function字段（第一个tool_calls的function），转成json字符串保存
                try:
                    function_col = json.dumps(func_info, ensure_ascii=False)
                except Exception:
                    function_col = str(func_info)
                row = {"query": query, "arguments": arguments_dict, "name": name, "function": function_col}
            else:
                row = {"query": query, "arguments": {"unknown": {}}, "name": "", "function": ""}
        else:
            row = {"query": query, "arguments": {"unknown": {}}, "name": "", "function": ""}
        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_excel(output_excel_path, index=False)
    print(f"写入完毕，输出到 {output_excel_path}")

def change_excel_to_jsonl(excel_file_path, output_jsonl_path):
    # 读取excel文件
    df = pd.read_excel(excel_file_path)
    df = df[df['source'] == 'XufeiChen']
    print('----------df.shape---------', df.shape)
    with open(output_jsonl_path, "w", encoding="utf-8") as fout:
        for user_content in df["input"]:
            # 跳过空值
            if pd.isna(user_content):
                continue
            line = json.dumps({"query": user_content}, ensure_ascii=False)
            fout.write(line + "\n")
    print(f"jsonl已写入 {output_jsonl_path}")

if __name__ == "__main__":
    root = "/data/work/CHenXuFei/data/function_call_data/train_data_fc_0114_fixed/"
    json_file_path = root + "mcp_test_fc.json"
    output_jsonl_path = root + "mcp_test_fc.jsonl"
    output_excel_path = root + "mcp_test_fc.xlsx"
    change_json_to_jsonl(json_file_path, output_jsonl_path)

    #change_json_to_excel(json_file_path, output_excel_path)
