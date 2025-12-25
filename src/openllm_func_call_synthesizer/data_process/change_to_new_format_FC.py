import json
import sys

sys.path.append("/data0/work/SusieSu/project/openllm_func_call_synthesizer/src/openllm_func_call_synthesizer/utils")


# 只需要加载一次 openai_tools
with open("/data0/work/SusieSu/project/openllm_func_call_synthesizer/openai_tools.json") as f:
    OPENAI_TOOLS = json.load(f)

# def replace_none_with_empty_string(obj):
#     """Recursively replace all None values with empty strings in the given object."""
#     if isinstance(obj, dict):
#         new_dict = {}
#         for k, v in obj.items():
#             if k == "default" and v is None:
#                 new_dict[k] = ""
#             else:
#                 new_dict[k] = replace_none_with_empty_string(v)
#         return new_dict
#     elif isinstance(obj, list):
#         return [replace_none_with_empty_string(item) for item in obj]
#     else:
#         return obj

# # 不要转成字符串，保留为 list
# OPENAI_TOOLS = replace_none_with_empty_string(OPENAI_TOOLS_DATA)


def convert_to_new_format_v2(sample):
    messages = sample["messages"]

    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant.",
    }

    user_msg, assistant_msg = None, None
    for m in messages:
        if m["role"] == "user":
            user_msg = {"role": "user", "content": m["content"]}
        elif m["role"] == "assistant":
            # 仅当 tool_calls 存在且为非空list时保留，否则不加 tool_calls 字段
            if (
                "tool_calls" in m
                and m["tool_calls"] is not None
                and isinstance(m["tool_calls"], list)
                and len(m["tool_calls"]) > 0
            ):
                assistant_msg = {"role": "assistant", "tool_calls": m["tool_calls"]}
            else:
                # assistant 只有 content 普通回复
                assistant_msg = {"role": "assistant", "content": m.get("content", "")}

    # 注意：OPENAI_TOOLS一定是list类型，直接使用，不要转list(OPENAI_TOOLS)和不要转字符串
    result = {"messages": [system_msg, user_msg, assistant_msg], "tools": OPENAI_TOOLS}
    return result


# 数据格式转换：将原始data的格式转换成目标格式
def convert_to_new_format(sample):
    # messages 保持不变， tools 替换成 OPENAI_TOOLS
    messages = sample["messages"]
    result = {"messages": messages, "tools": OPENAI_TOOLS}
    return result


def check_data_for_tool_calls_none(data):
    errors = []
    for idx, row in enumerate(data):
        messages = row.get("messages", [])
        for m in messages:
            if m.get("role", "") == "assistant":
                if "tool_calls" in m:
                    tool_calls_value = m["tool_calls"]
                    if tool_calls_value is None:
                        errors.append(f"idx {idx}: assistant tool_calls is None")
                    elif isinstance(tool_calls_value, str) and tool_calls_value.strip().lower() in ["null", "none"]:
                        errors.append(f"idx {idx}: assistant tool_calls is the string '{tool_calls_value}'")
                    elif not isinstance(tool_calls_value, list):
                        errors.append(f"idx {idx}: assistant tool_calls is not list but {type(tool_calls_value)}")
    if errors:
        print("Data check found issues with 'tool_calls':")
        for e in errors:
            print(e)
        print(f"Total {len(errors)} error(s) found!")
    else:
        print("No NoneType or 'null'/'None' string tool_calls found in your data.")


def sanitize_messages(example):
    new_msgs = []
    for m in example["messages"]:
        m = dict(m)
        if "tool_calls" in m and m["tool_calls"] is None:
            print("删除 tool_calls = None ")
            del m["tool_calls"]

        if "tool_calls" in m and isinstance(m["tool_calls"], list) and len(m["tool_calls"]) == 0:
            print("删除空 tool_calls")
            del m["tool_calls"]

        if m.get("role") == "assistant":
            if "tool_calls" in m and "content" in m:
                print("删除 assistant 同时有 content + tool_calls")
                del m["content"]

        new_msgs.append(m)

    example["messages"] = new_msgs
    return example


if __name__ == "__main__":
    with open(
        "/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1212_functino_call/train_data_fc_1212/mcp_dev_fc.json"
    ) as f:
        data = json.load(f)
    print(len(data))
    data = data[0:100]

    print("检查输入数据")
    check_data_for_tool_calls_none(data)

    new_data = [convert_to_new_format_v2(row) for row in data]

    # new_data = [convert_to_new_format(row) for row in data]

    # # 检查new_data里面tools字段的类型，如果为字符串报错
    for i, d in enumerate(new_data):
        if isinstance(d.get("tools", None), str):
            print(f"[Error] new_data[{i}]['tools'] is str: {d['tools']}")
        elif not isinstance(d.get("tools", None), list):
            print(f"[Error] new_data[{i}]['tools'] not a list, but {type(d['tools'])}")

    # print("Validating new_data for None or invalid tool_calls ...")
    # print("检查输出数据")
    # check_data_for_tool_calls_none(new_data)

    # new_data2 =  [ sanitize_messages(aa) for aa in new_data ]

    with open(
        "/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1212_functino_call/train_data_fc_1223/mcp_dev_fc.json",
        "w",
    ) as f:
        # 保证tools字段是list并且不是字符串
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print("--len(new_data)--", len(new_data))
