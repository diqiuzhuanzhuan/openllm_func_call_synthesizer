import json


def select_functions_by_names(json_path, function_names):
    """
    从指定的 function_docs111.json 文件中选出名字在 function_names 列表内的 function。
    返回一个tools列表。
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    tools = data.get("tools", [])
    selected_tools = [tool for tool in tools if tool.get("name") in function_names]
    return selected_tools


if __name__ == "__main__":
    # 例子: 指定要选择的函数名字
    target_names = [
        "create_album",
        "search_photos",
        # 可以添加更多名字
    ]
    path = "../examples/function_docs_all.json"  # 改成你的json文件路径
    results = select_functions_by_names(path, target_names)
    rs = json.dumps({"tools": results}, ensure_ascii=False, indent=2)
    print(rs)
