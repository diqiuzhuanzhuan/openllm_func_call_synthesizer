import argparse
import ast
import json
import time

import pandas as pd
import yaml
from llm_api_caller import batch_llm_predict_api

# 导入本地模型和API调用模块
from llm_local_caller import batch_llm_predict


def load_config(config_file):
    """
    加载YAML配置文件
    如果配置了prompt_file和prompt_key，会从JSON文件读取system_prompt
    支持 ${root} 等变量替换
    """
    import os

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 第一步：处理变量替换（如 ${root}）
    def replace_variables(config_dict):
        """递归替换配置中的变量"""
        if not isinstance(config_dict, dict):
            return config_dict

        # 先收集所有可用的变量（root等）
        variables = {}
        for key, value in config_dict.items():
            if isinstance(value, str) and not value.startswith("${"):
                variables[key] = value

        # 递归替换
        for key, value in config_dict.items():
            if isinstance(value, str):
                # 替换形如 ${variable_name} 的占位符
                for var_name, var_value in variables.items():
                    placeholder = f"${{{var_name}}}"
                    if placeholder in value:
                        value = value.replace(placeholder, var_value)
                config_dict[key] = value
            elif isinstance(value, dict):
                config_dict[key] = replace_variables(value)

        return config_dict

    config = replace_variables(config)

    # 读取 prompt 文件
    # 方式1: 根据 prompt_key 选择对应的 txt 文件（推荐）
    if "prompt_key" in config:
        prompt_key = config["prompt_key"]
        prompt_file_key = f"system_prompt_{prompt_key}_file"
        print("---prompt_file_key---", prompt_file_key)

        if prompt_file_key in config:
            prompt_file_path = config[prompt_file_key]
            try:
                with open(prompt_file_path, encoding="utf-8") as f:
                    config["system_prompt"] = f.read().strip()
                print(f"✅ 从 {prompt_file_path} 读取 {prompt_key} prompt")
            except FileNotFoundError:
                print(f"⚠️  警告: 未找到文件 {prompt_file_path}")
            except Exception as e:
                print(f"⚠️  警告: 读取文件出错 {e}")
        else:
            print(f"⚠️  警告: 配置中未找到 {prompt_file_key} 字段")

    # 方式2: 从 JSON 文件读取（保留兼容性）
    if "json_prompt_file" in config and "prompt_key" in config:
        # 获取配置文件所在目录
        config_dir = os.path.dirname(os.path.abspath(config_file))
        prompt_file_path = config["prompt_file"]

        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(prompt_file_path):
            prompt_file_path = os.path.join(config_dir, prompt_file_path)

        # 读取JSON文件
        try:
            with open(prompt_file_path, encoding="utf-8") as f:
                prompts = json.load(f)

            prompt_key = config["prompt_key"]
            if prompt_key in prompts:
                config["system_prompt"] = prompts[prompt_key]
                print(f"✅ 从 {prompt_file_path} 读取 prompt: {prompt_key}")
            else:
                print(f"⚠️  警告: JSON文件中未找到key '{prompt_key}'，使用配置文件中的默认prompt")
        except FileNotFoundError:
            print(f"⚠️  警告: 未找到文件 {prompt_file_path}，使用配置文件中的默认prompt")
        except json.JSONDecodeError as e:
            print(f"⚠️  警告: JSON文件格式错误 {e}，使用配置文件中的默认prompt")

    return config


# 评测相关代码
def evaluate_arguments(ground_truth, model_response):
    """
    评测函数参数的准确性
    Returns:
        str: 评测结果 ('exact_match', 'has_extra_fields', 'partial_match', 'no_match', 'subset')
    """
    if not isinstance(ground_truth, dict) or not isinstance(model_response, dict):
        return "no_match"
    if ground_truth == {} and model_response == {}:
        return "exact_match"
    if not ground_truth or not model_response:
        return "no_match"
    # 1. 完全一致
    if ground_truth == model_response:
        return "exact_match"

    # 2. 检查是否有多余字段
    gt_keys = set(ground_truth.keys())
    mr_keys = set(model_response.keys())
    has_extra_fields = len(mr_keys - gt_keys) > 0
    if has_extra_fields:
        return "has_extra_fields"

    ##新增子集关系
    # 3. 子集关系
    def is_subset(a, b):
        """判断 a 是否为 b 的子集（递归支持 dict、str）"""
        if isinstance(a, dict) and isinstance(b, dict):
            for k, v in a.items():
                if k not in b:
                    return False
                if not is_subset(v, b[k]):
                    return False
            return True
        elif isinstance(a, str) and isinstance(b, str):
            # 忽略括号、空格、引号差异后再判断
            def normalize(s):
                for ch in ["「", "」", "（", "）", "(", ")", '"', "'", " "]:
                    s = s.replace(ch, "")
                return s

            a_norm, b_norm = normalize(a), normalize(b)
            return a_norm in b_norm or b_norm in a_norm
        else:
            return a == b

    if is_subset(ground_truth, model_response) or is_subset(model_response, ground_truth):
        return "subset"

    # 4. 部分匹配
    common_keys = gt_keys & mr_keys
    if common_keys:
        for key in common_keys:
            if ground_truth[key] == model_response[key]:
                return "partial_match"
    return "no_match"


def evaluate_module(config, data=None):
    if data is None:
        data = pd.read_excel(config["evaluate_input_file"])
    # 字段名
    ground_truth_intent = config["ground_truth_intent"]
    ground_truth_slot = config["ground_truth_slot"]
    llm_intent = config["llm_intent"]
    llm_slot = config["llm_slot"]
    # 保证字段转dict
    data[ground_truth_slot] = data[ground_truth_slot].apply(lambda x: eval(x) if isinstance(x, str) else x)
    data[llm_slot] = data[llm_slot].apply(lambda x: eval(x) if isinstance(x, str) else x)
    # 函数/槽分开比对
    data["function_same"] = data.apply(lambda x: True if x[ground_truth_intent] == x[llm_intent] else False, axis=1)
    data["argument_same"] = data.apply(lambda x: True if x[ground_truth_slot] == x[llm_slot] else False, axis=1)
    print("函数相等分布:", data["function_same"].value_counts())
    print("参数相等分布:", data["argument_same"].value_counts())

    # 漂亮打印
    from rich import print as rprint

    summary_info = []
    for col in ["function_same", "argument_same"]:
        true_count = data[col].sum()
        false_count = (~data[col]).sum()
        total = true_count + false_count
        accuracy = true_count / total if total > 0 else 0

        summary = {
            "指标名称": col,
            "总数": total,
            "正确(True)": int(true_count),
            "错误(False)": int(false_count),
            "准确率": f"{accuracy:.4%}",
        }
        summary_info.append(summary)

    # 漂亮打印
    rprint("[bold cyan]=== 评测结果简表 ===[/bold cyan]")
    for info in summary_info:
        rprint(
            f"[bold yellow]{info['指标名称']}[/bold yellow] | "
            f"总数: [bold magenta]{info['总数']}[/bold magenta] | "
            f"正确: [bold green]{info['正确(True)']}[/bold green] | "
            f"错误: [bold red]{info['错误(False)']}[/bold red] | "
            f"准确率: [bold blue]{info['准确率']}[/bold blue]"
        )
    print("\n详细数值如下：")
    for info in summary_info:
        print(
            f"{info['指标名称']}: 正确={info['正确(True)']}, 错误={info['错误(False)']}, "
            f"准确率={info['准确率']} ({info['正确(True)']}/{info['总数']})"
        )

    # 参数细粒度评测
    data["argument_evaluation_result"] = data.apply(
        lambda x: evaluate_arguments(x[ground_truth_slot], x[llm_slot]), axis=1
    )
    # 查看评测结果分布
    rprint("[bold cyan]=== argument评测结果分布 ===[/bold cyan]")
    value_counts = data["argument_evaluation_result"].value_counts()
    print(value_counts)

    # 统计相关数量
    total = len(data)
    exact_match_count = (data["argument_evaluation_result"] == "exact_match").sum()
    partial_match_count = (data["argument_evaluation_result"] == "partial_match").sum()
    subset_match_count = (data["argument_evaluation_result"] == "subset").sum()

    exact_plus_partial = exact_match_count + partial_match_count
    combined_count = exact_match_count + partial_match_count + subset_match_count

    accuracy = exact_plus_partial / total if total > 0 else 0

    # 漂亮打印 argument详情
    from rich import print as rprint

    rprint("\n[bold cyan]========== argument具体计算过程如下 ==========[/bold cyan]")
    rprint(
        f"[bold yellow]数据总数 total[/bold yellow] = \
        [bold magenta]{total}[/bold magenta]"
    )
    rprint(
        f"[bold yellow]完全一致 exact_match 数量[/bold yellow]\
        = [bold green]{exact_match_count}[/bold green]"
    )
    rprint(
        f"[bold yellow]部分匹配 partial_match 数量[/bold yellow] = \
        [bold blue]{partial_match_count}[/bold blue]"
    )
    rprint(
        f"[bold yellow]子集关系 subset 数量[/bold yellow] = \
        [bold cyan]{subset_match_count}[/bold cyan]"
    )

    rprint(
        f"[bold]([green]exact_match[/green] + [blue]partial_match[/blue]) / total = "
        f"[green]{exact_match_count + partial_match_count}[/green] / [magenta]{total}[/magenta] \
            = [bold red]{accuracy:.4%}[/bold red][/bold]"
    )

    combined_accuracy = combined_count / total if total > 0 else 0

    rprint(
        f"[bold]([green]exact_match[/green] + \
            [blue]partial_match[/blue] + [cyan]subset[/cyan]) / total = "
        f"[green]{combined_count}[/green] / [magenta]{total}[/magenta] \
            [bold red]{combined_accuracy:.4%}[/bold red][/bold]"
    )

    # 每种情况分别展示两条例子
    print("\n各评测结果类型示例:")
    eval_types = ["exact_match", "partial_match", "subset", "has_extra_fields", "no_match"]
    for eval_type in eval_types:
        subset = data[data["argument_evaluation_result"] == eval_type]
        if len(subset) == 0:
            continue
        print(f"--- 评测结果类型: {eval_type} ---")
        for idx, row in subset.head(2).iterrows():
            print(f"行 {idx}:")
            print(f"  Ground Truth: {row[config['ground_truth_slot']]}")
            print(f"  Model Response: {row[config['llm_slot']]}")
            print(f"  评测结果: {row['argument_evaluation_result']}")
            print()

    data.to_excel(config["evaluate_output_file"])
    print("评测结果保存于:", config["evaluate_output_file"])
    return data


def evaluate_output_str_module(config, data=None):
    """
    针对模型输出为纯字符串的情况进行评测：
    直接比较 evaluate_output_str.ground_truth_col 与 evaluate_output_str.llm_intent_col 是否相等。
    若未传入 data，则从 evaluate_input_file 读取。
    """
    if data is None:
        data = pd.read_excel(config["evaluate_input_file"])

    eval_cfg = config.get("evaluate_output_str", {})
    gt_col = eval_cfg.get("ground_truth_col") or eval_cfg.get("ground_truth")
    llm_col = eval_cfg.get("llm_intent_col") or eval_cfg.get("llm_intent")
    if not gt_col or not llm_col:
        raise ValueError("evaluate_output_str 配置不完整，需要 ground_truth_col 与 llm_intent_col")

    def normalize_to_str(v):
        if pd.isna(v):
            return ""
        try:
            return str(v).strip()
        except Exception:
            return ""

    data["string_same"] = data.apply(
        lambda x: normalize_to_str(x.get(gt_col)) == normalize_to_str(x.get(llm_col)), axis=1
    )
    total = len(data)
    true_count = int(data["string_same"].sum()) if total > 0 else 0
    accuracy = true_count / total if total > 0 else 0.0

    print("\n=== 纯字符串输出评测（严格相等） ===")
    print(f"对比列: GT={gt_col} vs LLM={llm_col}")
    print(f"总数: {total}, 正确: {true_count}, 准确率: {accuracy:.4%}")

    out_path = config["evaluate_output_file"]
    data.to_excel(out_path, index=False)
    print("字符串评测结果保存于:", out_path)
    return data


def fix_inner_single_quotes(s):
    """
    如果字符串本身是被双引号包裹的(err: "修正后依旧失败...")，说明里面的内容是一个带单引号的str，
    此时我们只需要对内部内容进行处理，而不要动外层的引号。
    """
    # 先尝试外层是否是双引号，并且内容形如 "{'intent': ...}"
    if s.startswith('"') and s.endswith('"'):
        s_inner = s[1:-1]
        # 内部可能还有 \n 或 \"，先还原
        s_inner = s_inner.replace('\\"', '"').replace("\\\\", "\\")
        try:
            # 先直接 ast.literal_eval 尝试
            return ast.literal_eval(s_inner)
        except Exception:
            # 不行再粗暴地把内部的单引号替换成双引号，然后JSON解析
            s_fixed = s_inner.replace("'", '"')
            try:
                return json.loads(s_fixed)
            except Exception:
                print(f"fix_inner_single_quotes也失败: {s}")
                return None
    return None


def robust_parse_v2(x):
    """
    新的解析方法, 优先用ast解析，如果失败并且字符串是用外层双引号包起来的且内容为Python字典，
    使用fix_inner_single_quotes进一步解析。最后再尝试替换单引号再json解析一遍。
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        # 先尝试ast
        try:
            return ast.literal_eval(x)
        except Exception:
            print(f"ast解析失败: {x}\n")
            print("模型原始输出", x)
            print("模型原始输出类型", type(x))
            print("错误: {e1}")
            res = fix_inner_single_quotes(x)
            if res is not None:
                return res
            # 最后兜底: 尝试把所有单引号变为双引号再json.loads，
            # 注意会有问题, 仅用于最后兜底防崩
            try:
                x_json_like = x.replace("'", '"')
                return json.loads(x_json_like)
            except Exception:
                print(f"所有方式均失败: {x}")
                return {}
    return x


def postprocess_data(config):
    """
    数据后处理：提取LLM响应和Ground Truth到标准字段名
    合并了原来的ensure_llm_output_columns和ensure_gt_output_columns功能
    """
    input_file = config["postprocess_input_file"]
    output_file = config["postprocess_output_file"]

    df = pd.read_excel(input_file)

    # 处理LLM输出列
    llm_response_col = "llm_response"
    llm_intent_col = config["llm_intent"]
    llm_slot_col = config["llm_slot"]

    # 保证llm_response为dict
    df[llm_response_col] = df[llm_response_col].apply(robust_parse_v2)
    df[llm_intent_col] = df[llm_response_col].apply(lambda x: x.get("intent", "") if isinstance(x, dict) else "")
    df[llm_slot_col] = df[llm_response_col].apply(lambda x: x.get("slots", {}) if isinstance(x, dict) else {})

    # 处理Ground Truth列
    gt_col = config["ground_truth"]
    gt_intent_col = config["ground_truth_intent"]
    gt_slot_col = config["ground_truth_slot"]

    # 保证为dict
    df[gt_col] = df[gt_col].apply(robust_parse_v2)
    df[gt_intent_col] = df[gt_col].apply(lambda x: x.get("intent", "") if isinstance(x, dict) else "")
    df[gt_slot_col] = df[gt_col].apply(lambda x: x.get("slots", {}) if isinstance(x, dict) else {})

    df.to_excel(output_file, index=False)
    print(f"postprocess_data save to: {output_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Function Call Synthesizer with Configurable Evaluation")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    args = parser.parse_args()
    config = load_config(args.config)

    # 根据配置文件中的steps配置自动执行相应步骤
    steps_config = config.get("steps", {})

    print("======== 开始执行流水线 ========", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("执行步骤配置:", steps_config)

    df = None  # 用于在步骤间传递数据

    # 步骤1: LLM推理
    if steps_config.get("inference", False):
        print("------begin inference------", time.strftime("%Y-%m-%d %H:%M:%S"))
        # 判断是使用本地模型还是API
        use_api = config.get("use_api", False)
        if use_api:
            print("使用API模式进行推理")
            df = batch_llm_predict_api(config)
        else:
            print("使用本地模型进行推理")
            df = batch_llm_predict(config)
        print("------end inference------", time.strftime("%Y-%m-%d %H:%M:%S"))

    # 步骤2: 数据后处理
    if steps_config.get("postprocess", False):
        print("------begin postprocess------", time.strftime("%Y-%m-%d %H:%M:%S"))
        df = postprocess_data(config)
        print("------end postprocess------", time.strftime("%Y-%m-%d %H:%M:%S"))

    # 步骤3: 评测
    if steps_config.get("evaluate", False):
        print("------begin evaluate------", time.strftime("%Y-%m-%d %H:%M:%S"))
        evaluate_module(config, data=df)
        print("------end evaluate------", time.strftime("%Y-%m-%d %H:%M:%S"))

    # 步骤4: 纯字符串输出评测
    if steps_config.get("evaluate_output_str", False):
        print("------begin evaluate_output_str------", time.strftime("%Y-%m-%d %H:%M:%S"))
        evaluate_output_str_module(config, data=df)
        print("------end evaluate_output_str------", time.strftime("%Y-%m-%d %H:%M:%S"))

    print("======== 流水线执行完成 ========", time.strftime("%Y-%m-%d %H:%M:%S"))
