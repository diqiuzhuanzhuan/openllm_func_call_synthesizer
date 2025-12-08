import argparse
import ast
import json
import time

import pandas as pd
import yaml
from llm_api_caller import batch_llm_predict_api

# 导入本地模型和API调用模块
from llm_local_caller import batch_llm_predict_threaded
from calculate_confusion_matrix import get_confusion_matrix

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
    # 方式1: 根据 prompt_key 选择对应的 txt 文件（推荐，保留兼容性）
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

    # 新增：始终读取 mcp 和 uliya 两个 prompt
    if "system_prompt_mcp_file" in config:
        try:
            with open(config["system_prompt_mcp_file"], encoding="utf-8") as f:
                config["system_prompt_mcp"] = f.read().strip()
            print(f"✅ 从 {config['system_prompt_mcp_file']} 读取 MCP prompt")
            print("----config[system_prompt_mcp]----", config["system_prompt_mcp"])
        except Exception as e:
            print(f"⚠️  警告: 读取 MCP prompt 文件出错 {e}")
            config["system_prompt_mcp"] = ""

    if "system_prompt_uliya_file" in config:
        try:
            with open(config["system_prompt_uliya_file"], encoding="utf-8") as f:
                config["system_prompt_uliya"] = f.read().strip()
            print(f"✅ 从 {config['system_prompt_uliya_file']} 读取 Uliya prompt")
            print("----config[system_prompt_uliya]----", config["system_prompt_uliya"])
        except Exception as e:
            print(f"⚠️  警告: 读取 Uliya prompt 文件出错 {e}")
            config["system_prompt_uliya"] = ""

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


def evaluate_single_dataset(data, config, dataset_name="All"):
    """
    对单个数据集进行评测
    Args:
        data: 要评测的DataFrame
        config: 配置字典
        dataset_name: 数据集名称（用于打印）
    Returns:
        评测后的DataFrame
    """
    print(f"\n{'=' * 60}")
    print(f"开始评测数据集: {dataset_name} (共 {len(data)} 条)")
    print(f"{'=' * 60}")

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
    print(f"\n【{dataset_name}】函数相等分布:", data["function_same"].value_counts())
    print(f"【{dataset_name}】参数相等分布:", data["argument_same"].value_counts())

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
    rprint(f"[bold cyan]=== 【{dataset_name}】评测结果简表 ===[/bold cyan]")
    for info in summary_info:
        rprint(
            f"[bold yellow]{info['指标名称']}[/bold yellow] | "
            f"总数: [bold magenta]{info['总数']}[/bold magenta] | "
            f"正确: [bold green]{info['正确(True)']}[/bold green] | "
            f"错误: [bold red]{info['错误(False)']}[/bold red] | "
            f"准确率: [bold blue]{info['准确率']}[/bold blue]"
        )
    print(f"\n【{dataset_name}】详细数值如下：")
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
    rprint(f"[bold cyan]=== 【{dataset_name}】argument评测结果分布 ===[/bold cyan]")
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
    rprint(f"\n[bold cyan]========== 【{dataset_name}】argument具体计算过程如下 ==========[/bold cyan]")
    rprint(f"[bold yellow]数据总数 total[/bold yellow] = [bold magenta]{total}[/bold magenta]")
    rprint(f"[bold yellow]完全一致 exact_match 数量[/bold yellow] = [bold green]{exact_match_count}[/bold green]")
    rprint(f"[bold yellow]部分匹配 partial_match 数量[/bold yellow] = [bold blue]{partial_match_count}[/bold blue]")
    rprint(f"[bold yellow]子集关系 subset 数量[/bold yellow] = [bold cyan]{subset_match_count}[/bold cyan]")

    rprint_msg1 = (
        "[bold]([green]exact_match[/green] + [blue]partial_match[/blue]) / total = "
        f"[green]{exact_match_count + partial_match_count}[/green] / [magenta]{total}[/magenta] = "
        f"[bold red]{accuracy:.4%}[/bold red][/bold]"
    )
    rprint(rprint_msg1)

    combined_accuracy = combined_count / total if total > 0 else 0

    rprint_msg2 = (
        "[bold]([green]exact_match[/green] + [blue]partial_match[/blue] + [cyan]subset[/cyan]) / total = "
        f"[green]{combined_count}[/green] / [magenta]{total}[/magenta] = "
        f"[bold red]{combined_accuracy:.4%}[/bold red][/bold]"
    )
    rprint(rprint_msg2)

    # 每种情况分别展示两条例子
    print(f"\n【{dataset_name}】各评测结果类型示例:")
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

    return data


def evaluate_module(config, data=None):
    """
    主评测函数：根据配置决定是合并评测还是分开评测
    """
    if data is None:
        data = pd.read_excel(config["evaluate_input_file"])

    # 获取评测模式配置
    evaluate_separate = config.get("evaluate_separate", False)

    if evaluate_separate:
        # 分开评测模式
        print("\n" + "=" * 80)
        print("【分开评测模式】根据 intent 类型分别评测 MCP 和 Uliya 数据")
        print("=" * 80)

        mcp_intent_list = config.get("mcp_intent_list", [])
        uliya_intent_list = config.get("uliya_intent_list", [])

        # 分离数据
        data_mcp = data[data["gt_intent"].isin(mcp_intent_list)].copy()
        data_uliya = data[data["gt_intent"].isin(uliya_intent_list)].copy()
        data_other = data[~data["gt_intent"].isin(mcp_intent_list + uliya_intent_list)].copy()

        print(f"\n数据分布: MCP={len(data_mcp)} 条, Uliya={len(data_uliya)} 条, Other={len(data_other)} 条")

        # 分别评测
        if len(data_mcp) > 0:
            data_mcp = evaluate_single_dataset(data_mcp, config, dataset_name="MCP")

        if len(data_uliya) > 0:
            data_uliya = evaluate_single_dataset(data_uliya, config, dataset_name="Uliya")

        if len(data_other) > 0:
            data_other = evaluate_single_dataset(data_other, config, dataset_name="Other")

        # 合并结果并保存
        data_evaluated = pd.concat([data_mcp, data_uliya, data_other], ignore_index=False).sort_index()

        # 保存分别的结果
        # output_dir = config["evaluate_output_file"].rsplit("/", 1)[0] \
        # if "/" in config["evaluate_output_file"] else "."
        output_base = config["evaluate_output_file"].rsplit(".", 1)[0]

        if len(data_mcp) > 0:
            data_mcp.to_excel(f"{output_base}_mcp.xlsx", index=False)
            print(f"\nMCP 评测结果保存于: {output_base}_mcp.xlsx")

        if len(data_uliya) > 0:
            data_uliya.to_excel(f"{output_base}_uliya.xlsx", index=False)
            print(f"Uliya 评测结果保存于: {output_base}_uliya.xlsx")

        if len(data_other) > 0:
            data_other.to_excel(f"{output_base}_other.xlsx", index=False)
            print(f"Other 评测结果保存于: {output_base}_other.xlsx")

        data_evaluated.to_excel(config["evaluate_output_file"], index=False)
        print(f"完整评测结果保存于: {config['evaluate_output_file']}")

        return data_evaluated
    else:
        # 合并评测模式
        print("\n" + "=" * 80)
        print("【合并评测模式】对所有数据进行统一评测")
        print("=" * 80)

        data = evaluate_single_dataset(data, config, dataset_name="All")
        data.to_excel(config["evaluate_output_file"], index=False)
        print(f"\n评测结果保存于: {config['evaluate_output_file']}")
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
    增强版解析函数：
    1. 优先尝试标准 JSON 解析 (支持 null)
    2. 失败则尝试 Python AST 解析 (支持 None, 单引号)
    3. 自动解包列表 (如果模型输出了List包含Dict)
    """
    if isinstance(x, dict):
        return x
    
    parsed_result = None
    
    if isinstance(x, str):
        # --- 尝试 1: 标准 JSON (处理 "null", 双引号) ---
        try:
            parsed_result = json.loads(x)
        except Exception:
            # --- 尝试 2: Python AST (处理 "None", 单引号) ---
            try:
                # 预处理: 把 json 的 null/true/false 换成 python 的 None/True/False
                x_python = x.replace("null", "None").replace("true", "True").replace("false", "False")
                parsed_result = ast.literal_eval(x_python)
            except Exception:
                # --- 尝试 3: 暴力修复 (修正双引号包裹单引号字典的情况) ---
                try:
                    res = fix_inner_single_quotes(x)
                    if res is not None:
                        parsed_result = res
                    else:
                        # 最后兜底
                        x_json_like = x.replace("'", '"')
                        parsed_result = json.loads(x_json_like)
                except Exception:
                    print(f"所有方式均失败: {x}")
                    return {}

    # --- 统一处理结果 ---
    
    # 情况 A: 解析出来是列表 (e.g. [{"intent":...}]) -> 取第一个元素
    if isinstance(parsed_result, list):
        if len(parsed_result) > 0 and isinstance(parsed_result[0], dict):
            return parsed_result[0]
        else:
            return {} # 列表为空或里面不是字典

    # 情况 B: 解析出来是字典 -> 直接返回
    if isinstance(parsed_result, dict):
        return parsed_result

    # 情况 C: 其他奇怪类型 -> 返回空字典
    return {}


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

    df[llm_intent_col] = df[llm_response_col].apply(
        lambda x: x.get("intent", "") if isinstance(x, dict) and "intent" in x.keys() else (x.get("name", "") if isinstance(x, dict) and "name" in x.keys()
                else ""))
    
    df[llm_slot_col] = df[llm_response_col].apply(
        lambda x: x.get("slots", "") if isinstance(x, dict) and "slots" in x.keys() else (x.get("arguments", "") if isinstance(x, dict) and "arguments" in x.keys()
                else ""))
    # else:
    #     df[llm_intent_col] = df[llm_response_col].apply(lambda x: x.get("name", "") if isinstance(x, dict) else "")
    #     df[llm_slot_col] = df[llm_response_col].apply(lambda x: x.get("arguments", {}) if isinstance(x, dict) else {})

    # 处理Ground Truth列
    gt_col = config["ground_truth"]
    gt_intent_col = config["ground_truth_intent"]
    gt_slot_col = config["ground_truth_slot"]

    # 保证为dict
    df[gt_col] = df[gt_col].apply(robust_parse_v2)


    df[gt_intent_col] = df[gt_col].apply(
        lambda x: x.get("intent", "") if isinstance(x, dict) and "intent" in x.keys() else (x.get("name", "") if isinstance(x, dict) and "name" in x.keys()
                else ""))
    df[gt_slot_col] = df[gt_col].apply(
        lambda x: x.get("slots", "") if isinstance(x, dict) and "slots" in x.keys() else (x.get("arguments", "") if isinstance(x, dict) and "arguments" in x.keys()
                else ""))

    # if config["model_type"] == "uliya_intent":
    #     df[gt_intent_col] = df[gt_col].apply(lambda x: x.get("intent", "") if isinstance(x, dict) else "")
    #     df[gt_slot_col] = df[gt_col].apply(lambda x: x.get("slots", {}) if isinstance(x, dict) else {})
    # else:
    #     df[gt_intent_col] = df[gt_col].apply(lambda x: x.get("name", "") if isinstance(x, dict) else "")
    #     df[gt_slot_col] = df[gt_col].apply(lambda x: x.get("arguments", {}) if isinstance(x, dict) else {})

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
            # df = batch_llm_predict(config)
            df = batch_llm_predict_threaded(config)
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
        get_confusion_matrix(df, config["confusion_matrix_file"])
        print("------end calculate confusion matrix------", time.strftime("%Y-%m-%d %H:%M:%S"))
    # 步骤4: 纯字符串输出评测
    if steps_config.get("evaluate_output_str", False):
        print("------begin evaluate_output_str------", time.strftime("%Y-%m-%d %H:%M:%S"))
        evaluate_output_str_module(config, data=df)
        print("------end evaluate_output_str------", time.strftime("%Y-%m-%d %H:%M:%S"))

    print("======== 流水线执行完成 ========", time.strftime("%Y-%m-%d %H:%M:%S"))
   