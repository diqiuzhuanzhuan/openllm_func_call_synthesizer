import os
import re
import json
import time
import yaml
import argparse
import pandas as pd
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_parse_function import parse_react_to_json

def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_cuda_devices(cuda_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model


def get_response(tokenizer, model, prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=5000,
        temperature=0.01,
        top_p=0.1,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content

def batch_llm_predict(config):
    print('------begin time------', time.strftime("%Y-%m-%d %H:%M:%S"))
    setup_cuda_devices(str(config['cuda_device']))
    tokenizer, model = load_model_and_tokenizer(config['model_name'])
    df = pd.read_excel(config['input_file'])
    # 生成prompt
    df['all_prompt_to_test'] = df.apply(lambda x: config['system_prompt'] + str(x[config['input_field']]), axis=1)
    df.to_excel(config['output_file'])

    def llm_once(x):
        try:
            rsp = get_response(tokenizer, model, x)
            result = parse_react_to_json(rsp)
            return result
        except Exception as e:
            print(f"LLM inference error for prompt: {x}\n{e}")
            return {}
    df['llm_response'] = df['all_prompt_to_test'].apply(llm_once)

    # 保存初步结果
    df.to_excel(config['output_file'])
    print("LLM inference done, result saved:", config['output_file'])
    return df

# 评测相关代码
def evaluate_arguments(ground_truth, model_response):
    """
    评测函数参数的准确性
    Returns:
        str: 评测结果 ('exact_match', 'has_extra_fields', 'partial_match', 'no_match')
    """
    if not isinstance(ground_truth, dict) or not isinstance(model_response, dict):
        return 'no_match'
    if ground_truth == {} and model_response == {}:
        return 'exact_match'
    if not ground_truth or not model_response:
        return 'no_match'
    # 1. 完全一致
    if ground_truth == model_response:
        return 'exact_match'
    # 2. 检查是否有多余字段
    gt_keys = set(ground_truth.keys())
    mr_keys = set(model_response.keys())
    has_extra_fields = len(mr_keys - gt_keys) > 0
    if has_extra_fields:
        return 'has_extra_fields'
    # 3. 部分匹配
    common_keys = gt_keys & mr_keys
    if common_keys:
        for key in common_keys:
            if ground_truth[key] == model_response[key]:
                return 'partial_match'
    return 'no_match'

def evaluate_module(config, data=None):
    if data is None:
        data = pd.read_excel(config['evaluate_input_file'])
    # 字段名
    ground_truth_intent = config['ground_truth_intent']
    ground_truth_slot = config['ground_truth_slot']
    llm_intent = config['llm_intent']
    llm_slot = config['llm_slot']
    # 保证字段转dict
    data[ground_truth_slot] = data[ground_truth_slot].apply(lambda x: eval(x) if isinstance(x, str) else x)
    data[llm_slot] = data[llm_slot].apply(lambda x: eval(x) if isinstance(x, str) else x)
    # 函数/槽分开比对
    data["function_same"] = data.apply(lambda x: True if x[ground_truth_intent]==x[llm_intent] else False, axis=1)
    data["argument_same"] = data.apply(lambda x: True if x[ground_truth_slot]==x[llm_slot] else False, axis=1)
    print("函数相等分布:", data["function_same"].value_counts())
    print("参数相等分布:", data["argument_same"].value_counts())

    # 漂亮打印
    from rich import print as rprint
    summary_info = []
    for col in ['function_same', 'argument_same']:
        true_count = data[col].sum()
        false_count = (~data[col]).sum()
        total = true_count + false_count
        accuracy = true_count / total if total > 0 else 0

        summary = {
            "指标名称": col,
            "总数": total,
            "正确(True)": int(true_count),
            "错误(False)": int(false_count),
            "准确率": f"{accuracy:.4%}"
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
    print("评测结果分布:")
    value_counts = data['argument_evaluation_result'].value_counts()
    print(value_counts)

    # 统计相关数量
    total = len(data)
    exact_match_count = (data['argument_evaluation_result'] == 'exact_match').sum()
    partial_match_count = (data['argument_evaluation_result'] == 'partial_match').sum()
    exact_plus_partial = exact_match_count + partial_match_count

    accuracy = exact_plus_partial / total if total > 0 else 0

    print("\n========== 具体计算过程如下 ==========")
    print(f"数据总数 total = {total}")
    print(f"exact_match 数量 = {exact_match_count}")
    print(f"partial_match 数量 = {partial_match_count}")
    print(f"exact_match + partial_match = {exact_plus_partial}")
    print(f"(exact_match + partial_match) / total = {exact_plus_partial} / {total} = {accuracy:.4%}")
    print(f"\n最终准确率 (exact_match+partial_match/total)：{accuracy:.4%}")

    # 每种情况分别展示两条例子
    print("\n各评测结果类型示例:")
    eval_types = ["exact_match", "partial_match", "has_extra_fields", "no_match"]
    for eval_type in eval_types:
        subset = data[data['argument_evaluation_result'] == eval_type]
        shown = 0
        if len(subset) == 0:
            continue
        print(f"--- 评测结果类型: {eval_type} ---")
        for idx, row in subset.head(2).iterrows():
            print(f"行 {idx}:")
            print(f"  Ground Truth: {row[config['ground_truth_slot']]}")
            print(f"  Model Response: {row[config['llm_slot']]}")
            print(f"  评测结果: {row['argument_evaluation_result']}")
            print()

    data.to_excel(config['evaluate_output_file'])
    print("评测结果保存于:", config['evaluate_output_file'])
    return data

def postprocess_data(config):
    """
    数据后处理：提取LLM响应和Ground Truth到标准字段名
    合并了原来的ensure_llm_output_columns和ensure_gt_output_columns功能
    """
    input_file = config['postprocess_input_file']
    output_file = config['postprocess_output_file']
    
    df = pd.read_excel(input_file)
    
    # 处理LLM输出列
    llm_response_col = 'llm_response'
    llm_intent_col = config['llm_intent']
    llm_slot_col = config['llm_slot']
    
    # 保证llm_response为dict
    df[llm_response_col] = df[llm_response_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df[llm_intent_col] = df[llm_response_col].apply(lambda x: x.get('intent', '') if isinstance(x, dict) else '')
    df[llm_slot_col] = df[llm_response_col].apply(lambda x: x.get('slots', {}) if isinstance(x, dict) else {})
    
    # 处理Ground Truth列
    gt_col = config['ground_truth']
    gt_intent_col = config['ground_truth_intent']
    gt_slot_col = config['ground_truth_slot']
    
    # 保证为dict
    df[gt_col] = df[gt_col].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df[gt_intent_col] = df[gt_col].apply(lambda x: x.get('intent', '') if isinstance(x, dict) else '')
    df[gt_slot_col] = df[gt_col].apply(lambda x: x.get('slots', {}) if isinstance(x, dict) else {})
    
    df.to_excel(output_file, index=False)
    print(f"postprocess_data save to: {output_file}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Function Call Synthesizer with Configurable Evaluation")
    parser.add_argument('--config', type=str, required=True, help='YAML配置文件路径')
    args = parser.parse_args()
    config = load_config(args.config)

    # 根据配置文件中的steps配置自动执行相应步骤
    steps_config = config.get('steps', {})
    
    print('======== 开始执行流水线 ========', time.strftime("%Y-%m-%d %H:%M:%S"))
    print('执行步骤配置:', steps_config)
    
    df = None  # 用于在步骤间传递数据
    
    # 步骤1: LLM推理
    if steps_config.get('inference', False):
        print('------begin inference------', time.strftime("%Y-%m-%d %H:%M:%S"))
        df = batch_llm_predict(config)
        print('------end inference------', time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # 步骤2: 数据后处理
    if steps_config.get('postprocess', False):
        print('------begin postprocess------', time.strftime("%Y-%m-%d %H:%M:%S"))
        df = postprocess_data(config)
        print('------end postprocess------', time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # 步骤3: 评测
    if steps_config.get('evaluate', False):
        print('------begin evaluate------', time.strftime("%Y-%m-%d %H:%M:%S"))
        evaluate_module(config, data=df)
        print('------end evaluate------', time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print('======== 流水线执行完成 ========', time.strftime("%Y-%m-%d %H:%M:%S"))