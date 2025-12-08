#!/usr/bin/env python

"""
LLM本地模型调用模块
直接加载本地模型到GPU进行推理
"""

import ast
import json
import multiprocessing
import os
import time

import pandas as pd
from pandarallel import pandarallel
from parse_response_to_json import parse_react_to_json
from transformers import AutoModelForCausalLM, AutoTokenizer

pandarallel.initialize()


def setup_cuda_devices(cuda_device):
    """设置CUDA设备"""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def load_model_and_tokenizer(model_name):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    return tokenizer, model


with open("/data0/work/SusieSu/project/uliya/mcp/function_call_tools.json") as f:
    fun_ = json.load(f)

FUNCTIONS = fun_
function_call_system_prompt = "You are a helpful assistant. You are given a query and a function call. \
    You need to determine if the function call is correct for the query."


def get_response(tokenizer, model, prompt):
    """
    使用本地模型生成响应
    Args:
        tokenizer: 分词器
        model: 模型
        prompt: 输入提示词
    Returns:
        生成的文本
    """
    messages = [{"role": "system", "content": function_call_system_prompt}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, tools=FUNCTIONS
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=5000,
        temperature=0.01,
        top_p=0.1,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    _thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content


def get_response_mcp_intent(tokenizer, model, prompt):
    """
    使用本地模型生成响应
    Args:
        tokenizer: 分词器
        model: 模型
        prompt: 输入提示词
    Returns:
        生成的文本
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=5000,
        temperature=0.01,
        top_p=0.1,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content


def extract_intent(x):
    """从output字段中提取intent"""
    if pd.isna(x):
        return ""
    try:
        if isinstance(x, str):
            try:
                parsed = json.loads(x)
            except Exception:
                parsed = ast.literal_eval(x)
        else:
            parsed = x
        if isinstance(parsed, dict):
            return parsed.get("intent", "")
    except Exception:
        return ""
    return ""


def preprocess_data(df, config):
    """
    统一数据预处理：创建 mcp_input, uliya_input, gt_intent 列
    Args:
        df: 输入的DataFrame
        config: 配置字典
    Returns:
        处理后的DataFrame
    """
    input_field = config["input_field"]
    ground_truth_field = config.get("ground_truth", "output")
    system_prompt_mcp = config.get("system_prompt_mcp", "")
    system_prompt_uliya = config.get("system_prompt_uliya", "")

    # 创建两个输入列
    df["mcp_input"] = df[input_field].apply(lambda x: system_prompt_mcp + str(x))
    df["uliya_input"] = df[input_field].apply(lambda x: system_prompt_uliya + str(x))

    # 提取 gt_intent
    df["gt_intent"] = df[ground_truth_field].apply(extract_intent)

    return df


def batch_llm_predict(config):
    """
    使用本地模型批量调用LLM进行预测
    Args:
        config: 配置字典，需要包含以下字段:
            - cuda_device: CUDA设备号
            - model_file_path: 本地模型路径
            - input_file: 输入Excel文件路径
            - output_file: 输出Excel文件路径
            - system_prompt_mcp: MCP系统提示词
            - system_prompt_uliya: Uliya系统提示词
            - input_field: 输入字段名
            - ground_truth: Ground Truth字段名
            - mode: 模式选择 ('hybrid' 或 'split')
            - mcp_intent_list: MCP intent列表 (split模式需要)
            - uliya_intent_list: Uliya intent列表 (split模式需要)
    Returns:
        处理后的DataFrame
    """
    print("------begin time (Local Model)------", time.strftime("%Y-%m-%d %H:%M:%S"))

    # 初始化模型
    setup_cuda_devices(str(config["cuda_device"]))
    tokenizer, model = load_model_and_tokenizer(config["model_file_path"])

    # 读取数据并预处理
    df = pd.read_excel(config["input_file"])
    df = preprocess_data(df, config)

    # 获取模式配置
    mode = config.get("mode", "hybrid")
    print(f"Running in {mode} mode")

    # 定义单次推理函数
    def llm_once(x):
        try:
            rsp = get_response(tokenizer, model, x)
            try:
                parsed = parse_react_to_json(rsp)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed
            else:
                return rsp
        except Exception as e:
            print(f"LLM inference error for prompt: {x}\n{e}")
            return {}

    # 根据模式执行推理
    if mode == "hybrid":
        df = _run_hybrid_mode(df, llm_once, config)
    elif mode == "split":
        df = _run_split_mode(df, llm_once, config)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'hybrid' or 'split'.")

    # 保存结果
    df.to_excel(config["output_file"])
    print("Local Model LLM inference done, result saved:", config["output_file"])
    print("------end time (Local Model)------", time.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def _run_hybrid_mode(df, llm_once, config):
    """混合模式：先用 mcp_input 请求，如果没有结果，再用 uliya_input 请求"""
    print("Using hybrid mode: trying mcp_input first, then uliya_input if no result")

    def hybrid_inference(row):
        # 先用 mcp_input
        result = llm_once(row["mcp_input"])
        # 检查是否有有效结果
        if isinstance(result, dict):
            if result.get("intent", "") not in ["", "unknown"]:
                return result
        elif isinstance(result, str) and result.strip():
            return result

        # 如果 mcp_input 没有有效结果，使用 uliya_input
        print("-----------------------------用uliya prompt, Now use uliya prompt-----------------------------")
        print(row[config["input_field"]])
        return llm_once(row["uliya_input"])

    df["llm_response"] = df.apply(hybrid_inference, axis=1)
    return df


def _run_split_mode(df, llm_once, config):
    """分割模式：根据 gt_intent 选择对应的 prompt"""
    print("Using split mode: selecting prompt based on gt_intent")
    mcp_intent_list = config.get("mcp_intent_list", [])
    uliya_intent_list = config.get("uliya_intent_list", [])

    def split_inference(row):
        intent = row["gt_intent"]
        if intent in mcp_intent_list:
            return llm_once(row["mcp_input"])
        elif intent in uliya_intent_list:
            return llm_once(row["uliya_input"])
        else:
            # 如果不在任何列表中，默认使用 mcp_input
            print(f"Intent '{intent}' not in any list, using mcp_input by default")
            return llm_once(row["mcp_input"])

    df["llm_response"] = df.apply(split_inference, axis=1)
    return df


# 全局变量，用于多进程推理时在子进程中存储模型和tokenizer
global_tokenizer = None
global_model = None


def init_worker(model_file_path, cuda_device):
    """
    子进程初始化函数：在每个worker进程中加载模型
    Args:
        model_file_path: 模型路径
        cuda_device: CUDA设备号
    """
    global global_tokenizer, global_model
    setup_cuda_devices(str(cuda_device))
    global_tokenizer, global_model = load_model_and_tokenizer(model_file_path)


def llm_once_mp(x):
    """
    多进程环境下的单次LLM推理函数
    Args:
        x: prompt文本
    Returns:
        解析后的dict或原始字符串
    """
    global global_tokenizer, global_model
    try:
        rsp = get_response(global_tokenizer, global_model, x)
        try:
            parsed = parse_react_to_json(rsp)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and len(parsed) > 0:
            return parsed
        else:
            return rsp
    except Exception as e:
        print(f"LLM inference error for prompt: {x}\n{e}")
        return {}


def batch_llm_predict_threaded(config):
    """
    使用本地模型批量调用LLM进行预测（多进程版本）
    Args:
        config: 配置字典，需要包含以下字段:
            - cuda_device: CUDA设备号
            - model_file_path: 本地模型路径
            - input_file: 输入Excel文件路径
            - output_file: 输出Excel文件路径
            - system_prompt_mcp: MCP系统提示词
            - system_prompt_uliya: Uliya系统提示词
            - input_field: 输入字段名
            - ground_truth: Ground Truth字段名
            - mode: 模式选择 ('hybrid' 或 'split')
            - mcp_intent_list: MCP intent列表 (split模式需要)
            - uliya_intent_list: Uliya intent列表 (split模式需要)
    Returns:
        处理后的DataFrame
    """
    print("------begin time (Local Model, Multiprocessing)------", time.strftime("%Y-%m-%d %H:%M:%S"))

    # 初始化设备
    setup_cuda_devices(str(config["cuda_device"]))

    # 读取数据并预处理
    df = pd.read_excel(config["input_file"])
    df = preprocess_data(df, config)

    # 获取模式配置
    mode = config.get("mode", "hybrid")
    print(f"Running in {mode} mode")

    # 多进程推理配置
    num_processes = config.get("num_processes", max(1, multiprocessing.cpu_count() // 2))
    print("----num_processes------", num_processes)

    # 根据模式执行推理
    if mode == "hybrid":
        df = _run_hybrid_mode_mp(df, config, num_processes)
    elif mode == "split":
        df = _run_split_mode_mp(df, config, num_processes)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose 'hybrid' or 'split'.")

    # 保存结果
    df.to_excel(config["output_file"])
    print("Local Model LLM inference (multiprocessing) done, result saved:", config["output_file"])
    print("------end time (Local Model)------", time.strftime("%Y-%m-%d %H:%M:%S"))
    return df


def _run_hybrid_mode_mp(df, config, num_processes):
    """多进程混合模式"""
    print("Using hybrid mode: trying mcp_input first, then uliya_input if no result")

    # 第一轮：用 mcp_input
    prompts_mcp = df["mcp_input"].tolist()
    with multiprocessing.get_context("spawn").Pool(
        processes=num_processes, initializer=init_worker, initargs=(config["model_file_path"], config["cuda_device"])
    ) as pool:
        results_mcp = list(pool.imap(llm_once_mp, prompts_mcp))

    # 检查哪些需要用 uliya_input
    prompts_uliya_needed = []
    indices_need_uliya = []
    for idx, result in enumerate(results_mcp):
        need_uliya = False
        if isinstance(result, dict):
            if result.get("intent", "") in ["", "unknown"]:
                need_uliya = True
        elif isinstance(result, str) and not result.strip():
            need_uliya = True
        elif not result:
            need_uliya = True

        if need_uliya:
            prompts_uliya_needed.append(df.iloc[idx]["uliya_input"])
            indices_need_uliya.append(idx)

    # 第二轮：对需要的行用 uliya_input
    if prompts_uliya_needed:
        print(
            "--------------------需要用uliya prompt 的数据  use uliya prompt-----------------------------",
            prompts_uliya_needed,
        )
        print(f"Re-running {len(prompts_uliya_needed)} rows with uliya_input")
        with multiprocessing.get_context("spawn").Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(config["model_file_path"], config["cuda_device"]),
        ) as pool:
            results_uliya = list(pool.imap(llm_once_mp, prompts_uliya_needed))

        # 替换结果
        for i, idx in enumerate(indices_need_uliya):
            results_mcp[idx] = results_uliya[i]

    df["llm_response"] = results_mcp
    return df


def _run_split_mode_mp(df, config, num_processes):
    """多进程分割模式"""
    print("Using split mode: selecting prompt based on gt_intent")
    mcp_intent_list = config.get("mcp_intent_list", [])
    uliya_intent_list = config.get("uliya_intent_list", [])

    prompts = []
    for _idx, row in df.iterrows():
        intent = row["gt_intent"]
        if intent in mcp_intent_list:
            prompts.append(row["mcp_input"])
        elif intent in uliya_intent_list:
            prompts.append(row["uliya_input"])
        else:
            # 如果不在任何列表中，默认使用 mcp_input
            print(f"Intent '{intent}' not in any list, using mcp_input by default")
            prompts.append(row["mcp_input"])

    with multiprocessing.get_context("spawn").Pool(
        processes=num_processes, initializer=init_worker, initargs=(config["model_file_path"], config["cuda_device"])
    ) as pool:
        results = list(pool.imap(llm_once_mp, prompts))

    df["llm_response"] = results
    return df
