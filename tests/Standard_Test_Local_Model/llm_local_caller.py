#!/usr/bin/env python
# coding: utf-8

"""
LLM本地模型调用模块
直接加载本地模型到GPU进行推理
"""

import os
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from parse_response_to_json import parse_react_to_json


def setup_cuda_devices(cuda_device):
    """设置CUDA设备"""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device


def load_model_and_tokenizer(model_name):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model


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
    """
    使用本地模型批量调用LLM进行预测
    Args:
        config: 配置字典，需要包含以下字段:
            - cuda_device: CUDA设备号
            - model_file_path: 本地模型路径
            - input_file: 输入Excel文件路径
            - output_file: 输出Excel文件路径
            - system_prompt: 系统提示词
            - input_field: 输入字段名
    Returns:
        处理后的DataFrame
    """
    print('------begin time (Local Model)------', time.strftime("%Y-%m-%d %H:%M:%S"))
    setup_cuda_devices(str(config['cuda_device']))
    tokenizer, model = load_model_and_tokenizer(config['model_file_path'])
    df = pd.read_excel(config['input_file'])
    
    # 生成prompt
    df['all_prompt_to_test'] = df.apply(lambda x: config['system_prompt'] + str(x[config['input_field']]), axis=1)
    df.to_excel(config['output_file'])

    def llm_once(x):
        try:
            rsp = get_response(tokenizer, model, x)
            # 如果返回能解析为非空字典，则走结构化解析；否则直接返回原始字符串
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
    
    df['llm_response'] = df['all_prompt_to_test'].apply(llm_once)

    # 保存初步结果
    df.to_excel(config['output_file'])
    print("Local Model LLM inference done, result saved:", config['output_file'])
    print('------end time (Local Model)------', time.strftime("%Y-%m-%d %H:%M:%S"))
    return df

