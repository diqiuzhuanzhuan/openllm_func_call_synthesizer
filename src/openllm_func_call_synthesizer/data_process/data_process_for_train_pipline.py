#!/usr/bin/env python
# coding: utf-8

# # 训练数据生成

import pandas as pd
import json
import numpy as np
import re
import os
import sys
sys.path.append('/data0/work/SusieSu/project')
sys.path.append('/data0/work/SusieSu/project/openllm_func_call_synthesizer/src/openllm_func_call_synthesizer/utils')
sys.path.append('/data0/work/SusieSu/project/openllm_func_call_synthesizer/tests/Standard_Test_Local_Model/')
from utils import read_json_file, format_value_counts
from parse_response_to_json import parse_react_to_json
from Call_LLM_Utils.read_file_util import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MCP_INTENT_LIST = ['video_search_control', 'create_album', 'search_photos',  'get_system_info', 'music_play_control', 'get_album_list', 'video_play_control', 'music_settings_control', 'music_search_control', 'unknown']
ULIYA_INTENT_LIST = ['search_document', 'general_query', 'translate', 'summary_document']

def detect_language(input_text):
    if isinstance(input_text, str):
        s = input_text.strip()
        # German check: e.g., contains ü, ö, ä, ß or typical German words
        if re.search(r'[üöäß]|[ÄÖÜ]', s) or re.search(r'\b(der|die|das|und|ein|nicht|ist|zu|in|den|mit|auf|für|sie|es|dem|von)\b', s, re.I):
            return 'ger'
        # Japanese check: hiragana, katakana, or frequent Japanese-specific patterns
        # Only classify as Japanese if contains **hiragana or katakana** (NOT CJK, which includes Chinese)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', s):
            return 'jap'
        # English check: contains [a-zA-Z] and high ASCII ratio, avoid misclassifying numbers/symbols
        if re.search(r'[a-zA-Z]', s):
            ascii_ratio = sum(1 for c in s if ord(c) < 128) / max(1, len(s))
            if ascii_ratio > 0.7:
                return 'en'
    # Default: assume Chinese if not above
    return 'zh'

# 填补language列为空的行
def complete_language_column(row):
    if isinstance(row['language'], str) and row['language'].strip():
        return row['language']
    else:
        return detect_language(row['input'])

def pre_columns_check(df):
  ## 1.1.如果没有language  给补充上
  df['language'] = df.apply(complete_language_column, axis=1)
  lang_intent_dist = df.groupby('language')['intent'].value_counts()
  print(lang_intent_dist)  # 打印每种语言下 intent 的分布

  ## 1.2 去掉bad case
  if 'bad_case' in df.columns:
    print('before remove bad case, df shape: ', df.shape)
    df['bad_case'] = df['bad_case'].replace('', np.nan)
    df = df[df['bad_case'].isna()]
    print('after remove bad case, df shape: ', df.shape, df.columns)
  
  print(df.shape, df.columns)
  print(df.iloc[0:2])

  return df

def standard_eval_json(x):
    try:
        if isinstance(x, str):
            return json.dumps(eval(x), ensure_ascii=False)
        else:
            print('not str', type(x), x)
            return x   
    except:
        print('error', type(x), x)
        return {} 

def pre_process(df):
  df = df.drop_duplicates(subset='input', inplace=False)
  # output 字段处理成标准的json
  df['output_ori']  = df['output'].copy()
  df['output'] = df['output'].apply(standard_eval_json)
  print(df.shape)
  
  df = df[df['output'] != {}]
  print('after remove empty output, df.shape: ', df.shape)
  print(df['intent'].value_counts().to_dict())

  return df

def concat_prompt(df):
  with open('/data0/work/SusieSu/project/openllm_func_call_synthesizer/examples/prompt_dict/system_prompt_mcp.txt', 'r', encoding='utf-8') as f1:
    system_prompt_mcp = f1.read()
  with open('/data0/work/SusieSu/project/openllm_func_call_synthesizer/examples/prompt_dict/system_prompt_uliya.txt', 'r', encoding='utf-8') as f2:
    system_prompt_uliya = f2.read()

  lora_input_list = []
  for i , df_0 in df.iterrows():
      df_ = df_0.to_dict()
      intent_ = df_.get('intent', '')
      system_prompt = system_prompt_mcp if intent_ in MCP_INTENT_LIST else system_prompt_uliya
      lora_input_list.append({
          "instruction": system_prompt,
          "input": df_.get("input", ""),
          "output": df_.get("output", "")  # json.dumps(df_.get('output', ""), ensure_ascii=False)   
      })
  df['lora_input'] = lora_input_list
  
  return df
  
def train_dev_test_split(df, root):

  df = shuffle(df)

  # 第一步：将数据分为训练集和临时集（包含 dev + test）
  train_df, temp_df = train_test_split(df, test_size=0.2, random_state=2025)

  # 第二步：将临时集再分为 dev 和 test
  dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=2025)

  print('train_df.shape, dev_df.shape, test_df.shape: ', train_df.shape, dev_df.shape, test_df.shape)
  print('train_df language value counts', train_df['language'].value_counts())
  print('dev_df language value counts: ', dev_df['language'].value_counts())
  print('test_df language value counts: ', test_df['language'].value_counts())

  # mcp_intent = ['video_search_control', 'create_album', 'search_photos',  'get_system_info', 'music_play_control', 'get_album_list', 'unknown', 'video_play_control', 'music_settings_control', 'music_search_control']
  # uliya_intent = ['search_document', 'general_query', 'translate', 'summary_document']

  # test_df1 = test_df[test_df['intent'].isin(mcp_intent)]
  # test_df2 = test_df[test_df['intent'].isin(uliya_intent)]
  # print(test_df1.shape, test_df2.shape)
  # test_df1.to_excel(root + 'test_all_mcp.xlsx')
  # test_df2.to_excel(root + 'test_all_uliya.xlsx')

  train_df.to_excel(root + 'train_all.xlsx')
  dev_df.to_excel(root + 'dev_all.xlsx')
  test_df.to_excel(root + 'test_all.xlsx')
  
  lora_input_list_train = train_df['lora_input'].to_list()
  lora_input_list_dev = dev_df['lora_input'].to_list()
  lora_input_list_test = test_df['lora_input'].to_list()
  
  with open(root + 'mcp_train.json', 'w') as fin:
    json.dump(lora_input_list_train, fin, ensure_ascii=False, indent=2)
  
  with open(root + 'mcp_dev.json', 'w') as fin2:
    json.dump(lora_input_list_dev, fin2, ensure_ascii=False, indent=2)
  
  with open(root + 'mcp_test.json', 'w') as fin3:
    json.dump(lora_input_list_test, fin3, ensure_ascii=False, indent=2)  

  return train_df, dev_df, test_df

if __name__ == '__main__':
  # "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/function_call_for_train_1112/function_call_for_train_1112.xlsx"
  root = '/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/function_call_for_train_1112_v2'
  for_train_root = os.path.join(root, 'mcp_data_1112_for_train/')
  if not os.path.exists(for_train_root):
      os.makedirs(for_train_root)

  input_file = os.path.join(root, 'function_call_for_train_1112.xlsx')
  output_file = os.path.join(root, 'raw_data_1112_for_train.xlsx')
  
  # 定义一个控制每一步是否执行的字典
  steps_control = {
      "post_process": False,
      "pre_process": False,
      "concat_prompt": False,
      "train_dev_test_split": True
  }

  # 0. 读取数据
  df = pd.read_excel(input_file)
  df.to_excel(output_file)

  # 1. 前处理：a.如果没有language给补充上 b.去掉bad case
  if steps_control["post_process"]:
      df = pre_columns_check(df)
      df.to_excel(output_file)

  # 2. 数据预处理  转成标准json格式。  这里可以调用标准的json处理方法
  if steps_control["pre_process"]:
      df = pre_process(df)
      df.to_excel(output_file)

  # 3. 拼接prompt， 不同的intent  拼不同prompt
  if steps_control["concat_prompt"]:
      df = concat_prompt(df)
      df.to_excel(output_file)

  # 4. 拆分 train  dev test，  保存好 xlsx和 json。 
  import ast
  # new add
  df['lora_input'] = df['lora_input'].apply(ast.literal_eval)
  if steps_control["train_dev_test_split"]:
      train_dev_test_split(df, for_train_root)



