import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
import json
import re
import time

import pandas as pd
import requests

URL = "http://localhost:8010/v1/chat/completions"

def get_vllm_response_one(user_input, system_prompt, api_url=URL, model_name="qwen3_1.7b_mix", temperature=0.01, top_p=0.1, max_tokens=500):
    """
    调用vLLM服务进行推理，返回LLM输出内容
    """
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": temperature,
        "top_p": top_p,
        # 这个是最大输出长度
        "max_tokens": max_tokens
        # "repetition_penalty": 1.5
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # vllm标准返回格式
    return data["choices"][0]["message"]["content"]

def parse_fuction_call_response(response):
    """
    处理LLM返回结果:
    - 支持有<think>...</think>和<tool_call>...</tool_call>嵌套、或只返回<think>...或普通文本等。
    - 返回结果(dict 或 str)
    返回内容规则:
        若存在<tool_call>...</tool_call>，返回解析后的dict
        否则返回主体文本（去除<think>片段，strip整体）
    """

    import re
    import json

    if not isinstance(response, str):
        return response

    s = response.strip()

    # 提取<tool_call>块
    tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_call_match = tool_call_pattern.search(s)
    if tool_call_match:
        content = tool_call_match.group(1).strip()
        try:
            result = json.loads(content)
            # arguments 字段如果是字符串的json,再解析一次
            if 'arguments' in result and isinstance(result['arguments'], str):
                try:
                    arguments_dict = json.loads(result['arguments'])
                    result['arguments'] = arguments_dict
                except Exception:
                    pass
            return result
        except Exception:
            # 非法json,直接返回提取到的内容
            return content
    
    # 没有<tool_call>，去掉<think>块，仅保留剩余主体文本
    # 移除所有<think>...</think>
    no_think = re.sub(r"<think>(.*?)</think>", "", s, flags=re.DOTALL)
    # 如果去掉think后还剩内容，直接strip
    result = no_think.strip()
    print('-------result-------',result)
    return result

def get_vllm_response(x, system_prompt):
    try:
        rs = get_vllm_response_one(system_prompt, x)
        return parse_fuction_call_response(rs)
    except Exception as e:
        print(e)
        return None
                
if __name__ == "__main__":
    system_prompt = "You are a helpful assistant. You are given a query and a function call.  You need to determine if the function call is correct for the query."
    # # input = "我喜欢蔡健雅，你喜欢不啊"
    # # input = "我想找周杰伦的稻香"
    # input = "请播放一些古典音乐。"
    # # input = "给我翻译下这个句子好不好？"
    # rs = get_response(system_prompt, input, MODEL, TOKENIZER)
    # rs1 = parse_fuction_call_response(rs)
    # print('-------------------------\n',rs1,type(rs1))


    df = pd.read_excel('/data0/work/SusieSu/project/openllm_datas_and_temp_codes/DPO_data/1208/test_all.xlsx')
    df.shape, df.columns

    df['llm_response'] = df['input'].apply(lambda x: get_vllm_response(x))
    df.to_excel('/data0/work/SusieSu/project/openllm_datas_and_temp_codes/DPO_data/1208/test_all_llm_response_vllm.xlsx', index=False)