import ast
import json

# from LLM_result2excel import json_to_excel_answers
import re

import pandas as pd
from openai import OpenAI

# 初始化客户端
# client = OpenAI(api_key="你的API密钥")

# openai_api_key = "sk-hN0QOFujOfx5BZTZvG8UjAdtle7qmzFFsJCLTkcmFWvNUXo7"
openai_api_key = "sk-l0JoMukojgY9jWCCHVfW6zzBtPHrC9F2d3Bom9eGzdKom3K1"
openai_api_base = "https://api9.xhub.chat/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def get_llm_response(system_prompt, input_list_str):
    # 调用 GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",  # 指定模型为 GPT-4o
        messages=[{"role": "system", "content": str(system_prompt)}, {"role": "user", "content": str(input_list_str)}],
        temperature=0.1,  # 控制输出随机性（0-1，越高越随机）
        max_tokens=1000,  # 限制最大输出 tokens
    )

    return response.choices[0].message.content.strip()


system_prompt = """
你是一个智能翻译助手，你将协助用户将输入内容翻译成日语，只输出翻译本身，不要任何解释说明。
注意: 1.中文人名替换成日本明星名字
      2.中文电影名翻译成日语电影名，如 琅琊榜--->千と千尋の神隠し
      3.输出时请用双引号包裹字符串，例如请输出 "子供のアルバムをつくり作ります" 而不是'子供のアルバムをつくり作ります'

其中有些对应规则：
宝宝相册--->ベビーアルバム
条件相册--->条件付きアルバム
人物相册--->人物アルバム
普通相册--->一般アルバム

例：
用户输入：
{
"input":"新建一个名为校园生活的相册",
"output":"{'intent': 'create_album', 'slots': {'album_name': '校园生活'}}"
}

请严格输出json格式：
{
"input": "「学校生活」というアルバムを作成する",
"output": "{"intent": "create_album", "slots": {"album_name": "「学校生活」"}}"
}
"""

system_prompt_jap = """你是一个智能翻译助手，你将协助用户将输入内容翻译成日语，只输出翻译本身，不要任何解释说明。

注意：
1. 中文人名替换成日本明星名字；
2. 中文电影名翻译成日语电影名，例如：琅琊榜 ---> 千と千尋の神隠し；
3. 输出时请使用双引号包裹字符串，例如：请输出 "子供のアルバムをつくり作ります" 而不是 '子供のアルバムをつくり作ります'；
4. 如果 `output` 中的 `slots` 包含中文内容，需将其值与翻译后的 `input` 内容一致，不能直接使用原始的中文值。

例子：
用户输入：
{
"input": "最近有什么新电影可以看？",
"output": "{'intent': 'video_search_control', 'slots': {'title': '新电影', 'type': 'movie'}}"
}

翻译输出：
{
"input": "最近の新しい映画は何ですか？",
"output": "{"intent": "video_search_control", "slots": {"title": "最近の新しい映画", "type": "movie"}}"
}

请严格按照上述要求进行翻译，确保 `output` 的 `slots` 值与翻译后的 `input` 内容保持一致。"""

root = "/data0/work/SusieSu/project/openllm_func_call_synthesizer/src/openllm_func_call_synthesizer/data_process/"
input_path = root + "new_mcp_datas/mcp_zh_data_1014_v2.xlsx"
output_path = root + "new_mcp_datas/mcp_zh_data_1014_v2_jap.xlsx"
output_path_check = root + "new_mcp_datas/mcp_zh_data_1014_v2_jap_check.xlsx"

translated = []
df = pd.read_excel(input_path)
df = df.drop_duplicates(subset=["input"])
df["output"] = df["output"].apply(eval)


def safe_parse(s):
    """多层解析，直到不能解析为止"""
    if not isinstance(s, str):
        return s
    for _ in range(3):
        try:
            s = ast.literal_eval(s)
        except Exception:
            try:
                s = json.loads(s)
            except Exception:
                break
    return s


for _idx, line in df.iterrows():
    user_input = {"input": line["input"], "output": line["output"]}
    result = get_llm_response(system_prompt_jap, user_input)
    print(result)  # 观察模型输出

    # 尝试多层解析
    final_res = safe_parse(result)
    if not isinstance(final_res, dict):
        translated.append({"input": line["input"], "output": result})
        continue

    input_val = final_res.get("input", line["input"])
    output_val = safe_parse(final_res.get("output", ""))

    translated.append({"input": input_val, "output": output_val})

df_result = pd.DataFrame(translated)
df_result.to_excel(output_path, index=False)
print(f"✅ 转换完成，结果保存至 {output_path}")

# 检查数据

df = pd.read_excel(output_path)


def safe_eval_output(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return x  # 如果解析失败，保留原字符串


df["output"] = df["output"].apply(safe_eval_output)


def contains_japanese(text):
    # 判断字符串是否包含日语字符
    if not isinstance(text, str):
        return False
    return re.search("[\u3040-\u30ff\u4e00-\u9faf]", text) is not None


def contains_chinese(text):
    # 判断字符串是否包含中文字符
    if not isinstance(text, str):
        return False
    return re.search("[\u4e00-\u9fff]", text) is not None


def is_pure_english(s):
    if not isinstance(s, str):
        return False
    # 匹配纯英文字母或空格
    return re.fullmatch(r"[A-Za-z\s\-_0-9]+", s) is not None


def slots_japanese_in_intent(output, intent_text):
    try:
        slots = output.get("slots", {})
        # intent_text = str(output.get('intent', ''))
        for v in slots.values():
            v_str = str(v)
            # 只检查含日语的slot内容
            if contains_japanese(v_str):
                if v_str not in intent_text:
                    print(v_str, intent_text)
                    return False
            # 纯英文 value 跳过检查
            elif is_pure_english(v_str):
                continue
        return True
    except Exception:
        return False


def slots_chinese_in_intent(output, intent_text):
    try:
        slots = output.get("slots", {})
        # intent_text = str(output.get('intent', ''))
        for v in slots.values():
            v_str = str(v)
            # 只检查含日语的slot内容
            if contains_chinese(v_str):
                if v_str not in intent_text:
                    print(v_str, intent_text)
                    return False
            # 纯英文 value 跳过检查
            elif is_pure_english(v_str):
                continue
        return True
    except Exception:
        return False


df["slots_in_intent"] = df.apply(lambda x: slots_japanese_in_intent(x["output"], x["input"]), axis=1)

print(df["slots_in_intent"].value_counts())
df2 = df[df["slots_in_intent"]]

df2["intent"] = df2["output"].apply(lambda x: x.get("intent", ""))
df2["intent"].value_counts()
df2.to_excel(output_path_check, index=False)
