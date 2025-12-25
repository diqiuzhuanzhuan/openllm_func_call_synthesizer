import ast
import json

# from LLM_result2excel import json_to_excel_answers
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


system_prompt_en = """
你是一个智能翻译助手，你将协助用户将输入内容翻译成英语，只输出翻译本身，不要任何解释说明。
注意: 1.中文人名替换成英国、美国明星名字
      2.中文电影名替换成英语电影名，如 琅琊榜--->Good Will Hunting  或 The Truman Show 或 Forrest Gump
      3.输出时请用双引号包裹字符串，例如请输出 "The Truman Show" 而不是'The Truman Show'

例子：
用户输入：
{
"input": "最近有什么新电影可以看？",
"output": "{'intent': 'video_search_control', 'slots': {'title': '最近上映', 'type': 'movie'}}"
}

翻译输出：
{
"input": "What are the latest movies available to watch?",
"output": "{"intent": "video_search_control", "slots": {"title": "the latest movies", "type": "movie"}}"
}

请严格按照上述要求进行翻译，确保 `output` 的 `slots` 值与翻译后的 `input` 内容保持一致。"""

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
"output": "{'intent': 'video_search_control', 'slots': {'title': '最近上映', 'type': 'movie'}}"
}

翻译输出：
{
"input": "最近の新しい映画は何ですか？",
"output": "{"intent": "video_search_control", "slots": {"title": "最近の新しい映画", "type": "movie"}}"
}

请严格按照上述要求进行翻译，确保 `output` 的 `slots` 值与翻译后的 `input` 内容保持一致。"""


input_path = "./new_mcp_datas/expanded_train_data_susie.xlsx"
output_path = "./new_mcp_datas/expanded_train_data_susie_en.xlsx"

translated = []
df = pd.read_excel(input_path, sheet_name="merge")
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


df = df.reset_index(drop=True)


for _idx, line in df.iterrows():
    user_input = {"input": line["input"], "output": line["output"]}
    result = get_llm_response(system_prompt_en, user_input)
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
