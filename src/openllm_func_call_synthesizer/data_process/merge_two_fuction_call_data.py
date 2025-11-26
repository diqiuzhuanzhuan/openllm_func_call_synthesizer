# function call 数据合并

import pandas as pd
import json

# df = pd.read_excel('/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1124/function_call_data_1124.xlsx')
# df.shape, df.columns

# # 将 'lora_input' 列写进 jsonl 文件，每行为一个json对象
# with open('/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1124/susie_train_raw.jsonl', 'w', encoding='utf-8') as fout:
#     for item in df['lora_input']:
#         # 如果内容已经是字符串，确保内容正常
#         if isinstance(item, str):
#             # 你可以选择直接写字符串, 或 eval/load 并json.dump
#             try:
#                 # 尝试将字符串转为对象
#                 obj = json.loads(item)
#             except Exception:
#                 try:
#                     obj = eval(item)
#                 except Exception:
#                     obj = item  # 直接写原始字符串
#             json.dump(obj, fout, ensure_ascii=False)
#             fout.write('\n')
#         else:
#             json.dump(item, fout, ensure_ascii=False)
#             fout.write('\n')
# 读取这两个文件，合并成一个新文件 jsonl文件（每行一个json对象）
import json

file1 = '/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1124/train.jsonl'
file2 = '/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1124/susie_train_raw.jsonl'
output_file = '/data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1124/raw_train_1124.jsonl'

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data

data1 = read_jsonl(file1)
data2 = read_jsonl(file2)

# 合并数据
all_data = data1 + data2

with open(output_file, 'w', encoding='utf-8') as fout:
    for item in all_data:
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"合并完成, 共 {len(all_data)} 条数据，写入 {output_file}")
