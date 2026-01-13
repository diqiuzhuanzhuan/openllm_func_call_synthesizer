from datasets import load_dataset, load_from_disk
import json


def convert_tool_calls(example):
    if example["messages"][-1]["tool_calls"]:
        print(example["messages"][-1]["tool_calls"])
        for i in range(len(example["messages"][-1]["tool_calls"])):
            example["messages"][-1]["tool_calls"][i]["function"]['arguments'] = json.loads(example["messages"][-1]["tool_calls"][i]["function"]['arguments'])
        print(json.dumps(example["messages"][-1]["tool_calls"], ensure_ascii=False))




data_files = "./data/function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07_llama_factory_20251212"
from pathlib import Path
critic_dataset_path = Path("./data/susie_function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07")
data = load_dataset(data_files)
data = load_dataset("json", data_files={"train": str(critic_dataset_path / "train.jsonl")})
data = data.map(lambda x: convert_tool_calls(x))
data['train'].to_json("data/susie_function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07_llama_factory_20260102/train.jsonl", orient="records", lines=True, force_ascii=True)
data["train"].to_csv("data/susie_function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07_llama_factory_20260102/train.csv", index=False)
data["train"].to_parquet("data/susie_function_call_gpt_4o_critiqued_by_gpt_5_mini_2025_08_07_llama_factory_20260102/train.parquet")
