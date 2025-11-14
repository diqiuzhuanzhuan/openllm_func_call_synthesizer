# UGREEN License
#
# Copyright (c) 2025 UGREEN. All Rights Reserved.
#
# This software and associated documentation files (the "Software") are
# the proprietary information of UGREEN.
#
# The Software is provided solely for internal use within UGREEN
# and may not be copied, modified, distributed, or disclosed to any
# third party without prior written consent from UGREEN.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from typing import Dict, List
from datasets import load_dataset, Dataset
from datasets import DatasetDict

def convert_to_dataset(data: List[Dict]) -> Dataset:
    """Convert a list of dictionaries to a Hugging Face Dataset.

    Args:
        data: A list of dictionaries.

    Returns:
        A Hugging Face Dataset.
    """
    load_dataset("json", data_files={"train": data})
    
    return Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})

def format_openai(example: Dict, system_prompt: str) -> Dict:
    """Format an example for OpenAI.

    Args:
        example: A dictionary containing the example.

    Returns:
        A dictionary containing the formatted example.
    """
    print(type(example["functions"]))
    print("helo")
    import json
    return {
        "messages": [
            {"role": "system", "content": system_prompt, "tool_calls": []},
            {"role": "user", "content": example["query"], "tool_calls": []},
            {"role": example["answer"]["role"], "content": example["answer"]["content"] or "", "tool_calls": example["answer"]["tool_calls"] or []},
        ],
        "tools": [json.loads(json_str) for json_str in example["functions"]],
    }

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    file = Path(__file__).parent / "train.jsonl"
    dataset = load_dataset("json", data_files=file.as_posix())
    openai_format_dataset = dataset.map(
        format_openai, 
        fn_kwargs={"system_prompt": "You are a helpful assistant."},
        ).remove_columns(dataset['train'].column_names)
    openai_format_dataset['train'].to_json("openai_format_dataset.jsonl", orient="records", lines=True)
    print(openai_format_dataset)
    """
    in LLaMA_FACTORY, DatasetInfo should be like this:

    "openai_format_dataset": {
        "file_name": "openai_format_dataset.jsonl",
        "formatting": "openai",
        "columns": {
        "messages": "messages",
        "tools": "tools"
        },
        "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
        "function_tag": "tool_calls"
        }
    }
    """

    
