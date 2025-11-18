# MIT License
#
# Copyright (c) 2025, Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import asyncio
import json
from pathlib import Path

import hydra
from datasets import concatenate_datasets, load_dataset
from fastmcp import Client
from omegaconf import DictConfig, OmegaConf
from rich import pretty

from openllm_func_call_synthesizer.core.critic import Critic
from openllm_func_call_synthesizer.core.synthesizer import (
    FunctionCallGenerator,
    QueryGenerator,
)
from openllm_func_call_synthesizer.utils import (
    convert_to_openai_tools,
    pick_unique,
    tool_format_convert,
)


async def get_mcp_tools(mcp_cfg: dict) -> list[dict]:
    """Get tools from MCP server."""
    mcp_cfg = mcp_cfg
    client = Client(**mcp_cfg)
    async with client:
        tools = await client.list_tools()
    return tools


def generate_query_dataset(cfg: DictConfig, function_docs: list[dict]):
    data_file = cfg.synthesizer.query_generation.function_docs
    query_generator_cfg = cfg.synthesizer.query_generation
    OmegaConf.set_struct(query_generator_cfg, False)
    if not Path(data_file).exists():
        raise FileNotFoundError(f"File {data_file} not found")
    # 从 function_docs.json 读取包含 query 的工具定义
    with open(data_file) as f:
        json_tools_data = json.load(f)

    # 创建一个字典，以工具名称为 key，query 为 value
    tool_queries_map = {}
    for tool in json_tools_data.get("tools", []):
        tool_name = tool.get("name")
        if tool_name and "query" in tool:
            tool_queries_map[tool_name] = tool["query"]

    print("---------tool_queries_map------------", tool_queries_map)

    dataset_records = []
    for tool in function_docs["tools"]:
        # 获取工具名称
        function_info = tool.get("function", {})
        tool_name = function_info.get("name")

        # 从 map 中获取对应的 query
        seed_queries = tool_queries_map.get(tool_name, None)
        print(f"---------tool: {tool_name}, seed_queries: {seed_queries}------------")

        tool_copy = dict(tool)
        function_repr = json.dumps(tool_copy, ensure_ascii=False, indent=2)

        if isinstance(seed_queries, list):
            print("---------is list------------", seed_queries)
            for seed in seed_queries:
                print("---------seed------------", seed)
                dataset_records.append({"function": function_repr, "query": seed})
        elif isinstance(seed_queries, str) and seed_queries.strip():
            dataset_records.append({"function": function_repr, "query": seed_queries})
        else:
            dataset_records.append({"function": function_repr})
    data = dataset_records
    print("#################### data ####################", data)
    pretty.pprint(data)
    # Loop over configured languages to generate multilingual query variations
    languages = query_generator_cfg.get("languages", ["English"])
    output_datasets = []
    for language in languages:
        for name in query_generator_cfg.providers:
            print(f"provider: {name}, language: {language}")
            provider = query_generator_cfg.providers[name]
            for model in provider.models:
                print(f"model: {model}")
                # Instantiate generator with language
                qg = QueryGenerator(
                    model_name=model,
                    language=language,
                    backend=provider.backend,
                    backend_params=provider.backend_params,
                )
                # Generate records by iterating through examples and their variations
                queries = qg(dataset=data)
                ds = queries.dataset.map(lambda x: {"provider": name, "model": model})

                # No need to flatten explicitly; iterate over dataset
                # Collect this provider/model/language dataset
                output_datasets.append(ds)
    # Combine and save all provider/model datasets
    combined = concatenate_datasets(output_datasets)

    # Ensure output directory exists
    out_dir = Path(query_generator_cfg.get("output_dir", "data")) / cfg.synthesizer.query_generation.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save in multiple formats
    # Save JSON Lines under 'train.jsonl' so HuggingFace load_dataset can load it as the 'train' split
    combined.to_json(str(out_dir / "train.jsonl"), orient="records", lines=True)
    combined.to_csv(str(out_dir / "output.csv"))
    combined.to_parquet(str(out_dir / "output.parquet"))
    print(f"Dataset saved to {out_dir} in train.jsonl, csv, parquet formats.")
    # You can load the JSONL with:
    load_dataset("json", data_files={"train": str(out_dir / "train.jsonl")})


def generate_function_call_dataset(cfg: DictConfig, mcp_tools: list[dict]):
    # Load the function dataset
    function_call_cfg = cfg.synthesizer.function_call_generation
    function_dataset_path = Path(function_call_cfg.function_dataset)
    if not Path(function_dataset_path).exists():
        raise FileNotFoundError(f"File {function_dataset_path} not found")
    dataset = load_dataset("json", data_files={"train": str(function_dataset_path / "train.jsonl")})
    hashes = dataset["train"].unique("function_hash")
    print(f"Found {len(hashes)} unique functions")
    import random

    chosen = random.sample(hashes, len(hashes))
    # filter to only those hashes, select all currently because the number of functions is small
    sampled = pick_unique(dataset["train"], "function_hash", len(chosen))

    # sampled = dataset['train'].filter(lambda ex, chosen=chosen: ex["function_hash"] in chosen)
    print(f"sampled {len(sampled)} functions: {sampled}")
    functions = sampled["function"]
    fc_kwargs = OmegaConf.to_container(function_call_cfg.provider, resolve=True)
    
    function_docs = tool_format_convert(mcp_tools, fc_kwargs["model_name"])
    function_call_generator = FunctionCallGenerator(
        **fc_kwargs,
        generation_params={"tools": function_docs["tools"]},
    )
    max_num = function_call_cfg.max_num
    if max_num > 0:
        dataset = dataset["train"].select(range(max_num))
    else:
        dataset = dataset["train"]
    dataset = dataset.map(lambda x: {"functions": functions})
    fcg = function_call_generator(dataset=dataset)

    # write function dataset to disk
    output_dir = Path(function_call_cfg.output_dir) / function_call_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    fcg.dataset.to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    fcg.dataset.to_csv(str(output_dir / "output.csv"))
    fcg.dataset.to_parquet(str(output_dir / "output.parquet"))
    print(f"Dataset saved to {output_dir} in train.jsonl, csv, parquet formats.")


def critic_function_call_dataset(cfg: DictConfig):
    critic_cfg = cfg.synthesizer.critic
    function_call_dataset_path = Path(critic_cfg.function_call_dataset)
    if not function_call_dataset_path.exists():
        raise FileNotFoundError(f"File {function_call_dataset_path} not found")
    dataset = load_dataset("json", data_files={"train": str(function_call_dataset_path / "train.jsonl")})
    cg_args = OmegaConf.to_container(cfg.synthesizer.critic.provider, resolve=True)
    cg_args["query_field"] = critic_cfg.query_field
    cg_args["task_prompt_field"] = critic_cfg.task_prompt_field
    cg_args["label_field"] = critic_cfg.label_field
    cg_args["functions_field"] = critic_cfg.functions_field
    cg_args["response_field"] = critic_cfg.response_field
    critic_generate = Critic(**cg_args)
    max_num = cfg.synthesizer.function_call_generation.max_num
    if max_num > 0:
        dataset = dataset["train"].select(range(max_num))
    else:
        dataset = dataset["train"]
    cg = critic_generate(dataset=dataset)
    output_dir = Path(critic_cfg.output_dir) / critic_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cg.dataset.to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    cg.dataset.to_csv(str(output_dir / "output.csv"))
    cg.dataset.to_parquet(str(output_dir / "output.parquet"))
    print(f"Dataset saved to {output_dir} in train.jsonl, csv, parquet formats.")


def choose_tools(openai_format_tools):
    # target_names = ["search_photos", "create_album"]
    target_names = ["search_photos", "create_album", "get_album_list", "music_play_control", "music_search_control","music_settings_control", "video_search_control", "video_play_control", "get_system_info"]

    # 遍历整个 openai_format_tools['tools']，只保留 function name 在 target_names 内的工具
    filtered_tools = []
    for tool in openai_format_tools["tools"]:
        # 确保有function字段并且有name
        function_info = tool.get("function", {})
        func_name = function_info.get("name", None)
        if func_name in target_names:
            filtered_tools.append(tool)

    # 覆盖原 openai_format_tools，使格式保持不变，只保留符合的tools
    openai_format_tools2 = {"tools": filtered_tools}

    # print("------------openai_format_tools2------------", type(openai_format_tools2), openai_format_tools2)

    return openai_format_tools2


@hydra.main(config_path="../examples/conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pretty.pprint("loading config:")
    pretty.pprint(cfg)
    print("loading tools from MCP server:")
    loop = asyncio.get_event_loop()
    mcp_tools = loop.run_until_complete(get_mcp_tools(mcp_cfg=cfg.synthesizer.mcp_servers["ugreen_mcp"]))
    openai_format_tools = convert_to_openai_tools(mcp_tools)
    print("------------openai_format_tools------------", openai_format_tools)
    # choose part tools
    # openai_format_tools = choose_tools(openai_format_tools)
    # print("------------openai_format_tools------------", openai_format_tools)
    pretty.pprint(openai_format_tools)
    synth_cfg = cfg.synthesizer
    print("synth_config: ")
    pretty.pprint(synth_cfg)

    # generate_query_dataset(cfg, function_docs=openai_format_tools)
    # generate_function_call_dataset(cfg, mcp_tools=mcp_tools)
    critic_function_call_dataset(cfg)


if __name__ == "__main__":
    main()
