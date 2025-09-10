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

import hydra
from omegaconf import DictConfig, OmegaConf
from rich import pretty
from openllm_func_call_synthesizer.core.synthesizer import QueryGenerator, FunctionCallGenerator
from datasets import Dataset, concatenate_datasets
import json
from pathlib import Path


def query_generator(cfg: DictConfig):
    data_file = cfg.synthesizer.query_generation.function_docs
    if not Path(data_file).exists():
        raise FileNotFoundError(f"File {data_file} not found")
    with open(data_file, "r") as f:
        data = json.load(f)
    data = Dataset.from_list([{'function': e} for e in data['tools']])
    # Loop over configured languages to generate multilingual query variations
    languages = cfg.synthesizer.query_generation.get('languages', ['English'])
    output_datasets = []
    for language in languages:
        for name in cfg.llm.providers:
            print(f"provider: {name}, language: {language}")
            provider = cfg.llm.providers[name]
            for model in provider.models:
                print(f"model: {model}")
                # Instantiate generator with language
                qg = QueryGenerator(
                    model_name=model,
                    backend=provider.backend,
                    language=language,
                )
                # Generate records by iterating through examples and their variations
                queries = qg(dataset=data)
                ds = queries.dataset.map(lambda x: {'provider': name, 'model': model})
                
                # No need to flatten explicitly; iterate over dataset
                # Collect this provider/model/language dataset
                output_datasets.append(ds)
    # Combine and save all provider/model datasets
    combined = concatenate_datasets(output_datasets)

    # Ensure output directory exists
    out_dir = Path(cfg.synthesizer.query_generation.get('output_dir', 'data')) / cfg.synthesizer.query_generation.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save in multiple formats
    # Save JSON Lines under 'train.jsonl' so HuggingFace load_dataset can load it as the 'train' split
    combined.to_json(str(out_dir / f"train.jsonl"), orient="records", lines=True)
    combined.to_csv(str(out_dir / "output.csv"))
    combined.to_parquet(str(out_dir / "output.parquet"))
    print(f"Dataset saved to {out_dir} in train.jsonl, csv, parquet formats.")
    # You can load the JSONL with:
    from datasets import load_dataset
    load_dataset("json", data_files={"train": str(out_dir/"train.jsonl")})

def generate_function_call_dataset(cfg: DictConfig):
    # Load the function dataset
    function_dataset_path = Path(cfg.synthesizer.function_call_generation.function_dataset)
    if not Path(function_dataset_path).exists():
        raise FileNotFoundError(f"File {function_dataset_path} not found")
    from datasets import load_dataset
    dataset = load_dataset("json", data_files={"train": str(function_dataset_path/"train.jsonl")})
    hashes = dataset['train'].unique('function_hash')
    print(f"Found {len(hashes)} unique functions")
    import random
    chosen = random.sample(hashes, len(hashes))
    # filter to only those hashes
    sampled = dataset['train'].filter(lambda ex, chosen=chosen: ex["function_hash"] in chosen)
    print(sampled)
    function_call_generator = FunctionCallGenerator(
        model_name=cfg.llm.function_call.model,
        backend=cfg.llm.function_call.backend,
    )
    max_num = cfg.synthesizer.function_call_generation.max_num
    if max_num > 0:
        fcg = function_call_generator(dataset=dataset['train'].select(range(max_num)))
    else:
        fcg = function_call_generator(dataset=dataset['train'])

    # write function dataset to disk
    output_dir = Path(cfg.synthesizer.function_call_generation.output_dir)/cfg.synthesizer.function_call_generation.name
    output_dir.mkdir(parents=True, exist_ok=True)
    fcg.dataset.to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    fcg.dataset.to_csv(str(output_dir / "output.csv"))
    fcg.dataset.to_parquet(str(output_dir / "output.parquet"))
    print(f"Dataset saved to {output_dir} in train.jsonl, csv, parquet formats.")



@hydra.main(config_path="../examples/conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pretty.pprint(cfg)
    llm_cfg = cfg.llm
    synth_cfg = cfg.synthesizer
    print("llm_config: ")
    pretty.pprint(llm_cfg)
    print("synth_config: ")
    pretty.pprint(synth_cfg)
    #query_generator(cfg)
    generate_function_call_dataset(cfg)

if __name__ == "__main__":
    main()