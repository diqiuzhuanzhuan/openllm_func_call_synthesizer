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

import json

from bespokelabs import curator
from bespokelabs.curator.log import logger
from pydantic import BaseModel, Field
from rich import pretty
from xxhash import xxh64

from openllm_func_call_synthesizer.core.formatter import QUERY_GENERATE_SYSTEM_HEADER
from openllm_func_call_synthesizer.utils import extract_format, parse_hermes_tool_calls
import ollama
from transformers import AutoTokenizer
import tqdm

class OllamaFunctionCallGenerator:
    """Generator using local Ollama instance directly."""
    
    def __init__(self, model_name: str, host: str = "http://localhost:11434", tokenizer_path: str = "Qwen/Qwen3-1.7B", generation_params: dict = None, **kwargs):
        self.client = ollama.Client(host=host)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.generation_params = generation_params or {}
        print(f"Initialized OllamaFunctionCallGenerator with model={model_name}, host={host}")

    def __call__(self, dataset):
        results = []
        print(f"Processing {len(dataset)} samples with Ollama...")
        
        for item in tqdm.tqdm(dataset):
            # Construct messages
            # Note: The original dataset item might have 'query'
            query = item.get("query", "")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            
            # Extract tools
            tools = []
            if "functions" in item:
                if isinstance(item["functions"], str):
                    try:
                        tools = json.loads(item["functions"])
                    except:
                        pass
                else:
                    tools = item["functions"]
            
            # Apply chat template
            try:
                # Some tokenizers might not support tools argument in apply_chat_template directly 
                # strictly following huggingface structure, we convert to OpenAI tools format if needed
                # But here we assume the tokenizer supports it or we construct prompt manually
                # Qwen 2.5/3 usually supports it.
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tools=tools, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Template application failed: {e}. Fallback to raw prompt.")
                prompt = query

            # Call Ollama
            try:
                # Filter out incompatible options
                # Allow more Ollama generation parameters correctly
                # Refer to: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
                valid_options = [
                     "num_ctx", "seed",  "temperature", "seed", "stop", 
                   "num_predict", "top_k", "top_p" ,"min_p"
                ]
                
                options = {k: v for k, v in self.generation_params.items() if k in valid_options}
                
                # 'backend_params' are usually for litellm connectivity (base_url), 
                # but 'generation_params' from config also contain logic params.
                # 'num_ctx' is important specifically in ollama options.
                
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    think= self.generation_params.get("think", False),
                    options=options,
                    keep_alive="5m"
                )
                
                generated_text = response.get("response", "")
                
                # Store results similar to Curator format
                item["raw_output"] = generated_text
                item["prompt"] = prompt
                
                # Try to parse tool calls (assuming Hermes/Qwen format if raw text)
                # You might need specific parsing logic depending on what the model outputs
                # For Qwen with tools, it often outputs <tool_call>...
                
                # Using the existing utility from synthesizer
                # Note: parse_hermes_tool_calls expects a message dict usually, or text?
                # Let's verify what parse_hermes_tool_calls does. 
                # In strict mode, Qwen output might just be the content.
                
                # Let's create a fake message object for parsing compatibility
                fake_message = {"content": generated_text}
                parsed_fc = parse_hermes_tool_calls(fake_message)
                
                item["function_call"] = json.dumps(parsed_fc, ensure_ascii=False) if parsed_fc else ""
                item["answer"] = json.dumps(fake_message, ensure_ascii=False)
                
                pretty.pprint("query: ")
                pretty.pprint(query)
                pretty.pprint("output: ")
                pretty.pprint(generated_text)
                pretty.pprint("ground_truth: ")
                pretty.pprint(item.get("ground_truth",""))
                results.append(item)
                
            except Exception as e:
                print(f"Error processing item: {e}")
                item["error"] = str(e)
                results.append(item)
                
        # Return as a Dataset object to match interface, or DataFrame
        import pandas as pd
        from datasets import Dataset
        df = pd.DataFrame(results)
        return Dataset.from_pandas(df)


class FunctionCallGenerator(curator.LLM):
    """A simple function calling generator."""

    return_completions_object = True

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        # return f"""
        # {input["query"]}
        # """
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input["query"].strip()}
            ]
        return messages

    def _parse_function_call(self, raw_output: dict) -> dict:
        parsed = []
        for call in raw_output:
            # Handle standard format with "function" wrapper
            if "function" in call:
                func = call["function"]
                name = func.get("name")
                args_str = func.get("arguments", "{}")
            else:
                # Handle flat format
                name = call.get("name")
                args_str = call.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args = args_str  # fallback
            parsed.append({"name": name, "arguments": args})

        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def _deduplicate_input_ls(self, input_ls):
        """
        Deduplicate input_ls based on the fields: prompt, function_call, answer.
        If all three fields are identical (ignoring 'tool_call' id等唯一id), keep only one instance.

        Due to the fact that the 'answer' field contains the 'id' of 'tool_calls',
        and each call has a different id, deduplication fails.
        Therefore, when comparing, the 'id' field under 'tool_call' must be ignored!

        """
        import json

        def norm_answer(answer):
            # the format of answer is a string of json, so we need to parse it first
            try:
                data = json.loads(answer)
            except Exception:
                return answer  # just return the original answer if it's not a valid json
            # remove the id field under tool_calls
            if isinstance(data, dict) and "tool_calls" in data and isinstance(data["tool_calls"], list):
                for tc in data["tool_calls"]:
                    if isinstance(tc, dict) and "id" in tc:
                        tc.pop("id")
            try:
                return json.dumps(data, ensure_ascii=False, sort_keys=True)
            except Exception:
                return answer

        seen = set()
        deduped = []
        for item in input_ls:
            key = (item.get("prompt", ""), item.get("function_call", ""), norm_answer(item.get("answer", "")))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def parse(self, input: dict, response) -> list:
        """Parse each choice in the response to extract the function call or the message."""
        input_ls = []
        prompt = self.prompt(input)
        print("--------------choices response------------------", response["choices"])
        for choice in response["choices"]:
            this_input = dict(input)  # make a shallow copy
            this_input["prompt"] = prompt

            message = choice.get("message", {})
            # Convert message to dict if it's an object (like litellm Message object)
            if hasattr(message, "model_dump"):
                message = message.model_dump()
            elif hasattr(message, "__dict__"):
                message = message.__dict__

            this_input["raw_output"] = message
            parsed_fc = parse_hermes_tool_calls(message)
            # 无论是否解析到tool_calls，都序列化为JSON字符串;如果是
            this_input["function_call"] = json.dumps(parsed_fc, ensure_ascii=False) if parsed_fc else ""
            this_input["answer"] = json.dumps(message, ensure_ascii=False, indent=2)

            print("----input------", this_input)

            pretty.pprint("query: ")
            pretty.pprint(this_input["query"])
            if "answer" in this_input:
                pretty.pprint("answer: ")
                pretty.pprint(this_input["answer"])
            if "function_call" in this_input:
                pretty.pprint("function_call: ")
                pretty.pprint(this_input["function_call"])
            pretty.pprint("ground_truth: ")
            pretty.pprint(this_input.get("ground_truth",""))
            input_ls.append(this_input)
        # Deduplicate before return
        if len(input_ls) > 1:
            input_ls = self._deduplicate_input_ls(input_ls)
        print(" ------------ deduped input list ------------ ", input_ls)
        return input_ls


class QueryFunc(BaseModel):
    query: str = Field(..., description="The natural language query")
    function: str = Field(..., description="The function name to call")
    dimension: str = Field(..., description="The variation dimension")
    language: str = Field(..., description="The query language")


class QueryFuncItem(BaseModel):
    item: QueryFunc = Field(..., description="The query function item")


class QueryGenerator(curator.LLM):
    """A simple query generator."""

    return_completions_object = True

    def __init__(self, model_name: str = None, backend: str = None, language: str = "English", **kwargs):
        """Initialize with optional language for generation."""
        super().__init__(model_name=model_name, backend=backend, **kwargs)
        self.language = language

    def _hash_fingerprint(self, dataset_hash: str = "", disable_cache: bool = False):
        from xxhash import xxh64

        fingerprint = super()._hash_fingerprint(dataset_hash, disable_cache)
        fingerprint = f"{fingerprint}_{xxh64(self.language.encode('utf-8')).hexdigest()}"
        logger.info(f"Curator Cache Fingerprint: {fingerprint}")
        return fingerprint

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the query."""
        seed_query = input.get("query", "")
        return QUERY_GENERATE_SYSTEM_HEADER.format(
            language=self.language, function=input["function"], seed_query=seed_query, function_name=input["function"]
        )

    def parse(self, input: dict, response) -> list[dict]:
        """Parse the response to extract the query."""

        query = extract_format(format="json", content=response["choices"][0]["message"]["content"])
        function_hash = xxh64(str(input["function"]).encode("utf-8")).hexdigest()
        # Build a list of query variation records with metadata
        output = [
            {
                "query": ele["query"],
                "dimension": ele["dimension"],
                "language": self.language,
                "function": input["function"],
                "function_hash": function_hash,
            }
            for ele in query["variations"]
        ]
        return output


class ConversionGenerator(curator.LLM):
    """A simple conversion generator."""

    return_completions_object = True

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the conversion."""
        return f"""You are a conversion generation expert. Given the user request:
        {input["user_request"]}.
        Generate a conversion that can be used to satisfy the user request.
        """

    def parse(self, input: dict, response) -> dict:
        """Parse the response to extract the conversion."""
        input["conversion"] = response["choices"][0]["message"]["content"]
        return input


if __name__ == "__main__":
    qg = QueryGenerator()
