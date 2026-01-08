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
from openllm_func_call_synthesizer.utils import extract_format


class FunctionCallGenerator(curator.LLM):
    """A simple function calling generator."""

    return_completions_object = True

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        return f"""
        {input["query"]}
        """

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

            if "tool_calls" in message and message["tool_calls"]:
                # Flatten tool_calls to match user requirement (no id, no type)
                flat_tool_calls = []
                for tc in message["tool_calls"]:
                    if "function" in tc:
                        flat_tool_calls.append({
                            "name": tc["function"].get("name"),
                            "arguments": tc["function"].get("arguments")
                        })
                    else:
                        flat_tool_calls.append({
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments")
                        })
                message["tool_calls"] = flat_tool_calls

                this_input["function_call"] = self._parse_function_call(message["tool_calls"])
                this_input["answer"] = json.dumps(message, ensure_ascii=False, indent=2)
            else:
                # Handle the case where the model returns a string instead of a function call
                content = message.get("content", "")
                
                # Try to extract tool_call from <tool_call> tags
                import re
                tool_calls_list = []
                tool_call_pattern = re.compile(r"<tool_call>\s*(.+?)\s*</tool_call>", re.DOTALL)
                match = tool_call_pattern.search(content)
                
                if match:
                    tool_call_block = match.group(1)
                    # Try to parse line by line
                    for line in tool_call_block.strip().splitlines():
                        line = line.strip()
                        if line:
                            try:
                                tool_calls_list.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                    
                    # If list is still empty, try parsing the whole block
                    if not tool_calls_list:
                        try:
                            tool_calls_list.append(json.loads(tool_call_block))
                        except json.JSONDecodeError:
                            pass

                    # Remove <tool_call> block from content
                    content = tool_call_pattern.sub("", content).strip()
                    
                    # Update message content
                    message["content"] = content

                    # Populate message["tool_calls"]
                    if tool_calls_list:
                        import uuid
                        new_tool_calls = []
                        for tc in tool_calls_list:
                            new_tool_calls.append({
                             
                                    "name": tc.get("name"),
                                    "arguments": tc.get("arguments")
                            
                            })
                        message["tool_calls"] = new_tool_calls
                    
                    # Set function_call
                    this_input["function_call"] = json.dumps(tool_calls_list, ensure_ascii=False)
                    this_input["answer"] = json.dumps(message, ensure_ascii=False, indent=2)
                else:
                    # Fallback to extract_format
                    function_call = extract_format(format="json", content=content)
                    if function_call is None:
                        this_input["answer"] = json.dumps(message, ensure_ascii=False, indent=2)
                        this_input["function_call"] = None
                    else:
                        this_input["function_call"] = json.dumps(function_call, ensure_ascii=False)
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
