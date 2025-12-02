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
            func = call.get("function", {})
            name = func.get("name")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = func.get("arguments", {})  # fallback
            parsed.append({"name": name, "arguments": args})

        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def parse(self, input: dict, response) -> dict:
        """Parse the response to extract the function call or the message."""
        input["prompt"] = self.prompt(input)
        if "tool_calls" in response["choices"][0]["message"] and response["choices"][0]["message"]["tool_calls"]:
            input["function_call"] = self._parse_function_call(response["choices"][0]["message"]["tool_calls"])
            input["answer"] = json.dumps(response["choices"][0]["message"], ensure_ascii=False, indent=2)

        else:
            # Handle the case where the model returns a string instead of a function call
            function_call = extract_format(format="json", content=response["choices"][0]["message"]["content"])
            if function_call is None:
                input["answer"] = json.dumps(response["choices"][0]["message"], ensure_ascii=False, indent=2)
                input["function_call"] = None
                # raise ValueError("The model did not return a valid function call.")
            else:
                input["function_call"] = json.dumps(function_call, ensure_ascii=False)
                input["answer"] = json.dumps(response["choices"][0]["message"], ensure_ascii=False, indent=2)

        pretty.pprint("query: ")
        pretty.pprint(input["query"])
        if "answer" in input:
            pretty.pprint("answer: ")
            pretty.pprint(input["answer"])
        if "function_call" in input:
            pretty.pprint("function_call: ")
            pretty.pprint(input["function_call"])

        return input


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
