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
from xxhash import xxh64
from typing import Dict, List
from datasets import Dataset
from bespokelabs import curator
from openllm_func_call_synthesizer.utils import extract_format
from bespokelabs.curator.log import add_file_handler, logger
from rich import pretty


class FunctionCallGenerator(curator.LLM):
    """A simple function calling generator."""

    return_completions_object = True

    def _format_functions(self, functions: List[str]) -> str:
        """Format a list of function definitions into a human-readable block."""
        funcs = [json.dumps(json.loads(func), ensure_ascii=False, indent=2) for func in functions]
        return "\n\n".join(funcs)

    def prompt(self, input: Dict) -> str:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        return f"""
        You are an expert in structured function calling.
        The user request is:
        {input['query']}
        """


    def parse(self, input: Dict, response) -> Dict:
        """Parse the response to extract the function call or the message."""
        input['prompt'] = self.prompt(input)
        if "tool_calls" in response["choices"][0]["message"] and response["choices"][0]["message"]["tool_calls"]:
            input["function_call"] = json.dumps([tool_call["function"] for tool_call in response["choices"][0]["message"]["tool_calls"]])
            input['answer'] = response["choices"][0]["message"]

        else:
            # Handle the case where the model returns a string instead of a function call
            function_call = extract_format(format="json", content=response["choices"][0]["message"]["content"])
            if function_call is None:
                input['answer'] = response["choices"][0]["message"]
                input['function_call'] = None
                #raise ValueError("The model did not return a valid function call.")
            else:
                input['function_call'] = json.dumps(function_call, ensure_ascii=False)
                input['answer'] = response["choices"][0]["message"]


        pretty.pprint(input['query'])
        if "answer" in input:
            pretty.pprint(input['answer'])
        if "function_call" in input:
            pretty.pprint(input['function_call'])

        return input

from pydantic import BaseModel, Field

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
    
    def __init__(self, model_name: str = None, backend: str = None, language: str = "English", backend_params: Dict = {}):
        """Initialize with optional language for generation."""
        backend_params.update({"require_all_responses": False})
        super().__init__(model_name=model_name, backend=backend, backend_params=backend_params)
        self.language = language

    def _hash_fingerprint(self, dataset_hash: str = "", disable_cache: bool = False):
        from xxhash import xxh64
        fingerprint = super()._hash_fingerprint(dataset_hash, disable_cache)
        fingerprint = f"{fingerprint}_{xxh64(self.language.encode('utf-8')).hexdigest()}"
        logger.info(f"Curator Cache Fingerprint: {fingerprint}")
        return fingerprint

    def prompt(self, input: Dict) -> str:
        """The prompt is used to generate the query."""
        print(input)
        return f"""You are a query expansion system generating queries in {self.language}.
        Your task is to generate diverse natural language user queries that would reliably trigger the same function.

        ### Input
        - Language: {self.language}
        - Seed Query (optional): {input.get('query', None)}
        - Function Information: {input['function']}

        ### Task
        - All queries must be generated in **{self.language}** only.
        - Do not mix other languages.
        1. If a Seed Query is provided:
        - Produce **15–20 query variations** that are semantically aligned with the seed query and consistent with the given function.
        - Distribute the variations across the following dimensions (at least 3 per dimension, and avoid near-duplicates):

            **Linguistic Variations**
            - Synonyms and paraphrases
            - Different sentence structures
            - Formal vs. informal tone
            - Different question forms (what / how / can / could)

            **Specificity Levels**
            - More specific (adding details or constraints)
            - More general (removing details)
            - Different emphasis on parameters
            - Implicit vs. explicit parameter mentions

            **User Personas**
            - Expert / technical phrasing
            - Beginner / casual phrasing
            - Business-professional tone
            - Urgent or time-sensitive style

            **Intent Variations**
            - Direct command
            - Polite request
            - Problem description
            - Goal-oriented statement

            **Contextual Scenarios**
            - Different real-world use cases
            - Situational context changes
            - Follow-up or clarification query

        2. If NO Seed Query is provided:
        - Generate **15–20 plausible seed queries** directly from the function description and input schema.
        - Cover the same five dimensions above to ensure diversity.

        ### Output Format
        - Number each variation consecutively (e.g., 1, 2, 3, …).
        - Present only the rephrased queries (no explanations).
        - Ensure every query is in {self.language}, not in any other language.
        - Ensure overall count is **15–20 unique queries**.
        - Use JSON format exactly like this:

        ```json
        {{
        "variations": [
            {{
            "id": 1,
            "dimension": "Linguistic",
            "query": "..."
            }},
            {{
            "id": 2,
            "dimension": "Specificity",
            "query": "..."
            }}
        ]
        }}
        ```
        """


    def parse(self, input: Dict, response) -> List[Dict]:
        """Parse the response to extract the query."""

        query = extract_format(format='json', content=response["choices"][0]["message"]["content"])
        function_hash = xxh64(str(input['function']).encode('utf-8')).hexdigest()
        # Build a list of query variation records with metadata
        output = [
            {
                'query': ele['query'],
                'dimension': ele['dimension'],
                'language': self.language,
                'function': input['function'],
                'function_hash': function_hash,
            }
            for ele in query['variations']
        ]
        return output

        
class ConversionGenerator(curator.LLM):
    """A simple conversion generator."""

    return_completions_object = True

    def prompt(self, input: Dict) -> str:
        """The prompt is used to generate the conversion."""
        return f"""You are a conversion generation expert. Given the user request:
        {input['user_request']}.
        Generate a conversion that can be used to satisfy the user request.
        """

    def parse(self, input: Dict, response) -> Dict:
        """Parse the response to extract the conversion."""
        input["conversion"] = response["choices"][0]["message"]["content"]
        return input


if __name__ == "__main__":
    
    qg = QueryGenerator()
