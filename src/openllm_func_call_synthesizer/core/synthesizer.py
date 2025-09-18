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
from typing import Optional



class FunctionCallGenerator(curator.LLM):
    """A simple function calling generator."""

    return_completions_object = True

    def __init__(
        self, 
        prompt_type:str = "normal", 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prompt_type = prompt_type  

    def _format_functions(self, functions: List[str], prompt_type:str = "normal") -> str:
        """Format a list of function definitions into a human-readable block."""
        if prompt_type == "normal":
            funcs = [json.dumps(json.loads(func), ensure_ascii=False, indent=2) for func in functions]
            return "\n\n".join(funcs)
        elif prompt_type == "react":
            tool_descs = []
            for function in functions:
                json_function = json.loads(function)
                tool_desc = (
                    f"> Tool Name: {json_function['name']}\n"
                    f"Tool Description: {json_function['description']}\n"
                    f"Tool Args: {json.dumps(json_function['input_schema'], ensure_ascii=False)}\n"
                )
                tool_descs.append(tool_desc)
            return tool_descs
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    

    def react_prompt(self, input: Dict) -> str:
        """React-style prompt that uses tool descriptions and names."""
        # Format tool descriptions and extract tool names correctly
        funcs = [json.loads(func) for func in input.get('functions', [])]
        tool_descs = self._format_functions(input.get('functions', []), prompt_type="react")
        tools_name = [f.get('name') for f in funcs]
        return f"""
        You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

        ## Tools

        You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
        This may require breaking the task into subtasks and using different tools to complete each subtask.

        You have access to the following tools:
        {"\n".join(tool_descs)}
        Below is the user's request:
        {input['query']}

        ## Output Format

        Please answer in the same language as the question and use the following format:

        ```
        Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
        Action: tool name (one of {", ".join(tools_name)}) if using a tool.
        Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
        ```

        Please ALWAYS start with a Thought.

        NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

        """


    def prompt(self, input: Dict) -> str:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        if self.prompt_type == "react":
            return self.react_prompt(input)
        functions_block = self._format_functions(input.get('functions', []))
        return f"""
        You are an expert in structured function calling.

        The user request is:
        {input['query']}

        You have access to the following functions:
        {functions_block}

        Your task:
        - Choose the most appropriate function to fulfill the request.
        - Include all required parameters; use placeholders if not specified.
        - Return ONLY a JSON object with `name` and `arguments`.
        - If no function applies, return an empty JSON object: {{}}

        Desired format:
        {{
            "name": "<function_name>",
            "arguments": {{
                "param1": "value1",
                "param2": "value2"
            }}
        }}
        """


    def parse(self, input: Dict, response) -> Dict:
        """Parse the response to extract the function call or the message."""
        if self.prompt_type == "react":
            input['prompt'] = self.react_prompt(input)
        else:
            input['prompt'] = self.prompt(input)
        if "tool_calls" in response["choices"][0]["message"] and response["choices"][0]["message"]["tool_calls"]:
            input["function_call"] = str([tool_call["function"] for tool_call in response["choices"][0]["message"]["tool_calls"]])
        else:
            # Handle the case where the model returns a string instead of a function call
            if self.prompt_type == "react":

                def _extract_function_call(content: str) -> Optional[Dict]:
                    """Extract the function call from the content."""

                    # 提取 Action 名称
                    import re
                    action_match = re.search(r"Action:\s*(\w+)", content)
                    action = action_match.group(1) if action_match else None

                    # 提取 Action Input
                    input_match = re.search(r"Action Input:\s*(\{.*\})", content, re.DOTALL)
                    action_input = json.loads(input_match.group(1)) if input_match else None
                    if action:
                        return {"name": action, "arguments": action_input} if action_input else {"name": action}
                function_call = _extract_function_call(response["choices"][0]["message"]["content"]) 
            else:
                function_call = extract_format(format="json", content=response["choices"][0]["message"]["content"])
            if function_call is None:
                raise ValueError("The model did not return a valid function call.")
            else:
                input['function_call'] = json.dumps(function_call)

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
    
    def __init__(self, model_name: str = None, backend: str = None, language: str = "English"):
        """Initialize with optional language for generation."""
        super().__init__(model_name=model_name, backend=backend, backend_params={"require_all_responses": False})
        self.language = language

    def _hash_fingerprint(self, dataset_hash: str = "", disable_cache: bool = False):
        from xxhash import xxh64
        fingerprint = super()._hash_fingerprint(dataset_hash, disable_cache)
        fingerprint = f"{fingerprint}_{xxh64(self.language.encode('utf-8')).hexdigest()}"
        logger.info(f"Curator Cache Fingerprint: {fingerprint}")
        return fingerprint

    def prompt(self, input: Dict) -> str:
        """The prompt is used to generate the query."""
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
