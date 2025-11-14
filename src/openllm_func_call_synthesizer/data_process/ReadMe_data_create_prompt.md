# react  data prompt:
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
        {chr(10).join(tool_descs)}
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

# v0 prompt:

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
                                               ##-------------------------------------------------------------------------------------------------##
