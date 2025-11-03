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

import json
from typing import Dict, List
from datasets import Dataset
from bespokelabs import curator
from openllm_func_call_synthesizer.utils import extract_format
from bespokelabs.curator.log import add_file_handler, logger
from rich import pretty
from openllm_func_call_synthesizer.core.formatter import (
    CRITIC_SYSTEM_HEADER,
    CRITIC_FUNCTION_CALL_SYSTEM_HEADER,
)




system_prompt = """You are an evaluation assistant for an intent recognition and slot extraction model.

Your task is to **evaluate the quality of the model’s output** based on:
1. The given instruction (which defines all intents and slots),
2. The user’s input,
3. The model’s output (intent + slots in JSON format).

---

### Evaluation Criteria

You should score the model’s output from **0 to 10**, where:

- **10** → Perfect: The intent is correctly identified, and all required slots are correctly extracted with accurate values.
- **8–9** → Very good: Minor slot naming or value formatting issues, but intent and meaning are correct.
- **6–7** → Acceptable: Intent correct, but missing or partially wrong slot(s).
- **3–5** → Poor: Intent likely incorrect, or multiple slot extraction errors.
- **1–2** → Very poor: Intent completely wrong or irrelevant.
- **0** → Invalid: Output not in JSON format or completely meaningless.

---

### Special Notes
- If the intent is `"unknown"` but the user’s input clearly matches one of the defined intents → score ≤ 3.
- If the JSON format is invalid or fields are missing (e.g., missing `"intent"` or `"slots"`) → score = 0.
- Required slots must be filled; if missing, subtract points accordingly.
- For `music_play_control`, note that **either `title` or `source` must be present** — missing both is invalid.

---

### Output Format

You must output **only a JSON object** with the following structure:

```json
{
  "score": <integer from 0 to 10>,
  "reason": "<brief explanation of why this score was given>"
}
"""


class Critic(curator.LLM):
    """A simple critic for any tasks."""

    return_completions_object = True
    
    def __init__(
        self, 
        model_name, 
        response_format = None, 
        batch = False, 
        backend = None, 
        generation_params = None, 
        backend_params = None, 
        system_prompt = None,
        query_field = "query",
        task_prompt_field = "task_prompt",
        label_field = "label",    
        functions_field = "functions",
        response_field = "response",
        ):
        super().__init__(model_name, response_format, batch, backend, generation_params, backend_params, system_prompt)
        self.query_field = query_field
        self.task_prompt_field = task_prompt_field
        self.label_field = label_field
        self.functions_field = functions_field
        self.response_field = response_field


    def prompt(self, input: Dict) -> Dict:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        system_prompt = input.get("system_prompt", CRITIC_FUNCTION_CALL_SYSTEM_HEADER)
        task_prompt = input.get(self.task_prompt_field, "")
        if not task_prompt:
            raise ValueError("task_prompt is required")
        query = input.get(self.query_field, "")
        if not query:
            raise ValueError("query is required")
        functions = input.get(self.functions_field, "")
        if not functions:
            raise ValueError("functions is required")
        label = input.get(self.label_field, "")
        answer = input.get(self.response_field, "")
        if not label and not answer:
            raise ValueError("either label or answer is required")
        model_output = label if label else answer['content']

        user_prompt = f"""
        The given instruction is {task_prompt}.
        The available functions are: {functions}
        The model output is :{model_output}
        """
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]


    def parse(self, input: Dict, response) -> Dict:
        """Parse the response to extract the function call or the message."""
        input['prompt'] = self.prompt(input)
        json_extract = extract_format(format="json", content=response["choices"][0]["message"]["content"])
        score, reason = json_extract["score"], json_extract["reason"]
        input['score'] = score
        input['reason'] = reason
        return input

