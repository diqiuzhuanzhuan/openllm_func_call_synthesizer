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
import re
from bespokelabs import curator

from openllm_func_call_synthesizer.core.formatter import (
    CRITIC_FUNCTION_CALL_SYSTEM_HEADER,
)
from openllm_func_call_synthesizer.utils import extract_format

##意图识别模型评分
intent_system_prompt_old = """You are an evaluation assistant for an intent recognition and slot extraction model.

Your task is to **evaluate the quality of the model’s output** based on:
1. The given instruction (which defines all intents and slots),
2. The user’s input,
3. The model’s output (intent + slots in JSON format).

---

### Evaluation Criteria

You should score the model’s output from **0 to 10**, where:

- **10** → Perfect: The intent is correctly identified, \
    and all required slots are correctly extracted with accurate values.
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
intent_system_prompt="""You are an evaluation assistant for an intent recognition and slot extraction model.

Your task is to evaluate the quality of the model’s output based on:
1. The given instruction (which defines all valid intents and slots),
2. The user’s input,
3. The model’s predicted intent and slots.

IMPORTANT:
- You are evaluating the **semantic correctness** of intent recognition and slot extraction.
- Do NOT infer or assume user intent beyond what is explicitly stated.
- Do NOT penalize conservative predictions.

---

### How to Interpret the Model Output

- The logical evaluation target is a **single intent object** with the structure:
  {
    "intent": "<intent_name>",
    "slots": { ... }
  }

- The model output may be wrapped in different formats:
  - A single JSON object
  - A JSON list containing exactly ONE such object

- If the output is a list with one intent object, evaluate that object normally.
- Only treat the output as invalid if:
  - It cannot be parsed into an intent + slots structure at all, OR
  - The required fields (`intent`, `slots`) are missing after parsing.

---

### Evaluation Criteria (Score 0–10)

- **10** → Perfect  
  Intent is correct, and all required slots are correctly extracted with accurate values.

- **8–9** → Very good  
  Intent is correct; only minor slot naming or formatting issues.

- **6–7** → Acceptable  
  Intent is correct, but some required slots are missing or partially incorrect.

- **3–5** → Poor  
  Intent is likely incorrect, or there are multiple slot extraction errors.

- **1–2** → Very poor  
  Intent is clearly wrong or irrelevant to the user input.

- **0** → Invalid  
  Output cannot be parsed into a valid intent + slots structure.

---

### Special Rules for `unknown` Intent (CRITICAL)

- The intent `"unknown"` is a **valid and correct prediction** when:
  - The user input is vague, ambiguous, informational, or non-actionable, OR
  - The input does not clearly specify a single executable intent, OR
  - Required parameters for any defined intent are missing.

- You MUST NOT penalize `"unknown"` unless:
  - The user input **explicitly and unambiguously** expresses a single defined intent,
  - AND the intent is directly executable without guessing missing information.

- If `"unknown"` is reasonable given the user input, it may receive **high scores (8–10)**.

---

### Slot Validation Rules

- Required slots must be present to receive full score.
- Missing required slots should reduce the score proportionally.
- For `music_play_control`:
  - At least one of `title` or `source` MUST be present.
  - Missing both makes the prediction invalid for that intent.

---

### Output Format

You MUST output **only one JSON object**:

```json
{
  "score": <integer from 0 to 10>,
  "reason": "<brief explanation focusing on intent clarity and slot correctness>"
}
"""

class Critic(curator.LLM):
    """A simple critic for any tasks."""

    return_completions_object = True

    #    def _hash_fingerprint(self, dataset_hash: str = "", disable_cache: bool = False):
    #        return super()._hash_fingerprint("", disable_cache)

    def __init__(
        self,
        model_name,
        response_format=None,
        batch=False,
        backend=None,
        generation_params=None,
        backend_params=None,
        system_prompt=None,
        query_field="query",
        task_prompt_field="task_prompt",
        label_field="label",
        functions_field="functions",
        response_field="response",
        purpose="function_call",
        **kwargs,
    ):
        super().__init__(
            model_name, response_format, batch, backend, generation_params, backend_params, system_prompt, **kwargs
        )
        self.query_field = query_field
        self.task_prompt_field = task_prompt_field
        self.label_field = label_field
        self.functions_field = functions_field
        self.response_field = response_field
        self.purpose=purpose

    def prompt(self, input: dict) -> dict:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        #增加意图识别模型的测评

        if self.purpose=="function_call":
            system_prompt = input.get("system_prompt", CRITIC_FUNCTION_CALL_SYSTEM_HEADER)
        elif self.purpose=="mcp_intent":
            system_prompt= intent_system_prompt
        else:
            raise ValueError("purpose is wrong,you must provide \"function_call\"or \"mcp_intent\"")
        
        
        task_prompt = input.get(self.task_prompt_field, "")
        if not task_prompt:
            raise ValueError("task_prompt is required")
        query = input.get(self.query_field, "")
        if not query:
            raise ValueError("query is required")
        functions = input.get(self.functions_field, "")
        if not functions:
            raise ValueError("functions is required")
        if isinstance(functions, str):
            functions = json.dumps(json.loads(functions), ensure_ascii=False, indent=2)
        label = input.get(self.label_field, "")
        answer = input.get(self.response_field, "")
        if not label and not answer:
            raise ValueError("either label or answer is required")
        
        answer_data = json.loads(answer)
        content = answer_data.get("content")
        if content:
            answer_filter_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        else:
            answer_filter_think = answer
            
        print("===========answer_filter_think============\n", answer_filter_think)
        model_output = label if label else answer_filter_think
        user_prompt = f"""
        The given instruction is {task_prompt}.
        The available functions are: {functions}
        The model output is :{model_output}
        """
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def parse(self, input: dict, response) -> dict:
        """Parse the response to extract the function call or the message."""
        input["prompt"] = self.prompt(input)
        json_extract = extract_format(format="json", content=response["choices"][0]["message"]["content"])
        if json_extract is None:
            input["score"] = 0
            input["reason"] = "Failed to parse critic response as JSON"
        else:
            score, reason = json_extract.get("score", 0), json_extract.get("reason", "No reason provided")
            input["score"] = score
            input["reason"] = reason
        return input
