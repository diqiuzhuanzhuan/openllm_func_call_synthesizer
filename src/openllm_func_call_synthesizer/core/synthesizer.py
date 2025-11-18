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

from openllm_func_call_synthesizer.utils import extract_format


class FunctionCallGenerator(curator.LLM):
    """A simple function calling generator."""

    return_completions_object = True

    def _format_functions(self, functions: list[str]) -> str:
        """Format a list of function definitions into a human-readable block."""
        funcs = [json.dumps(json.loads(func), ensure_ascii=False, indent=2) for func in functions]
        return "\n\n".join(funcs)

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        return f"""
        You are an expert in structured function calling.
        The user request is:
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
            input["answer"] = response["choices"][0]["message"]

        else:
            # Handle the case where the model returns a string instead of a function call
            function_call = extract_format(format="json", content=response["choices"][0]["message"]["content"])
            if function_call is None:
                input["answer"] = response["choices"][0]["message"]
                input["function_call"] = None
                # raise ValueError("The model did not return a valid function call.")
            else:
                input["function_call"] = json.dumps(function_call, ensure_ascii=False)
                input["answer"] = response["choices"][0]["message"]

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
        print(input)
        return f"""You are a query expansion system generating queries in {self.language}.
Your goal is to produce diverse, natural, and human-like user queries that would reliably trigger the same function.

---

### 🎯 Objective
Generate **15–20** realistic, conversational, \
    and semantically equivalent user queries in **{self.language}** \
        that could all be interpreted as invoking the same function.

---

### 🧩 Input
- Language: {self.language}
- Seed Query (optional): {input.get("query", None)}
- Function Information: {input["function"]}

---

### 🧠 Task Details
All output must:
- Be written **only in {self.language}**
- Sound **natural, fluent, and human-like**
- Include **oral or colloquial expressions**, as if spoken or typed casually in chat
- Avoid near-duplicates — every query should have a unique tone or perspective

---

### 1️⃣ If a Seed Query is provided
Produce **15–20 query variations** that preserve meaning and intent while covering the following diversity dimensions:

#### **Linguistic Variations**
- Synonyms, paraphrases, or alternative phrasings
- Different sentence structures
- Formal vs. informal tone
- Various question forms (“what / how / can / could / is it possible to…”)
- **Add oral-style expressions**, e.g.:
  - “Hey, can you help me…?”
  - “Hmm, I’m trying to figure out how to…”
  - “Do you happen to know…?”
  - “Could you show me how to…?”
  - “I’m wondering if I can…”

#### **Specificity Levels**
- Broaden or narrow the level of detail
- Emphasize or omit certain parameters
- Switch between implicit and explicit parameter mentions

#### **User Personas**
- Expert phrasing
- Beginner-friendly or casual tone
- Business-professional or polite requests
- Time-sensitive or urgent style (“ASAP”, “right now”, etc.)

#### **Intent Variations**
- Direct commands
- Polite requests
- Descriptive or problem-reporting phrasing
- Goal- or outcome-oriented wording

#### **Contextual Scenarios**
- Place the query in different real-world contexts
- Frame as follow-ups (“Wait, what about…?”, “Actually, can you also…”)
- Add subtle scenario cues (e.g., “on my phone”, “for a client”, “before tomorrow”)

---

### 2️⃣ If NO Seed Query is provided
Generate **15–20 plausible seed queries** directly from the function’s description and schema.
Ensure diversity across all five dimensions above, with a mix of written and conversational tones.

---

### 🧾 Output Format
Return only the rephrased queries — **no explanations or commentary**.
Use strict JSON format like this:

```json
{{
  "variations": [
    {{
      "id": 1,
      "dimension": "Linguistic",
      "query": "Hey, can you help me convert a PDF to Word?"
    }},
    {{
      "id": 2,
      "dimension": "Specificity",
      "query": "How do I turn my PDF into a Word document quickly?"
    }}
  ]
}}
```
"""

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
