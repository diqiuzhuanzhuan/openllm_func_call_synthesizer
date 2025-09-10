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

from typing import List, Dict, Any
from pathlib import Path
from typing import Any
import yaml
    
def convert_to_mcp_tools(function_docs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    tools = []
    for f in function_docs:
        func = f["function"]
        tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }
        tools.append(tool)
    return {"tools": tools}

    
def extract_format(format: str='json', content: str="") -> Any:
    import re
    import json
    if format == 'json':
        pattern = re.compile(r'```json\s*([\s\S]*?)\s*```')
        match = pattern.search(content)
        if match:
            try:
                data = json.loads(match.group(1))
                return data
            except json.JSONDecodeError:
                pass
        else:
            try:
                if json.loads(content):
                    return json.loads(content)
            except json.JSONDecodeError:
                pass
    return None


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML config files. Install with `pip install pyyaml`.")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    

if __name__ == "__main__":
    data = extract_format(content='```json\n{"a": 1}\n```')
    print(data)

    # 示例输入（你提供的 function_docs）
    function_docs = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Retrieves current weather for the given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and country e.g. Bogotá, Colombia"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Units the temperature will be returned in."},
                    },
                    "required": ["location", "units"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_local_time",
                "description": "Get the local time of a given location",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "required": ["location", "timezone"],
                    "properties": {
                        "location": {"type": "string", "description": "The name or coordinates of the location for which to get the local time"},
                        "timezone": {"type": "string", "description": "The timezone of the location, defaults to the location's timezone if not provided"},
                    },
                    "additionalProperties": False,
                },
            },
        },
    ]

    # 转换
    mcp_tools = convert_to_mcp_tools(function_docs)

    # 打印结果
    import json
    print(json.dumps(mcp_tools, indent=2, ensure_ascii=False))
    with open("function_docs.json", "w") as f:
        json.dump(mcp_tools, f, indent=2, ensure_ascii=False)