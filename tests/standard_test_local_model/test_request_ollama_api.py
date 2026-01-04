import requests
import json

GENERATE_URL = "http://192.168.111.3:11434/api/generate" 
CHAT_URL = "http://192.168.111.3:11434/api/chat"

def request_ollama_generate_api(model, prompt, api_url=GENERATE_URL, temperature=0.01, tools=None):
    """
    通过Ollama API以generate形式请求模型回复
    tools: 如需 function call，传 openai 格式的 tools 列表，否则为 None
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature},
        "stream": False
    }
    if tools is not None:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama generate error: {resp.status_code} {resp.text}")
    data = resp.json()
    # Ollama generate接口的reply一般为 'response' 字段
    return data.get("response", "")

def request_ollama_chat_api(model, messages, api_url=CHAT_URL, temperature=0.01, tools=None):
    """
    通过Ollama API以chat形式请求模型回复
    messages: 列表，每个元素格式为 {"role": "system"/"user"/"assistant", "content": "..."}
    tools: 如果有 function call 需要，传 openai 格式的 tools 列表，否则为 None
    """
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False
    }
    if tools is not None:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama chat error: {resp.status_code} {resp.text}")
    data = resp.json()
    # Ollama chat接口返回"message"字段，message为dict
    return data.get("message", {})

# 示例用法
if __name__ == "__main__":
    # 如果你有工具 json，可以从文件加载
    # with open("openai_tools.json") as f:
    #     TOOLS = json.load(f)
    # TOOLS = None  # 如不测试 function call，可为 None
    with open("/data/work/SusieSu/project/openllm_func_call_synthesizer/openai_tools.json") as f:
        TOOLS = json.load(f)

    # ----------------------------------generate示例----------------------------------
    model = "function_call_rl_0104-q4_K_M"
    prompt = "我想听周杰伦的歌？"
    print("=== generate 结果 ===")
    try:
        gen_resp = request_ollama_generate_api(model, prompt, tools=TOOLS)
        print(gen_resp)
    except Exception as e:
        print("Generate接口调用失败：", e)

    # ----------------------------------chat示例----------------------------------
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "我想听周杰伦的歌？"}
    ]
    print("\n=== chat 结果 ===")
    try:
        chat_resp = request_ollama_chat_api(model, messages, tools=TOOLS)
        print(chat_resp.get("content", chat_resp))
    except Exception as e:
        print("Chat接口调用失败：", e)
