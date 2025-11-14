#!/usr/bin/env python

# # 训练数据生成

# In[2]:


import json
import sys

import pandas as pd

sys.path.append("/data0/work/SusieSu/project")
from Call_LLM_Utils.read_file_util import *

# root = '/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/data/lora_train_data/react_data/'
root = "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/data/lora_train_data/v0_data/"
input_file_name = "/data0/work/SusieSu/project/openllm_func_call_synthesizer/data/data/function_call_0919/voted_function_call_rs_0919_3models.xlsx"
KEY_PROMPT = "v0"  # v0 react


def value_counts(df, key_column):
    # 计算计数和占比
    counts = df[key_column].value_counts()
    proportions = df[key_column].value_counts(normalize=True)

    # 合并为 DataFrame
    result = pd.DataFrame(
        {
            "count": counts,
            "proportion": proportions,
        }
    ).reset_index()

    # 重命名列
    result.columns = [key_column, "count", "proportion"]

    # 可选：保留两位小数
    result["proportion"] = result["proportion"].apply(lambda x: f"{x * 100:.2f}%")

    total_count = result["count"].sum()

    # 添加总计行
    total_row = pd.DataFrame({key_column: ["Total"], "count": [total_count], "proportion": ["100%"]})

    # 合并结果
    result = pd.concat([result, total_row], ignore_index=True)

    # print(result)
    return result


def read_excel(path, sheet_name=None):
    if sheet_name != None:
        df = pd.read_excel(path)
        print(df.shape)
        print(df.columns)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)
        print(df.shape)
        print(df.columns)

    return df


import os


def read_json_file(file_path):
    """
    读取 JSON 文件
    :param file_path: JSON 文件路径
    :return: 解析后的字典/列表数据
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        print("JSON 文件读取成功")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON 文件格式错误: {e}")
    except Exception as e:
        print(f"读取 JSON 文件时出错: {e}")


# In[ ]:

df = pd.read_excel(input_file_name)
df.shape, df.columns


df


# In[5]:


df.iloc[111]["voted_function_call"]


# In[6]:


eval(df.iloc[111]["model_function_calls"])


# In[7]:


df = df.drop_duplicates(subset="query", inplace=False)
df.shape


# In[ ]:

system_prompt_v0 = """You are an expert in structured function calling.

You have access to the following functions:
{
  "name": "create_album",
  "description": "Create a new photo album",
  "input_schema": {
    "type": "object",
    "properties": {
      "album_name": {
        "type": "string",
        "description": "The name of the album to be created"
      },
      "album_type": {
        "type": "string",
        "enum": [
          "normal"
        ],
        "description": "The type of the album, default by normal"
      }
    },
    "required": [
      "album_name",
      "album_type"
    ],
    "additionalProperties": false
  }
}

{
  "name": "search_photos",
  "description": "Search for photos or images",
  "input_schema": {
    "type": "object",
    "properties": {
      "keyword": {
        "type": "string",
        "description": "The search keyword for photos or images. It can be descriptive text or a file name, e.g., 'photos taken last August' or 'dog on the grass'."
      }
    },
    "required": [
      "keyword"
    ],
    "additionalProperties": false
  }
}

{
  "name": "get_album_list",
  "description": "Retrieve the list of photo albums, including regular albums, people albums, baby albums, conditional albums, and object recognition albums.",
  "input_schema": {
    "type": "object",
    "properties": {
      "album_type": {
        "type": "string",
        "enum": [
          "normal",
          "face",
          "baby",
          "condition",
          "object"
        ],
        "description": "The type of album to retrieve. Options: normal (regular album), face (people album), baby (baby album), condition (conditional album), object (object recognition album, 识物相册)."
      }
    },
    "required": [
      "album_type"
    ],
    "additionalProperties": false
  }
}

{
  "name": "music_play_control",
  "description": "Music control tool: play songs, albums, artists, playlists, and other music content. Supports playback modes, and retrieving content from recent history or favorites.",
  "input_schema": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "Name or title of the music content"
      },
      "source": {
        "type": "string",
        "enum": [
          "recent",
          "favorites"
        ],
        "description": "Content source: recent=recently played, favorites=liked songs. Only specify when user explicitly mentions recent or favorite content."
      },
      "play_mode": {
        "type": "string",
        "enum": [
          "normal",
          "random",
          "single",
          "loop"
        ],
        "description": "Playback mode: normal=sequential, random=shuffle, single=repeat single track, loop=repeat all."
      }
    },
    "anyOf": [
      {
        "required": [
          "title"
        ]
      },
      {
        "required": [
          "source"
        ]
      }
    ],
    "additionalProperties": false
  }
}

{
  "name": "music_settings_control",
  "description": "Control music app settings",
  "input_schema": {
    "type": "object",
    "properties": {
      "auto_stop_time": {
        "type": "number",
        "description": "Set sleep timer duration, for example, stop playback after 15 minutes"
      }
    },
    "required": [
      "auto_stop_time"
    ],
    "additionalProperties": false
  }
}

{
  "name": "video_search_control",
  "description": "Video search tool: search TV series, movies, and other video content. ",
  "input_schema": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "Name or title of the video content, supports fuzzy matching."
      },
      "type": {
        "type": "string",
        "enum": [
          "tv",
          "movie",
          "collection"
        ],
        "description": "Content type: tv=TV series/drama, movie=films/blockbusters, collection=movie series/collections."
      }
    },
    "required": [
      "title"
    ],
    "additionalProperties": false
  }
}

{
  "name": "video_play_control",
  "description": "Video play tool: play TV series, movies, and other video content. Supports retrieving content from recently watched history and favorites.",
  "input_schema": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "Name or title of the video content, supports fuzzy matching."
      },
      "type": {
        "type": "string",
        "enum": [
          "tv",
          "movie",
          "collection"
        ],
        "description": "Content type: tv=TV series/drama, movie=films/blockbusters, collection=movie series/collections."
      }
    },
    "required": [
      "title"
    ],
    "additionalProperties": false
  }
}

{
  "name": "get_system_info",
  "description": "Retrieve detailed information about the device, operating system, storage, network status, warranty, or UGREEN Link account.",
  "input_schema": {
    "type": "object",
    "properties": {
      "system_type": {
        "type": "string",
        "description": "The category of information to query. Options: system=system info, device=device info, storage=storage info, network=network info, uglink=UGREEN Link related info.",
        "enum": [
          "system",
          "device",
          "storage",
          "network",
          "uglink"
        ]
      }
    },
    "required": [
      "system_type"
    ],
    "additionalProperties": false
  }
}

Your task:
- Choose the most appropriate function to fulfill the request.
- Include all required parameters; use placeholders if not specified.
- Return ONLY a JSON object with `name` and `arguments`.
- If no function applies, return an empty JSON object: {}

Desired format:
{
	"name": "<function_name>",
	"arguments": {
		"param1": "value1",
		"param2": "value2"
	}
}

Below is the user's request:

"""


react_system_prompt = """You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
> Tool Name: create_album
Tool Description: Create a new photo album
Tool Args: {"type": "object", "properties": {"album_name": {"type": "string", "description": "The name of the album to be created"}, "album_type": {"type": "string", "enum": ["normal"], "description": "The type of the album, default by normal"}}, "required": ["album_name", "album_type"], "additionalProperties": false}

> Tool Name: search_photos
Tool Description: Search for photos or images
Tool Args: {"type": "object", "properties": {"keyword": {"type": "string", "description": "The search keyword for photos or images. It can be descriptive text or a file name, e.g., 'photos taken last August' or 'dog on the grass'."}}, "required": ["keyword"], "additionalProperties": false}

> Tool Name: get_album_list
Tool Description: Retrieve the list of photo albums, including regular albums, people albums, baby albums, conditional albums, and object recognition albums.
Tool Args: {"type": "object", "properties": {"album_type": {"type": "string", "enum": ["normal", "face", "baby", "condition", "object"], "description": "The type of album to retrieve. Options: normal (regular album), face (people album), baby (baby album), condition (conditional album), object (object recognition album, 识物相册)."}}, "required": ["album_type"], "additionalProperties": false}

> Tool Name: music_play_control
Tool Description: Music control tool: play songs, albums, artists, playlists, and other music content. Supports playback modes, and retrieving content from recent history or favorites.
Tool Args: {"type": "object", "properties": {"title": {"type": "string", "description": "Name or title of the music content"}, "source": {"type": "string", "enum": ["recent", "favorites"], "description": "Content source: recent=recently played, favorites=liked songs. Only specify when user explicitly mentions recent or favorite content."}, "play_mode": {"type": "string", "enum": ["normal", "random", "single", "loop"], "description": "Playback mode: normal=sequential, random=shuffle, single=repeat single track, loop=repeat all."}}, "anyOf": [{"required": ["title"]}, {"required": ["source"]}], "additionalProperties": false}

> Tool Name: music_settings_control
Tool Description: Control music app settings
Tool Args: {"type": "object", "properties": {"auto_stop_time": {"type": "number", "description": "Set sleep timer duration, for example, stop playback after 15 minutes"}}, "required": ["auto_stop_time"], "additionalProperties": false}

> Tool Name: video_search_control
Tool Description: Video search tool: search TV series, movies, and other video content.
Tool Args: {"type": "object", "properties": {"title": {"type": "string", "description": "Name or title of the video content, supports fuzzy matching."}, "type": {"type": "string", "enum": ["tv", "movie", "collection"], "description": "Content type: tv=TV series/drama, movie=films/blockbusters, collection=movie series/collections."}}, "required": ["title"], "additionalProperties": false}

> Tool Name: video_play_control
Tool Description: Video play tool: play TV series, movies, and other video content. Supports retrieving content from recently watched history and favorites.
Tool Args: {"type": "object", "properties": {"title": {"type": "string", "description": "Name or title of the video content, supports fuzzy matching."}, "type": {"type": "string", "enum": ["tv", "movie", "collection"], "description": "Content type: tv=TV series/drama, movie=films/blockbusters, collection=movie series/collections."}}, "required": ["title"], "additionalProperties": false}

> Tool Name: get_system_info
Tool Description: Retrieve detailed information about the device, operating system, storage, network status, warranty, or UGREEN Link account.
Tool Args: {"type": "object", "properties": {"system_type": {"type": "string", "description": "The category of information to query. Options: system=system info, device=device info, storage=storage info, network=network info, uglink=UGREEN Link related info.", "enum": ["system", "device", "storage", "network", "uglink"]}}, "required": ["system_type"], "additionalProperties": false}

## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of create_album, search_photos, get_album_list, music_play_control, music_settings_control, video_search_control, video_play_control, get_system_info) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {"input": "hello world", "num_beams": 5})
```

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Below is the user's request:

"""

system_prompt_dict = {"v0": system_prompt_v0, "react": react_system_prompt}

system_prompt = system_prompt_dict.get(KEY_PROMPT)
# In[15]:


user_prompt = """Below is the user's request:
{query}"""


df = df[["query", "model_function_calls", "voted_function_call", "language"]]


# In[12]:


df


# In[13]:


lora_input_list = []
for i, df_0 in df.iterrows():
    df_ = df_0.to_dict()
    lora_input_list.append(
        {"instruction": system_prompt, "input": df_.get("query", ""), "output": str(df_.get("voted_function_call", ""))}
    )


# In[14]:


df["lora_input_list"] = lora_input_list


# In[15]:


df


# In[16]:


print(df.iloc[1]["lora_input_list"].get("input"))


# In[17]:


print(df.iloc[1]["lora_input_list"].get("output"))


# In[18]:


print(df.iloc[1]["lora_input_list"].get("instruction"))


# In[19]:


df.shape


# In[20]:


df.to_excel(root + "to_lora_raw_data.xlsx")


# In[21]:


df


# #  train dev test 拆分

# In[22]:


from sklearn.model_selection import train_test_split

# df = df2.copy()
# 第一步：将数据分为训练集和临时集（包含 dev + test）
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=2025)

# 第二步：将临时集再分为 dev 和 test
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=2025)

# test_df = temp_df.copy()
# dev_df = temp_df.copy()
# 结果：
# train_df: 80% × 80% = 64%
# dev_df:   80% × 20% = 16%
# test_df:  80% × 20% = 16%


# In[23]:


train_df.shape, dev_df.shape, test_df.shape


# In[24]:


train_df.iloc[1]["lora_input_list"]

train_df.to_excel(root + "train_all.xlsx")
dev_df.to_excel(root + "dev_all.xlsx")
test_df.to_excel(root + "test_all.xlsx")


# In[26]:


lora_input_list_train = train_df["lora_input_list"].to_list()
lora_input_list_dev = dev_df["lora_input_list"].to_list()
lora_input_list_test = test_df["lora_input_list"].to_list()


# In[27]:


len(lora_input_list_train), len(lora_input_list_dev), len(lora_input_list_test)


# In[28]:


lora_input_list_train


# In[29]:


with open(root + "mcp_train.json", "w") as fin:
    json.dump(lora_input_list_train, fin, ensure_ascii=False, indent=2)


# In[30]:


with open(root + "mcp_dev.json", "w") as fin2:
    json.dump(lora_input_list_dev, fin2, ensure_ascii=False, indent=2)


# In[31]:


with open(root + "mcp_test.json", "w") as fin3:
    json.dump(lora_input_list_test, fin3, ensure_ascii=False, indent=2)


# In[32]:


len(lora_input_list)
