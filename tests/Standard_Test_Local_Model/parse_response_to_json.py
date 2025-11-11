#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试parse_react_to_json函数的各种输入情况
"""

import json
import ast
import pandas as pd
import re

def extract_json_inside_braces(s):
    """
    用正则和计数法 提取从第一个{到最后一个}之间的内容。
    允许有```json标识或者其它前缀。
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    # 先找第一个{
    idx1 = s.find('{')
    # 从后面找最后一个}
    idx2 = s.rfind('}')
    if idx1 != -1 and idx2 != -1 and idx2 > idx1:
        return s[idx1:idx2+1]
    return None

def parse_react_to_json(s):
    """
    安全解析JSON字符串，支持多种格式，并兼容前后有其它内容的情况。
    1. 标准JSON格式: {"intent": "music_play_control", "slots": {"title": "周杰伦"}}
    2. Python字典格式: {'intent': 'music_play_control', 'slots': {'title': '周杰伦'}}
    3. 字符串包装的字典: "{'intent': 'music_play_control', 'slots': {'title': '周杰伦'}}"
    4. 外层带markdown等标识的情况, 自动截取第一个{到最后一个}尝试解析
    """
    # 兼容pandas缺失值
    if pd.isna(s):
        return {}
    if isinstance(s, dict):
        return s

    # to string
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()

    if not s:
        return {}

    # --- 新增：尝试提取第一个{到最后一个} ---
    outer_json = extract_json_inside_braces(s)
    parse_attempts = []
    if outer_json and outer_json != s:
        parse_attempts.append(outer_json)
    parse_attempts.append(s)

    for candidate in parse_attempts:
        # 统一处理每个版本
        candidate = candidate.strip()
        # 方法1: 优先json.loads
        try:
            print(f"方法1: json.loads(candidate): {candidate}")
            return json.loads(candidate)
        except Exception:
            pass
        # 方法2: ast.literal_eval
        try:
            print(f"方法2: ast.literal_eval(candidate): {candidate}")
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 方法3: 去除外层引号再json或ast
        if (candidate.startswith('"') and candidate.endswith('"')) or (candidate.startswith("'") and candidate.endswith("'")):
            inner_s = candidate[1:-1]
            try:
                print(f"方法3: json.loads(inner_s): {inner_s}")
                return json.loads(inner_s)
            except Exception:
                try:
                    print(f"方法3: ast.literal_eval(inner_s): {inner_s}")
                    obj = ast.literal_eval(inner_s)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
        # 方法4: 单引转双引再json
        try:
            json_str = candidate.replace("'", '"')
            print(f"方法4: json.loads(replace '): {json_str}")
            return json.loads(json_str)
        except Exception:
            pass

    print(f"无法解析字符串: {repr(s)}")
    return {}

def test_parse_function():
    """测试各种输入情况"""
    
    # 测试用例
    test_cases = [
        # 情况0: 标准JSON格式（有markdown前缀）
        """```json
{
"input": "Ich muss das Album 'Sternenhimmel Camping🌌Night' durchsuchen.",
"output": {"intent": "get_album_list", "slots": {"album_type": "", "keyword": "Sternenhimmel Camping🌌Night"}}
}""",

        # 情况1: 标准JSON格式（双引号）
        '{"intent": "music_play_control", "slots": {"title": "周杰伦"}}',
        
        # 情况2: Python字典格式（单引号）
        "{'intent': 'music_play_control', 'slots': {'title': '周杰伦'}}",
        
        # 情况3: 字符串包装的字典（外层双引号）
        '"{\\"intent\\": \\"music_play_control\\", \\"slots\\": {\\"title\\": \\"周杰伦\\"}}"',
        
        # 情况4: 字符串包装的字典（外层单引号）
        "'{'intent': 'music_play_control', 'slots': {'title': '周杰伦'}}'",
        
        # 情况5: 已经是字典
        {'intent': 'music_play_control', 'slots': {'title': '周杰伦'}},
        
        # 情况6: 空字符串
        '',
        
        # 情况7: None值
        None,
        
        # 情况8: 带空格的字符串
        '  {"intent": "music_play_control", "slots": {"title": "周杰伦"}}  ',
        
        # 情况9: 复杂嵌套
        "{'intent': 'create_album', 'slots': {'album_name': '我的相册', 'album_type': 'normal'}}",
        
        # 情况10: 无效字符串
        'invalid json string',
    ]
    
    print("=== 测试parse_react_to_json函数 ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"输入: {repr(test_case)}")
        
        try:
            result = parse_react_to_json(test_case)
            print(f"输出: {result}")
            print(f"类型: {type(result)}")
            
            # 验证结果
            if isinstance(result, dict):
                print("✅ 解析成功")
                if 'intent' in result:
                    print(f"   意图: {result['intent']}")
                if 'slots' in result:
                    print(f"   槽位: {result['slots']}")
            else:
                print("❌ 解析失败：结果不是字典")
                
        except Exception as e:
            print(f"❌ 解析出错: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_parse_function()
