#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试parse_react_to_json函数的各种输入情况
"""

import json
import ast
import pandas as pd

def parse_react_to_json(s):
    """
    安全解析JSON字符串，支持多种格式：
    1. 标准JSON格式: {"intent": "music_play_control", "slots": {"title": "周杰伦"}}
    2. Python字典格式: {'intent': 'music_play_control', 'slots': {'title': '周杰伦'}}
    3. 字符串包装的字典: "{'intent': 'music_play_control', 'slots': {'title': '周杰伦'}}"
    """
    if pd.isna(s):
        return {}
    if isinstance(s, dict):
        return s
    
    # 确保输入是字符串
    if not isinstance(s, str):
        s = str(s)
    
    # 去除首尾空白字符
    s = s.strip()
    
    # 如果是空字符串
    if not s:
        return {}
    
    try:
        # 方法1: 优先尝试标准JSON解析
        print(f"方法1: json.loads(s) {s}")
        return json.loads(s)
    except Exception:
        try:
            # 方法2: 尝试解析Python字典格式（单引号形式）
            print(f"方法2: ast.literal_eval: {s}")
            return ast.literal_eval(s)
        except Exception:
            try:
                # 方法3: 如果外层有引号，先去掉引号再解析
                if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                    inner_s = s[1:-1]
                    try:
                        print(f"方法3: json.loads(inner_s): {inner_s}")
                        return json.loads(inner_s)
                    except Exception:
                        print(f"方法3: json.loads(inner_s): {inner_s}")
                        return ast.literal_eval(inner_s)
                else:
                    return {}
            except Exception:
                # 方法4: 最后尝试替换单引号为双引号后解析JSON
                try:
                    # 简单的单引号到双引号转换（可能不完美，但适用于大多数情况）
                    json_str = s.replace("'", '"')
                    print(f"方法4: 最后尝试替换单引号为双引号后解析JSON: {json_str}")
                    return json.loads(json_str)
                except Exception:
                    print(f"无法解析字符串: {repr(s)}")
                    return {}

def test_parse_function():
    """测试各种输入情况"""
    
    # 测试用例
    test_cases = [
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
