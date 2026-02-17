#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：JSON处理
学习目标：掌握 JSON 数据的读写和解析
PyCharm 技巧：学习 JSON 格式化工具
============================================================================
"""

import json

# ============================================================================
# 第一部分：JSON 基础
# ============================================================================
"""
【概念讲解】
JSON (JavaScript Object Notation) 是一种轻量级数据交换格式。
Python 的 json 模块可以处理 JSON 数据。

JSON 类型映射：
- JSON object → Python dict
- JSON array → Python list
- JSON string → Python str
- JSON number → Python int/float
- JSON true/false → Python True/False
- JSON null → Python None
"""

# ----------------------------------------------------------------------------
# Python 对象转 JSON
# ----------------------------------------------------------------------------

# 字典
data = {
    "name": "张三",
    "age": 25,
    "is_student": True,
    "courses": ["Python", "Java", "数据库"],
    "address": None
}

# 转换为 JSON 字符串
json_str = json.dumps(data)
print(f"JSON 字符串: {json_str}")

# 格式化输出
json_formatted = json.dumps(data, indent=2, ensure_ascii=False)
print(f"格式化 JSON:\n{json_formatted}")

# 排序键
json_sorted = json.dumps(data, sort_keys=True, ensure_ascii=False)
print(f"排序后: {json_sorted}")

# ----------------------------------------------------------------------------
# JSON 字符串转 Python 对象
# ----------------------------------------------------------------------------

json_string = '{"name": "李四", "age": 30, "city": "北京"}'
parsed = json.loads(json_string)
print(f"解析结果: {parsed}")
print(f"类型: {type(parsed)}")

# ============================================================================
# 第二部分：文件操作
# ============================================================================

# ----------------------------------------------------------------------------
# 写入 JSON 文件
# ----------------------------------------------------------------------------

data = {
    "users": [
        {"id": 1, "name": "张三", "email": "zhang@example.com"},
        {"id": 2, "name": "李四", "email": "li@example.com"},
    ],
    "total": 2
}

with open("users.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("JSON 文件已写入")

# ----------------------------------------------------------------------------
# 读取 JSON 文件
# ----------------------------------------------------------------------------

with open("users.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

print(f"读取的数据: {loaded_data}")

# ============================================================================
# 第三部分：自定义序列化
# ============================================================================

# ----------------------------------------------------------------------------
# 处理不支持的类型
# ----------------------------------------------------------------------------

from datetime import datetime, date

class DateTimeEncoder(json.JSONEncoder):
    """自定义 JSON 编码器"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

# 使用自定义编码器
event = {
    "name": "会议",
    "time": datetime.now(),
    "date": date.today()
}

json_str = json.dumps(event, cls=DateTimeEncoder, ensure_ascii=False)
print(f"自定义编码: {json_str}")

# ----------------------------------------------------------------------------
# 自定义解码器
# ----------------------------------------------------------------------------

def datetime_decoder(dct):
    """自定义解码函数"""
    for key, value in dct.items():
        if isinstance(value, str):
            # 尝试解析日期时间
            try:
                dct[key] = datetime.fromisoformat(value)
            except ValueError:
                pass
    return dct

# ============================================================================
# 第四部分：实际应用
# ============================================================================

# ----------------------------------------------------------------------------
# 配置文件
# ----------------------------------------------------------------------------

class Config:
    """配置管理类"""
    
    def __init__(self, filename="config.json"):
        self.filename = filename
        self.data = {}
        self.load()
    
    def load(self):
        """加载配置"""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}
    
    def save(self):
        """保存配置"""
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value
        self.save()

# 使用
config = Config()
config.set("theme", "dark")
config.set("language", "zh-CN")
print(f"主题: {config.get('theme')}")

# ----------------------------------------------------------------------------
# API 响应处理
# ----------------------------------------------------------------------------

def parse_api_response(response_text):
    """解析 API 响应"""
    try:
        data = json.loads(response_text)
        if data.get("success"):
            return data.get("data")
        else:
            raise Exception(data.get("error", "未知错误"))
    except json.JSONDecodeError as e:
        raise Exception(f"JSON 解析失败: {e}")

# 模拟 API 响应
api_response = '{"success": true, "data": {"user": "张三", "score": 95}}'
result = parse_api_response(api_response)
print(f"API 结果: {result}")

# ============================================================================
# 第五部分：JSON 验证
# ============================================================================

def validate_json(json_str, schema):
    """简单 JSON 验证"""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return False, "无效的 JSON 格式"
    
    for key, expected_type in schema.items():
        if key not in data:
            return False, f"缺少字段: {key}"
        if not isinstance(data[key], expected_type):
            return False, f"字段 {key} 类型错误"
    
    return True, "验证通过"

# 验证示例
schema = {
    "name": str,
    "age": int,
    "email": str
}

test_json = '{"name": "张三", "age": 25, "email": "test@example.com"}'
valid, message = validate_json(test_json, schema)
print(f"验证结果: {valid}, {message}")

# ============================================================================
# 清理测试文件
# ============================================================================

import os
for f in ["users.json", "config.json"]:
    if os.path.exists(f):
        os.remove(f)

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. json.dumps() 和 json.loads()
2. json.dump() 和 json.load()
3. 格式化和编码选项
4. 自定义序列化
5. JSON 配置文件处理

➡️ 下一节：CSV处理
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JSON处理模块学习完成！")
    print("=" * 60)
