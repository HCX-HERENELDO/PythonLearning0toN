#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：序列化
学习目标：掌握 pickle 和其他序列化方法
PyCharm 技巧：学习查看序列化数据
============================================================================
"""

import pickle
import json
from pathlib import Path

# ============================================================================
# 第一部分：pickle 基础
# ============================================================================
"""
【概念讲解】
pickle 是 Python 标准库中的序列化模块，
可以将任意 Python 对象转换为字节流，
实现对象的持久化存储和网络传输。
"""

# ----------------------------------------------------------------------------
# 基本序列化和反序列化
# ----------------------------------------------------------------------------

# 要序列化的数据
data = {
    "name": "张三",
    "age": 25,
    "scores": [85, 90, 78],
    "active": True
}

# 序列化为字节
serialized = pickle.dumps(data)
print(f"序列化类型: {type(serialized)}")
print(f"序列化大小: {len(serialized)} 字节")

# 反序列化
deserialized = pickle.loads(serialized)
print(f"反序列化: {deserialized}")

# ----------------------------------------------------------------------------
# 序列化到文件
# ----------------------------------------------------------------------------

# 写入文件
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)
print("数据已保存到文件")

# 从文件读取
with open("data.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(f"从文件加载: {loaded_data}")

# ============================================================================
# 第二部分：pickle 协议版本
# ============================================================================

# 查看支持的协议版本
print(f"最高协议版本: {pickle.HIGHEST_PROTOCOL}")

# 使用不同协议版本
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    serialized = pickle.dumps(data, protocol=protocol)
    print(f"协议 {protocol}: {len(serialized)} 字节")

# 使用最高协议（推荐）
serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

# ============================================================================
# 第三部分：序列化复杂对象
# ============================================================================

# ----------------------------------------------------------------------------
# 序列化自定义类
# ----------------------------------------------------------------------------

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"

# 创建对象
person = Person("李四", 30)

# 序列化
person_bytes = pickle.dumps(person)
restored_person = pickle.loads(person_bytes)
print(f"恢复的对象: {restored_person}")

# ----------------------------------------------------------------------------
# 序列化函数和类
# ----------------------------------------------------------------------------

def greet(name):
    return f"Hello, {name}!"

# 序列化函数
func_bytes = pickle.dumps(greet)
restored_func = pickle.loads(func_bytes)
print(restored_func("王五"))

# ============================================================================
# 第四部分：pickle vs JSON
# ============================================================================

"""
【对比】

pickle:
+ 可以序列化任意 Python 对象
+ 自动处理复杂类型
- 只能在 Python 中使用
- 存在安全风险
- 不适合人类阅读

JSON:
+ 跨语言通用
+ 可读性好
+ 安全
- 只支持基本数据类型
- 需要额外处理复杂对象
"""

class Student:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores
    
    def to_dict(self):
        return {"name": self.name, "scores": self.scores}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["scores"])

student = Student("张三", [90, 85, 92])

# 使用 pickle
pickle_data = pickle.dumps(student)
print(f"pickle 大小: {len(pickle_data)} 字节")

# 使用 JSON（需要转换）
json_data = json.dumps(student.to_dict())
print(f"JSON 大小: {len(json_data)} 字节")
print(f"JSON 内容: {json_data}")

# ============================================================================
# 第五部分：安全注意事项
# ============================================================================

"""
【安全警告】
pickle 可能执行任意代码，不要加载不可信来源的 pickle 文件！

错误示例：
# 危险！可能执行恶意代码
data = pickle.loads(untrusted_bytes)

正确做法：
1. 只加载可信来源的 pickle 文件
2. 使用 JSON 替代 pickle（如果可能）
3. 对敏感数据使用签名验证
"""

def safe_load(filepath, expected_type=None):
    """安全加载 pickle 文件"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    if expected_type and not isinstance(data, expected_type):
        raise TypeError(f"预期类型 {expected_type}, 实际类型 {type(data)}")
    
    return data

# ============================================================================
# 第六部分：其他序列化选项
# ============================================================================

# ----------------------------------------------------------------------------
# dataclasses + JSON
# ----------------------------------------------------------------------------

from dataclasses import dataclass, asdict
import json

@dataclass
class Book:
    title: str
    author: str
    price: float
    tags: list

book = Book("Python编程", "作者", 59.9, ["编程", "Python"])

# 转换为字典再序列化
book_dict = asdict(book)
book_json = json.dumps(book_dict, ensure_ascii=False)
print(f"书籍 JSON: {book_json}")

# 恢复
book_data = json.loads(book_json)
restored_book = Book(**book_data)
print(f"恢复的书籍: {restored_book}")

# ----------------------------------------------------------------------------
# 使用 __getstate__ 和 __setstate__
# ----------------------------------------------------------------------------

class SecureData:
    def __init__(self, data, password):
        self.data = data
        self._password = password  # 不应该被序列化
    
    def __getstate__(self):
        """自定义序列化状态"""
        state = self.__dict__.copy()
        del state['_password']  # 移除敏感数据
        return state
    
    def __setstate__(self, state):
        """自定义反序列化状态"""
        self.__dict__.update(state)
        self._password = None  # 设置默认值

secure = SecureData("敏感数据", "secret123")
secure_bytes = pickle.dumps(secure)
restored = pickle.loads(secure_bytes)
print(f"恢复的数据: {restored.data}")
print(f"密码: {restored._password}")  # None

# ============================================================================
# 清理测试文件
# ============================================================================

if Path("data.pkl").exists():
    Path("data.pkl").unlink()

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. pickle 基本使用
2. 序列化协议版本
3. 序列化自定义类
4. pickle vs JSON 对比
5. 安全注意事项
6. 自定义序列化行为

➡️ 下一节：CSV处理
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("序列化模块学习完成！")
    print("=" * 60)
