#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：字符串操作
学习目标：深入掌握字符串的各种操作方法
PyCharm 技巧：学习使用代码补全和文档查看
============================================================================
"""

# ============================================================================
# 第一部分：字符串基础
# ============================================================================

# ----------------------------------------------------------------------------
# 字符串创建
# ----------------------------------------------------------------------------

# 四种创建字符串的方式
s1 = '单引号字符串'
s2 = "双引号字符串"
s3 = '''三个单引号
可以换行'''
s4 = """三个双引号
也可以换行"""

print(s1)
print(s2)
print(s3)
print(s4)

# 引号嵌套
sentence1 = "他说：'你好！'"    # 外双内单
sentence2 = '他说："你好！"'    # 外单内双
print(sentence1)
print(sentence2)

# ----------------------------------------------------------------------------
# 转义字符
# ----------------------------------------------------------------------------

"""
常用转义字符：
\\  - 反斜杠
\'  - 单引号
\"  - 双引号
\n  - 换行
\r  - 回车
\t  - 制表符
\b  - 退格
\f  - 换页
"""

# 转义字符示例
print("Hello\nWorld")       # 换行
print("Name:\tAlice")       # 制表符
print("Path: C:\\Users")    # 反斜杠

# 原始字符串（不处理转义）
raw_str = r"Raw string: \n \t \\"
print(raw_str)  # 原样输出

# 【PyCharm 技巧】
# 按 Ctrl + Q（或 F1）可以查看函数/方法的文档说明
# 输入 str. 后按 Ctrl + Space 可以查看所有字符串方法

# ============================================================================
# 第二部分：字符串索引与切片
# ============================================================================

# ----------------------------------------------------------------------------
# 字符串索引
# ----------------------------------------------------------------------------

text = "Python"

# 正向索引（从0开始）
print(f"text[0] = {text[0]}")   # P
print(f"text[1] = {text[1]}")   # y
print(f"text[5] = {text[5]}")   # n

# 反向索引（从-1开始）
print(f"text[-1] = {text[-1]}") # n（最后一个）
print(f"text[-2] = {text[-2]}") # o
print(f"text[-6] = {text[-6]}") # P（第一个）

# 索引图示：
"""
索引：    0   1   2   3   4   5
字符：    P   y   t   h   o   n
反向索引：-6  -5  -4  -3  -2  -1
"""

# ----------------------------------------------------------------------------
# 字符串切片
# ----------------------------------------------------------------------------

"""
切片语法：string[start:end:step]
- start: 起始索引（包含，默认0）
- end: 结束索引（不包含，默认到末尾）
- step: 步长（默认1）
"""

text = "Hello, Python!"

# 基本切片
print(f"text[0:5] = {text[0:5]}")    # Hello
print(f"text[7:13] = {text[7:13]}")  # Python
print(f"text[:5] = {text[:5]}")      # Hello（省略start）
print(f"text[7:] = {text[7:]}")      # Python!（省略end）

# 负数切片
print(f"text[-6:] = {text[-6:]}")    # thon!（后6个字符）
print(f"text[:-7] = {text[:-7]}")    # Hello, P（排除后7个）

# 步长切片
print(f"text[::2] = {text[::2]}")    # Hlo yhn（每隔一个取一个）
print(f"text[1::2] = {text[1::2]}")  # el,Pto!（从索引1开始，每隔一个）

# 反转字符串
print(f"text[::-1] = {text[::-1]}")  # !nohtyP ,olleH

# 切片不会报错（超出范围返回空字符串或有效部分）
print(f"text[100:200] = '{text[100:200]}'")  # 空字符串

# ============================================================================
# 第三部分：字符串常用方法
# ============================================================================

# ----------------------------------------------------------------------------
# 大小写转换
# ----------------------------------------------------------------------------

text = "Hello, Python World!"

print(f"原始: {text}")
print(f"upper(): {text.upper()}")       # 全大写
print(f"lower(): {text.lower()}")       # 全小写
print(f"capitalize(): {text.capitalize()}")  # 首字母大写
print(f"title(): {text.title()}")       # 每个单词首字母大写
print(f"swapcase(): {text.swapcase()}") # 大小写互换

# 判断大小写
print(f"isupper(): {'HELLO'.isupper()}")  # True
print(f"islower(): {'hello'.islower()}")  # True
print(f"istitle(): {'Hello World'.istitle()}")  # True

# ----------------------------------------------------------------------------
# 查找与替换
# ----------------------------------------------------------------------------

text = "Hello, Python! Python is great!"

# find() - 查找子串，返回第一个索引，找不到返回-1
print(f"find('Python'): {text.find('Python')}")    # 7
print(f"find('Java'): {text.find('Java')}")        # -1
print(f"find('Python', 10): {text.find('Python', 10)}")  # 15，从索引10开始

# index() - 类似find()，但找不到会报错
print(f"index('Python'): {text.index('Python')}")  # 7
# print(text.index('Java'))  # ValueError

# rfind() / rindex() - 从右边开始查找
print(f"rfind('Python'): {text.rfind('Python')}")  # 15

# count() - 统计子串出现次数
print(f"count('Python'): {text.count('Python')}")  # 2

# replace() - 替换
print(f"replace('Python', 'Java'): {text.replace('Python', 'Java')}")
print(f"replace('Python', 'Java', 1): {text.replace('Python', 'Java', 1)}")  # 只替换1次

# ----------------------------------------------------------------------------
# 分割与连接
# ----------------------------------------------------------------------------

# split() - 分割字符串
text = "apple,banana,orange,grape"
fruits = text.split(",")
print(f"split(','): {fruits}")  # ['apple', 'banana', 'orange', 'grape']

# 限制分割次数
print(f"split(',', 2): {text.split(',', 2)}")  # ['apple', 'banana', 'orange,grape']

# splitlines() - 按行分割
lines = "第一行\n第二行\n第三行"
print(f"splitlines(): {lines.splitlines()}")

# partition() - 分成三部分
text = "user@example.com"
before, sep, after = text.partition("@")
print(f"partition('@'): before='{before}', sep='{sep}', after='{after}'")

# join() - 连接字符串列表
words = ["Hello", "Python", "World"]
print(f"' '.join(words): {' '.join(words)}")    # Hello Python World
print(f"'-'.join(words): {'-'.join(words)}")    # Hello-Python-World
print(f"''.join(words): {''.join(words)}")      # HelloPythonWorld

# ----------------------------------------------------------------------------
# 去除空白字符
# ----------------------------------------------------------------------------

text = "   Hello, World!   "

print(f"原始: '{text}'")
print(f"strip(): '{text.strip()}'")     # 去除两端空白
print(f"lstrip(): '{text.lstrip()}'")   # 去除左边空白
print(f"rstrip(): '{text.rstrip()}'")   # 去除右边空白

# 去除指定字符
text = "xxHello, World!xx"
print(f"strip('x'): '{text.strip('x')}'")  # Hello, World!

# ----------------------------------------------------------------------------
# 判断方法
# ----------------------------------------------------------------------------

# 判断字符串开头/结尾
filename = "document.pdf"
print(f"startswith('doc'): {filename.startswith('doc')}")   # True
print(f"endswith('.pdf'): {filename.endswith('.pdf')}")     # True

# 判断字符串内容
print(f"'12345'.isdigit(): {'12345'.isdigit()}")        # True，全是数字
print(f"'hello'.isalpha(): {'hello'.isalpha()}")        # True，全是字母
print(f"'hello123'.isalnum(): {'hello123'.isalnum()}")  # True，字母或数字
print(f"'   '.isspace(): {'   '.isspace()}")            # True，全是空白
print(f"'Hello World'.isprintable(): {'Hello World'.isprintable()}")  # True

# ----------------------------------------------------------------------------
# 填充与对齐
# ----------------------------------------------------------------------------

text = "Python"

# 居中
print(f"center(20): '{text.center(20)}'")       # 两边填充空格
print(f"center(20, '*'): '{text.center(20, '*')}'")  # 指定填充字符

# 左对齐
print(f"ljust(20): '{text.ljust(20)}'")
print(f"ljust(20, '-'): '{text.ljust(20, '-')}'")

# 右对齐
print(f"rjust(20): '{text.rjust(20)}'")
print(f"rjust(20, '0'): '{text.rjust(20, '0')}'")

# 填充0（常用于数字）
num = "42"
print(f"zfill(5): '{num.zfill(5)}'")  # 00042

# ----------------------------------------------------------------------------
# 格式化方法
# ----------------------------------------------------------------------------

# format() 方法
template = "我叫{}，今年{}岁"
print(template.format("小明", 20))

# 带索引
template = "{0}说：{1}，{0}很高兴"
print(template.format("小明", "你好"))

# 带关键字
template = "姓名：{name}，年龄：{age}"
print(template.format(name="小红", age=18))

# 格式化数字
print("{:.2f}".format(3.14159))     # 3.14
print("{:,}".format(1234567))       # 1,234,567（千分位）
print("{:.1%}".format(0.856))       # 85.6%
print("{:+.2f}".format(3.14))       # +3.14（显示正负号）

# ============================================================================
# 第四部分：字符串格式化详解
# ============================================================================

# ----------------------------------------------------------------------------
# f-string（推荐）
# ----------------------------------------------------------------------------

"""
f-string 是 Python 3.6+ 引入的格式化方式，简洁高效。
"""

name = "小明"
age = 20
score = 95.5

# 基本用法
print(f"姓名：{name}，年龄：{age}")

# 表达式计算
print(f"明年 {age + 1} 岁")
print(f"成绩的平方：{score ** 2}")

# 调用方法
print(f"姓名大写：{name.upper()}")

# 格式化选项
pi = 3.14159265
print(f"保留两位小数：{pi:.2f}")
print(f"百分比：{0.856:.1%}")
print(f"科学计数法：{123456789:.2e}")
print(f"千分位：{1234567:,}")

# 宽度和对齐
print(f"右对齐：{name:>10}")
print(f"左对齐：{name:<10}")
print(f"居中：{name:^10}")
print(f"填充：{name:*^10}")

# 进制转换
num = 42
print(f"二进制：{num:b}")
print(f"八进制：{num:o}")
print(f"十六进制：{num:x}")
print(f"十六进制（大写）：{num:X}")

# 嵌套引号
print(f"他说：'{name}'")

# 多行 f-string
message = f"""
学生信息：
  姓名：{name}
  年龄：{age}
  成绩：{score}
"""
print(message)

# ============================================================================
# 第五部分：字符串编码
# ============================================================================

# ----------------------------------------------------------------------------
# 编码与解码
# ----------------------------------------------------------------------------

text = "你好，Python"

# 编码（字符串 → 字节）
bytes_utf8 = text.encode("utf-8")
bytes_gbk = text.encode("gbk")

print(f"UTF-8 编码: {bytes_utf8}")
print(f"GBK 编码: {bytes_gbk}")

# 解码（字节 → 字符串）
decoded = bytes_utf8.decode("utf-8")
print(f"解码结果: {decoded}")

# 处理编码错误
invalid_bytes = b'\xff\xfe'
# decoded = invalid_bytes.decode('utf-8')  # UnicodeDecodeError
decoded = invalid_bytes.decode('utf-8', errors='ignore')  # 忽略错误
print(f"忽略错误解码: '{decoded}'")

decoded = invalid_bytes.decode('utf-8', errors='replace')  # 替换错误
print(f"替换错误解码: '{decoded}'")

# ============================================================================
# 第六部分：字符串常用技巧
# ============================================================================

# ----------------------------------------------------------------------------
# 实用技巧
# ----------------------------------------------------------------------------

# 1. 反转字符串
s = "Python"
reversed_s = s[::-1]
print(f"反转: {reversed_s}")

# 2. 判断回文
def is_palindrome(s):
    """判断是否是回文"""
    s = s.lower().replace(" ", "")  # 忽略大小写和空格
    return s == s[::-1]

print(f"'上海自来水来自海上' 是回文: {is_palindrome('上海自来水来自海上')}")

# 3. 字符串去重
s = "aaabbbccc"
unique = "".join(sorted(set(s)))
print(f"去重: {unique}")

# 4. 找出最长的单词
sentence = "The quick brown fox jumps"
longest = max(sentence.split(), key=len)
print(f"最长单词: {longest}")

# 5. 统计字符出现次数
from collections import Counter
text = "hello world"
char_count = Counter(text)
print(f"字符统计: {char_count}")

# 6. 多行字符串处理
multiline = """
    第一行
    第二行
    第三行
"""
# 使用 textwrap 处理缩进
import textwrap
cleaned = textwrap.dedent(multiline)
print(f"清理缩进:\n{cleaned}")

# ============================================================================
# 练习题
# ============================================================================
"""
【练习1】字符串切片
给定字符串 "Hello, Python World!"，完成以下操作：
1. 提取 "Python"
2. 提取 "World"
3. 反转整个字符串
4. 每隔一个字符提取

【练习2】字符串格式化
创建一个学生成绩报告，包含：
- 学生姓名
- 三门课程的成绩
- 平均分（保留两位小数）
- 是否及格（平均分 >= 60）

【练习3】字符串处理
编写函数实现：
1. 统计字符串中元音字母的数量
2. 将字符串中的单词首字母大写
3. 删除字符串中的所有数字
"""

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. 字符串创建和转义字符
2. 字符串索引和切片
3. 字符串常用方法（大小写、查找、分割等）
4. f-string 格式化
5. 字符串编码和解码

🔧 PyCharm 技巧：
1. Ctrl + Q 查看文档
2. Ctrl + Space 代码补全
3. Ctrl + P 查看参数信息
4. Ctrl + Shift + Enter 补全语句

➡️ 下一节：条件语句
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("字符串操作模块学习完成！")
    print("=" * 60)
