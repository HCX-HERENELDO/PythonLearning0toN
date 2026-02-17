#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：作用域与闭包
学习目标：理解变量作用域和闭包函数的原理
PyCharm 技巧：学习查看变量的作用域链
============================================================================
"""

# ============================================================================
# 第一部分：变量作用域
# ============================================================================
"""
【概念讲解】
Python 有四种作用域（LEGB 规则）：
1. Local - 局部作用域（函数内部）
2. Enclosing - 嵌套作用域（外层函数）
3. Global - 全局作用域（模块级别）
4. Built-in - 内置作用域（Python 内置）
"""

# ----------------------------------------------------------------------------
# LEGB 规则演示
# ----------------------------------------------------------------------------

# Global 作用域
x = "global"

def outer():
    # Enclosing 作用域
    x = "enclosing"
    
    def inner():
        # Local 作用域
        x = "local"
        print(f"内部 x: {x}")  # local
    
    inner()
    print(f"外部 x: {x}")  # enclosing

outer()
print(f"全局 x: {x}")  # global

# ----------------------------------------------------------------------------
# global 关键字
# ----------------------------------------------------------------------------

count = 0

def increment():
    """使用 global 修改全局变量"""
    global count
    count += 1
    print(f"count = {count}")

increment()
increment()
increment()
print(f"最终 count = {count}")

# 【注意】尽量避免使用 global，改用参数传递和返回值

# ----------------------------------------------------------------------------
# nonlocal 关键字
# ----------------------------------------------------------------------------

def counter():
    """计数器工厂函数"""
    count = 0
    
    def increment():
        nonlocal count  # 引用外层函数的变量
        count += 1
        return count
    
    return increment

# 创建计数器
c1 = counter()
print(f"c1: {c1()}, {c1()}, {c1()}")

c2 = counter()  # 新的计数器
print(f"c2: {c2()}, {c2()}")

# ============================================================================
# 第二部分：闭包
# ============================================================================
"""
【概念讲解】
闭包是指函数与其引用环境（外部变量）的组合。
闭包允许函数"记住"创建时的环境。
"""

# ----------------------------------------------------------------------------
# 闭包的基本结构
# ----------------------------------------------------------------------------

def make_multiplier(factor):
    """
    创建乘法函数（闭包示例）
    
    factor 被"记住"在返回的函数中
    """
    def multiplier(x):
        return x * factor
    return multiplier

# 创建不同的乘法器
double = make_multiplier(2)
triple = make_multiplier(3)

print(f"double(5) = {double(5)}")  # 10
print(f"triple(5) = {triple(5)}")  # 15

# ----------------------------------------------------------------------------
# 闭包的实际应用
# ----------------------------------------------------------------------------

# 1. 配置函数
def create_formatter(prefix="", suffix=""):
    """创建格式化函数"""
    def format_text(text):
        return f"{prefix}{text}{suffix}"
    return format_text

bold = create_formatter("<b>", "</b>")
italic = create_formatter("<i>", "</i>")

print(bold("Hello"))    # <b>Hello</b>
print(italic("World"))  # <i>World</i>

# 2. 缓存/记忆化
def memoize(func):
    """记忆化装饰器（使用闭包）"""
    cache = {}
    
    def wrapper(*args):
        if args in cache:
            print(f"缓存命中: {args}")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(f"fibonacci(10) = {fibonacci(10)}")

# 3. 状态保持
def make_accumulator():
    """创建累加器"""
    total = 0
    
    def add(value):
        nonlocal total
        total += value
        return total
    
    return add

acc = make_accumulator()
print(f"累加: {acc(10)}, {acc(20)}, {acc(30)}")

# ============================================================================
# 第三部分：闭包的陷阱
# ============================================================================

# ----------------------------------------------------------------------------
# 循环中的闭包陷阱
# ----------------------------------------------------------------------------

# 错误示例
funcs_wrong = []
for i in range(3):
    funcs_wrong.append(lambda: i)

print("错误结果:", [f() for f in funcs_wrong])  # [2, 2, 2]

# 正确方法1：使用默认参数
funcs_correct1 = []
for i in range(3):
    funcs_correct1.append(lambda x=i: x)

print("正确结果1:", [f() for f in funcs_correct1])  # [0, 1, 2]

# 正确方法2：使用闭包工厂
def make_func(x):
    return lambda: x

funcs_correct2 = [make_func(i) for i in range(3)]
print("正确结果2:", [f() for f in funcs_correct2])

# ============================================================================
# 第四部分：查看闭包信息
# ============================================================================

def outer(x):
    def inner(y):
        return x + y
    return inner

add_5 = outer(5)

# 查看闭包变量
print(f"闭包变量: {add_5.__closure__}")
print(f"变量值: {add_5.__closure__[0].cell_contents}")

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. LEGB 作用域规则
2. global 和 nonlocal 关键字
3. 闭包的概念和结构
4. 闭包的实际应用
5. 循环中的闭包陷阱

➡️ 下一节：递归函数
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("作用域与闭包模块学习完成！")
    print("=" * 60)
