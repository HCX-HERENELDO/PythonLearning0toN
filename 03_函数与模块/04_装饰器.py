#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：装饰器
学习目标：理解装饰器原理，掌握装饰器的定义和使用
PyCharm 技巧：学习使用调试器跟踪装饰器的执行过程
============================================================================
"""

# ============================================================================
# 第一部分：装饰器基础
# ============================================================================
"""
【概念讲解】
装饰器是一种特殊的函数，用于修改或增强其他函数的功能。
装饰器本质上是一个函数，它接受一个函数作为参数，返回一个新的函数。
装饰器遵循"开放-封闭"原则：不修改原函数代码，而是扩展其功能。
"""

# ----------------------------------------------------------------------------
# 理解装饰器的本质
# ----------------------------------------------------------------------------

# 装饰器就是一个接收函数并返回函数的函数
def my_decorator(func):
    """一个简单的装饰器"""
    def wrapper():
        print("函数执行前")
        func()
        print("函数执行后")
    return wrapper

# 使用装饰器
def say_hello():
    print("Hello!")

# 手动应用装饰器
say_hello = my_decorator(say_hello)
say_hello()

# 使用 @ 语法糖（推荐）
@my_decorator
def say_goodbye():
    print("Goodbye!")

say_goodbye()

# 【PyCharm 技巧】
# 在装饰器处设置断点，观察函数的包装过程
# 在被装饰函数处设置断点，观察执行顺序

# ============================================================================
# 第二部分：装饰器的参数处理
# ============================================================================

# ----------------------------------------------------------------------------
# 处理被装饰函数的参数
# ----------------------------------------------------------------------------

def decorator_with_args(func):
    """带参数的装饰器"""
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}，参数: args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"返回: {result}")
        return result
    return wrapper

@decorator_with_args
def add(a, b):
    return a + b

@decorator_with_args
def greet(name, greeting="你好"):
    return f"{greeting}, {name}!"

print(add(3, 5))
print(greet("张三", greeting="欢迎"))

# ----------------------------------------------------------------------------
# functools.wraps 保持原函数信息
# ----------------------------------------------------------------------------

from functools import wraps

def bad_decorator(func):
    """不使用 wraps 的装饰器"""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def good_decorator(func):
    """使用 wraps 的装饰器"""
    @wraps(func)  # 保持原函数的元信息
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@bad_decorator
def func1():
    """这是 func1"""
    pass

@good_decorator
def func2():
    """这是 func2"""
    pass

print(f"func1 名称: {func1.__name__}")  # wrapper
print(f"func1 文档: {func1.__doc__}")   # None
print(f"func2 名称: {func2.__name__}")  # func2
print(f"func2 文档: {func2.__doc__}")   # 这是 func2

# ============================================================================
# 第三部分：带参数的装饰器
# ============================================================================

# ----------------------------------------------------------------------------
# 三层嵌套实现
# ----------------------------------------------------------------------------

def repeat(times):
    """重复执行装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(times=3)
def say_hi(name):
    return f"Hi, {name}!"

print(say_hi("张三"))

# ----------------------------------------------------------------------------
# 可选参数的装饰器
# ----------------------------------------------------------------------------

def smart_decorator(_func=None, *, option="default"):
    """支持有无参数的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"选项: {option}")
            return func(*args, **kwargs)
        return wrapper
    
    if _func is None:
        # 带参数调用 @decorator(option="value")
        return decorator
    else:
        # 无参数调用 @decorator
        return decorator(_func)

@smart_decorator
def func1():
    print("func1")

@smart_decorator(option="custom")
def func2():
    print("func2")

func1()
func2()

# ============================================================================
# 第四部分：常用装饰器示例
# ============================================================================

# ----------------------------------------------------------------------------
# 计时装饰器
# ----------------------------------------------------------------------------

import time
from functools import wraps

def timer(func):
    """计算函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

slow_function()

# ----------------------------------------------------------------------------
# 日志装饰器
# ----------------------------------------------------------------------------

def log(func):
    """记录函数调用日志"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] 调用 {func.__name__}")
        print(f"[LOG] 参数: {args}, {kwargs}")
        try:
            result = func(*args, **kwargs)
            print(f"[LOG] 返回: {result}")
            return result
        except Exception as e:
            print(f"[LOG] 异常: {e}")
            raise
    return wrapper

@log
def divide(a, b):
    return a / b

divide(10, 2)
# divide(10, 0)  # 记录异常

# ----------------------------------------------------------------------------
# 缓存装饰器
# ----------------------------------------------------------------------------

def memoize(func):
    """简单的缓存装饰器"""
    cache = {}
    
    @wraps(func)
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
    """计算斐波那契数"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))

# Python 内置缓存装饰器
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

print(fibonacci_cached(50))

# ----------------------------------------------------------------------------
# 重试装饰器
# ----------------------------------------------------------------------------

import random

def retry(times=3, delay=1):
    """失败重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == times - 1:
                        raise
                    print(f"第{attempt + 1}次失败: {e}，{delay}秒后重试")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(times=3, delay=0.5)
def unstable_function():
    """模拟不稳定函数"""
    if random.random() < 0.7:
        raise ConnectionError("连接失败")
    return "成功"

# print(unstable_function())

# ----------------------------------------------------------------------------
# 权限验证装饰器
# ----------------------------------------------------------------------------

def require_auth(role="user"):
    """权限验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = kwargs.get("user", {})
            if not user.get("is_authenticated"):
                raise PermissionError("请先登录")
            if role and user.get("role") != role:
                raise PermissionError(f"需要 {role} 权限")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@require_auth(role="admin")
def delete_user(user, user_id):
    print(f"删除用户 {user_id}")

# delete_user(user={"is_authenticated": True, "role": "user"}, user_id=123)

# ============================================================================
# 第五部分：类装饰器
# ============================================================================

# ----------------------------------------------------------------------------
# 用类实现装饰器
# ----------------------------------------------------------------------------

class CountCalls:
    """统计函数调用次数的类装饰器"""
    
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"第 {self.count} 次调用 {self.func.__name__}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()
say_hello()
say_hello()

# ----------------------------------------------------------------------------
# 装饰器类（为类添加功能）
# ----------------------------------------------------------------------------

def add_method(cls):
    """为类添加方法的装饰器"""
    def new_method(self):
        return "这是添加的新方法"
    
    cls.new_method = new_method
    return cls

@add_method
class MyClass:
    pass

obj = MyClass()
print(obj.new_method())

# ============================================================================
# 第六部分：装饰器叠加
# ============================================================================

# 多个装饰器的执行顺序（从下到上装饰，从上到下执行）

@decorator_with_args
@timer
def complex_function():
    """多个装饰器叠加"""
    time.sleep(0.1)
    return "完成"

# 等价于:
# complex_function = decorator_with_args(timer(complex_function))

# ============================================================================
# 练习题
# ============================================================================
"""
【练习1】编写装饰器
1. 编写一个打印函数执行时间的装饰器
2. 编写一个验证函数参数类型的装饰器
3. 编写一个限制函数调用频率的装饰器（如每秒最多调用一次）

【练习2】应用装饰器
1. 使用装饰器实现单例模式
2. 使用装饰器实现属性延迟计算
3. 使用装饰器实现函数结果缓存

【练习3】综合应用
1. 为一个 API 请求函数添加日志、重试、超时功能
2. 实现一个路由注册装饰器（类似 Flask 的 @app.route）
"""

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. 装饰器的本质和原理
2. @ 语法糖的使用
3. 带参数的装饰器
4. functools.wraps 保持函数信息
5. 常用装饰器模式（计时、日志、缓存、重试）
6. 类装饰器
7. 装饰器叠加

🔧 PyCharm 技巧：
1. 断点调试观察装饰器执行顺序
2. Ctrl+点击查看装饰器定义
3. 使用 Structure 面板查看装饰后的函数

➡️ 下一节：模块与包
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("装饰器模块学习完成！")
    print("=" * 60)
