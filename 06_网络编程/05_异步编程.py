#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：异步编程
学习目标：掌握 asyncio 异步编程基础
PyCharm 技巧：学习调试异步代码
============================================================================
"""

import asyncio
import time

# ============================================================================
# 第一部分：异步编程概念
# ============================================================================
"""
【概念讲解】
异步编程允许程序在等待 I/O 操作时执行其他任务，
提高程序的并发性能。

同步 vs 异步：
- 同步：顺序执行，等待每个操作完成
- 异步：并发执行，不阻塞等待

Python 异步关键字：
- async：定义异步函数
- await：等待异步操作完成
"""

# ============================================================================
# 第二部分：协程基础
# ============================================================================

# ----------------------------------------------------------------------------
# 定义协程
# ----------------------------------------------------------------------------

async def say_hello(name, delay):
    """异步函数（协程）"""
    print(f"开始: {name}")
    await asyncio.sleep(delay)  # 模拟耗时操作
    print(f"结束: {name}")
    return f"{name} 完成"

# 运行协程
async def main():
    # 方式1：顺序执行
    print("=== 顺序执行 ===")
    result1 = await say_hello("任务1", 1)
    result2 = await say_hello("任务2", 1)
    print(f"结果: {result1}, {result2}")

# 运行
asyncio.run(main())

# ----------------------------------------------------------------------------
# 并发执行
# ----------------------------------------------------------------------------

async def main_concurrent():
    """并发执行"""
    print("\n=== 并发执行 ===")
    start = time.time()
    
    # 创建任务
    task1 = asyncio.create_task(say_hello("任务A", 1))
    task2 = asyncio.create_task(say_hello("任务B", 1))
    task3 = asyncio.create_task(say_hello("任务C", 1))
    
    # 等待所有任务完成
    results = await asyncio.gather(task1, task2, task3)
    
    elapsed = time.time() - start
    print(f"总耗时: {elapsed:.2f} 秒")  # 约1秒，而不是3秒
    print(f"结果: {results}")

asyncio.run(main_concurrent())

# ============================================================================
# 第三部分：asyncio 常用操作
# ============================================================================

# ----------------------------------------------------------------------------
# asyncio.gather - 并发运行多个协程
# ----------------------------------------------------------------------------

async def fetch_data(url, delay):
    """模拟获取数据"""
    await asyncio.sleep(delay)
    return f"数据来自 {url}"

async def demo_gather():
    """演示 gather"""
    urls = ["url1", "url2", "url3", "url4"]
    
    # 并发获取所有数据
    tasks = [fetch_data(url, 0.5) for url in urls]
    results = await asyncio.gather(*tasks)
    
    print(f"Gather 结果: {results}")

asyncio.run(demo_gather())

# ----------------------------------------------------------------------------
# asyncio.wait - 更灵活的等待
# ----------------------------------------------------------------------------

async def demo_wait():
    """演示 wait"""
    tasks = [
        asyncio.create_task(fetch_data(f"url{i}", i * 0.3))
        for i in range(1, 4)
    ]
    
    # 等待所有完成
    done, pending = await asyncio.wait(tasks)
    
    print(f"完成任务数: {len(done)}")
    for task in done:
        print(f"  结果: {task.result()}")

asyncio.run(demo_wait())

# ----------------------------------------------------------------------------
# asyncio.Timeout - 超时控制
# ----------------------------------------------------------------------------

async def demo_timeout():
    """演示超时"""
    try:
        result = await asyncio.wait_for(
            fetch_data("slow_url", 5),
            timeout=1.0
        )
        print(f"结果: {result}")
    except asyncio.TimeoutError:
        print("请求超时")

asyncio.run(demo_timeout())

# ============================================================================
# 第四部分：异步上下文管理器和迭代器
# ============================================================================

# ----------------------------------------------------------------------------
# 异步上下文管理器
# ----------------------------------------------------------------------------

class AsyncContextManager:
    """异步上下文管理器"""
    
    async def __aenter__(self):
        print("进入异步上下文")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("退出异步上下文")
        await asyncio.sleep(0.1)

async def demo_context():
    """演示异步上下文管理器"""
    async with AsyncContextManager() as manager:
        print("在异步上下文中执行")

asyncio.run(demo_context())

# ----------------------------------------------------------------------------
# 异步迭代器
# ----------------------------------------------------------------------------

class AsyncRange:
    """异步范围迭代器"""
    
    def __init__(self, count):
        self.count = count
    
    def __aiter__(self):
        self.current = 0
        return self
    
    async def __anext__(self):
        if self.current < self.count:
            await asyncio.sleep(0.1)
            value = self.current
            self.current += 1
            return value
        raise StopAsyncIteration

async def demo_iterator():
    """演示异步迭代器"""
    async for i in AsyncRange(5):
        print(f"异步迭代: {i}")

asyncio.run(demo_iterator())

# ============================================================================
# 第五部分：异步 HTTP 请求
# ============================================================================

# 需要安装: pip install aiohttp

async def fetch_with_aiohttp():
    """使用 aiohttp 进行异步 HTTP 请求"""
    try:
        import aiohttp
        
        urls = [
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/1",
        ]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                task = session.get(url)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            for response in responses:
                print(f"状态码: {response.status}")
        
    except ImportError:
        print("aiohttp 未安装，跳过此示例")

asyncio.run(fetch_with_aiohttp())

# ============================================================================
# 第六部分：异步编程最佳实践
# ============================================================================

"""
【最佳实践】

1. 不要在异步函数中使用阻塞操作
   ❌ time.sleep(1)
   ✅ await asyncio.sleep(1)

2. 使用 asyncio.run() 作为入口点
   在 Python 3.7+ 中推荐使用

3. 合理设置并发数量
   避免创建过多任务

4. 异常处理
   使用 try-except 或 gather(return_exceptions=True)

5. 资源管理
   使用 async with 管理资源
"""

async def demo_best_practices():
    """演示最佳实践"""
    
    # 异常处理
    async def may_fail(n):
        if n == 2:
            raise ValueError(f"错误: {n}")
        await asyncio.sleep(0.1)
        return n
    
    # 方式1：捕获异常
    tasks = [may_fail(i) for i in range(4)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务 {i} 失败: {result}")
        else:
            print(f"任务 {i} 成功: {result}")

asyncio.run(demo_best_practices())

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. 异步编程概念
2. async/await 语法
3. 协程的创建和运行
4. asyncio.gather 并发执行
5. 超时控制
6. 异步 HTTP 请求
7. 最佳实践

➡️ 下一节：数据库操作
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("异步编程模块学习完成！")
    print("=" * 60)
