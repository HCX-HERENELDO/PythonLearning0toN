#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：网络编程基础
学习目标：掌握 Python 网络编程的基本概念和技术
PyCharm 技巧：学习网络请求的调试方法
============================================================================
"""

# ============================================================================
# 第一部分：HTTP 请求 - requests 库
# ============================================================================
"""
【概念讲解】
HTTP（超文本传输协议）是 Web 通信的基础。
requests 是 Python 最流行的 HTTP 库，简单易用。

安装：pip install requests
"""

import requests
import json

# ----------------------------------------------------------------------------
# GET 请求
# ----------------------------------------------------------------------------

# 基本 GET 请求
response = requests.get('https://httpbin.org/get')
print(f"状态码: {response.status_code}")
print(f"响应头: {dict(response.headers)[:100]}...")

# 带参数的 GET 请求
params = {
    'name': '张三',
    'age': 25
}
response = requests.get('https://httpbin.org/get', params=params)
print(f"请求URL: {response.url}")

# 解析 JSON 响应
data = response.json()
print(f"响应数据: {data['args']}")

# ----------------------------------------------------------------------------
# POST 请求
# ----------------------------------------------------------------------------

# 表单数据
form_data = {
    'username': 'admin',
    'password': '123456'
}
response = requests.post('https://httpbin.org/post', data=form_data)
print(f"表单提交: {response.json()['form']}")

# JSON 数据
json_data = {
    'name': '张三',
    'email': 'zhangsan@example.com'
}
response = requests.post(
    'https://httpbin.org/post',
    json=json_data
)
print(f"JSON提交: {response.json()['json']}")

# ----------------------------------------------------------------------------
# 请求头和认证
# ----------------------------------------------------------------------------

headers = {
    'User-Agent': 'Python-Learning/1.0',
    'Accept': 'application/json',
    'Authorization': 'Bearer your-token-here'
}

response = requests.get(
    'https://httpbin.org/headers',
    headers=headers
)
print(f"请求头: {response.json()['headers']}")

# ----------------------------------------------------------------------------
# 处理响应
# ----------------------------------------------------------------------------

response = requests.get('https://httpbin.org/html')

# 获取文本内容
print(f"文本内容（前100字符）: {response.text[:100]}")

# 获取二进制内容（用于下载文件）
# content = response.content

# 获取编码
print(f"编码: {response.encoding}")

# 检查请求是否成功
if response.ok:
    print("请求成功")
elif response.status_code == 404:
    print("资源不存在")
elif response.status_code == 500:
    print("服务器错误")

# ----------------------------------------------------------------------------
# 超时和异常处理
# ----------------------------------------------------------------------------

try:
    response = requests.get(
        'https://httpbin.org/delay/1',
        timeout=3  # 3秒超时
    )
except requests.Timeout:
    print("请求超时")
except requests.ConnectionError:
    print("连接错误")
except requests.RequestException as e:
    print(f"请求异常: {e}")

# ============================================================================
# 第二部分：会话管理
# ============================================================================

# ----------------------------------------------------------------------------
# 使用 Session 保持会话
# ----------------------------------------------------------------------------

# 创建会话
session = requests.Session()

# 设置会话级别的请求头
session.headers.update({
    'User-Agent': 'Python-Learning/1.0'
})

# 使用会话发送请求（会保持 cookies）
response = session.get('https://httpbin.org/cookies/set/session_cookie/test123')
print(f"Cookies: {session.cookies.get_dict()}")

# 后续请求会携带 cookies
response = session.get('https://httpbin.org/cookies')
print(f"后续请求的Cookies: {response.json()['cookies']}")

# 关闭会话
session.close()

# 使用 with 语句自动管理
with requests.Session() as s:
    s.get('https://httpbin.org/get')

# ============================================================================
# 第三部分：API 调用实战
# ============================================================================

# ----------------------------------------------------------------------------
# 调用 REST API
# ----------------------------------------------------------------------------

def get_weather(city="Beijing"):
    """获取天气信息（示例API）"""
    # 使用免费的天气 API
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 检查HTTP错误
        
        data = response.json()
        
        # 解析天气数据
        current = data['current_condition'][0]
        weather_info = {
            '温度': f"{current['temp_C']}°C",
            '天气': current['weatherDesc'][0]['value'],
            '湿度': f"{current['humidity']}%",
            '风速': f"{current['windspeedKmph']} km/h"
        }
        
        return weather_info
        
    except requests.RequestException as e:
        print(f"获取天气失败: {e}")
        return None

# 测试
weather = get_weather("Beijing")
if weather:
    print("天气信息:")
    for key, value in weather.items():
        print(f"  {key}: {value}")

# ----------------------------------------------------------------------------
# 分页获取数据
# ----------------------------------------------------------------------------

def fetch_all_posts():
    """获取所有文章（分页示例）"""
    base_url = "https://jsonplaceholder.typicode.com/posts"
    all_posts = []
    page = 1
    per_page = 10
    
    while True:
        params = {
            '_page': page,
            '_limit': per_page
        }
        
        response = requests.get(base_url, params=params)
        posts = response.json()
        
        if not posts:  # 没有更多数据
            break
            
        all_posts.extend(posts)
        print(f"已获取 {len(all_posts)} 篇文章")
        
        page += 1
        
        # 示例只获取前30篇
        if len(all_posts) >= 30:
            break
    
    return all_posts

# posts = fetch_all_posts()
# print(f"总共获取 {len(posts)} 篇文章")

# ============================================================================
# 第四部分：异步请求（aiohttp）
# ============================================================================
"""
【概念讲解】
当需要并发发送大量请求时，使用异步请求可以显著提高效率。
aiohttp 是一个异步 HTTP 客户端库。

安装：pip install aiohttp
"""

# ----------------------------------------------------------------------------
# 异步请求示例
# ----------------------------------------------------------------------------

import asyncio
import aiohttp

async def fetch_url(session, url):
    """异步获取单个URL"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            return await response.json()
    except Exception as e:
        print(f"获取 {url} 失败: {e}")
        return None

async def fetch_multiple_urls(urls):
    """并发获取多个URL"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# 运行异步代码
async def main():
    urls = [
        'https://jsonplaceholder.typicode.com/posts/1',
        'https://jsonplaceholder.typicode.com/posts/2',
        'https://jsonplaceholder.typicode.com/posts/3',
    ]
    
    results = await fetch_multiple_urls(urls)
    for result in results:
        if result:
            print(f"文章: {result['title'][:30]}...")

# 运行（取消注释测试）
# asyncio.run(main())

# ============================================================================
# 第五部分：网页爬虫基础
# ============================================================================

# ----------------------------------------------------------------------------
# 使用 BeautifulSoup 解析网页
# ----------------------------------------------------------------------------

from bs4 import BeautifulSoup

def scrape_quotes():
    """爬取名言网站"""
    url = "https://quotes.toscrape.com/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到所有名言
        quotes = soup.find_all('div', class_='quote')
        
        results = []
        for quote in quotes:
            text = quote.find('span', class_='text').text
            author = quote.find('small', class_='author').text
            tags = [tag.text for tag in quote.find_all('a', class_='tag')]
            
            results.append({
                'text': text,
                'author': author,
                'tags': tags
            })
        
        return results
        
    except Exception as e:
        print(f"爬取失败: {e}")
        return []

# 测试爬虫
quotes = scrape_quotes()
print(f"\n爬取到 {len(quotes)} 条名言:")
for i, quote in enumerate(quotes[:3], 1):
    print(f"{i}. {quote['author']}: {quote['text'][:50]}...")

# ============================================================================
# 第六部分：网络编程最佳实践
# ============================================================================

# ----------------------------------------------------------------------------
# 封装请求类
# ----------------------------------------------------------------------------

class APIClient:
    """API 客户端封装"""
    
    def __init__(self, base_url, timeout=30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _request(self, method, endpoint, **kwargs):
        """发送请求"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            raise
    
    def get(self, endpoint, params=None):
        """GET 请求"""
        return self._request('GET', endpoint, params=params)
    
    def post(self, endpoint, data=None, json=None):
        """POST 请求"""
        return self._request('POST', endpoint, data=data, json=json)
    
    def close(self):
        """关闭会话"""
        self.session.close()

# 使用封装的客户端
client = APIClient('https://jsonplaceholder.typicode.com')
try:
    posts = client.get('/posts', params={'_limit': 5})
    print(f"\n获取 {len(posts)} 篇文章")
finally:
    client.close()

# ============================================================================
# 练习题
# ============================================================================
"""
【练习1】HTTP请求
1. 编写函数获取指定城市的天气信息
2. 实现一个简单的短链接服务调用
3. 批量下载图片并保存到本地

【练习2】API封装
1. 封装一个 GitHub API 客户端
2. 实现获取用户信息、仓库列表等功能
3. 处理分页请求

【练习3】爬虫实践
1. 爬取一个新闻网站的文章标题
2. 将数据保存到 JSON 文件
3. 处理反爬虫（设置 User-Agent、延迟请求）
"""

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. requests 库的基本使用
2. GET/POST 请求
3. 请求头、参数、JSON处理
4. 会话管理
5. 异步请求（aiohttp）
6. 网页爬虫基础
7. API 客户端封装

🔧 PyCharm 技巧：
1. HTTP Client 插件测试 API
2. 断点调试网络请求
3. 使用 Variables 面板查看响应数据

➡️ 下一节：Socket 编程
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("网络编程基础模块学习完成！")
    print("=" * 60)
