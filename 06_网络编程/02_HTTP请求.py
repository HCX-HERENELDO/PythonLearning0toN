#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：HTTP请求
学习目标：使用 requests 库发送各种 HTTP 请求
PyCharm 技巧：学习查看网络请求的响应数据
============================================================================
"""

# 需要安装: pip install requests
import requests
import json

# ============================================================================
# 第一部分：基础 GET 请求
# ============================================================================
"""
【概念讲解】
HTTP GET 请求用于从服务器获取数据。
requests 库是最流行的 Python HTTP 客户端库。
"""

# ----------------------------------------------------------------------------
# 简单 GET 请求
# ----------------------------------------------------------------------------

# 发送 GET 请求
response = requests.get('https://httpbin.org/get')

# 查看响应状态码
print(f"状态码: {response.status_code}")

# 查看响应内容
print(f"响应内容类型: {type(response.text)}")

# 查看 JSON 数据
data = response.json()
print(f"JSON 数据: {json.dumps(data, indent=2, ensure_ascii=False)}")

# ----------------------------------------------------------------------------
# 带参数的 GET 请求
# ----------------------------------------------------------------------------

# 方式1：直接拼接 URL
response = requests.get('https://httpbin.org/get?name=张三&age=25')

# 方式2：使用 params 参数（推荐）
params = {
    'name': '张三',
    'age': 25,
    'city': '北京'
}
response = requests.get('https://httpbin.org/get', params=params)
print(f"请求 URL: {response.url}")

# ----------------------------------------------------------------------------
# 添加请求头
# ----------------------------------------------------------------------------

headers = {
    'User-Agent': 'PythonLearning/1.0',
    'Accept': 'application/json',
    'Authorization': 'Bearer your-token-here'
}

response = requests.get(
    'https://httpbin.org/headers',
    headers=headers
)
print(f"请求头响应: {response.json()}")

# ============================================================================
# 第二部分：POST 请求
# ============================================================================

# ----------------------------------------------------------------------------
# 表单数据
# ----------------------------------------------------------------------------

form_data = {
    'username': 'test_user',
    'password': '123456'
}

response = requests.post(
    'https://httpbin.org/post',
    data=form_data
)
print(f"表单提交响应: {response.status_code}")

# ----------------------------------------------------------------------------
# JSON 数据
# ----------------------------------------------------------------------------

json_data = {
    'name': '张三',
    'email': 'zhangsan@example.com',
    'age': 25
}

response = requests.post(
    'https://httpbin.org/post',
    json=json_data
)
print(f"JSON 提交响应: {response.json()}")

# ----------------------------------------------------------------------------
# 文件上传
# ----------------------------------------------------------------------------

# 模拟文件上传
files = {
    'file': ('test.txt', '这是文件内容', 'text/plain')
}

response = requests.post(
    'https://httpbin.org/post',
    files=files
)
print(f"文件上传响应: {response.status_code}")

# ============================================================================
# 第三部分：其他 HTTP 方法
# ============================================================================

# PUT - 更新资源
put_data = {'name': '更新后的名称'}
response = requests.put('https://httpbin.org/put', json=put_data)
print(f"PUT 响应: {response.status_code}")

# DELETE - 删除资源
response = requests.delete('https://httpbin.org/delete')
print(f"DELETE 响应: {response.status_code}")

# PATCH - 部分更新
response = requests.patch('https://httpbin.org/patch', json={'field': 'value'})
print(f"PATCH 响应: {response.status_code}")

# HEAD - 只获取响应头
response = requests.head('https://httpbin.org/get')
print(f"HEAD 响应头: {dict(response.headers)}")

# ============================================================================
# 第四部分：响应处理
# ============================================================================

response = requests.get('https://httpbin.org/get')

# 状态码
print(f"状态码: {response.status_code}")
print(f"状态描述: {response.reason}")

# 判断请求是否成功
if response.ok:  # status_code < 400
    print("请求成功")

if response.status_code == 200:
    print("OK")

# 响应内容
print(f"文本内容: {response.text[:100]}...")  # 前100字符
print(f"字节内容: {response.content[:100]}...")  # 字节格式
print(f"JSON 内容: {response.json()}")

# 响应头
print(f"Content-Type: {response.headers.get('Content-Type')}")

# Cookie
print(f"Cookies: {response.cookies}")

# ============================================================================
# 第五部分：会话管理
# ============================================================================

# 创建会话（保持 Cookie 和连接）
session = requests.Session()

# 设置会话级别的请求头
session.headers.update({
    'User-Agent': 'PythonLearning/1.0'
})

# 使用会话发送请求
response = session.get('https://httpbin.org/cookies/set?name=value')
print(f"设置 Cookie: {response.cookies}")

response = session.get('https://httpbin.org/cookies')
print(f"当前 Cookies: {response.json()}")

# 关闭会话
session.close()

# 使用 with 语句（推荐）
with requests.Session() as s:
    s.get('https://httpbin.org/get')

# ============================================================================
# 第六部分：超时和异常处理
# ============================================================================

# ----------------------------------------------------------------------------
# 设置超时
# ----------------------------------------------------------------------------

# 连接超时和读取超时
try:
    response = requests.get(
        'https://httpbin.org/delay/2',
        timeout=(3, 5)  # (连接超时, 读取超时)
    )
    print(f"请求成功: {response.status_code}")
except requests.exceptions.Timeout:
    print("请求超时")

# 统一超时
try:
    response = requests.get(
        'https://httpbin.org/delay/2',
        timeout=5  # 总超时时间
    )
except requests.exceptions.Timeout:
    print("请求超时")

# ----------------------------------------------------------------------------
# 异常处理
# ----------------------------------------------------------------------------

def safe_request(url, **kwargs):
    """安全的 HTTP 请求函数"""
    try:
        response = requests.get(url, **kwargs)
        response.raise_for_status()  # 检查 HTTP 错误
        return response
    except requests.exceptions.Timeout:
        print("请求超时")
    except requests.exceptions.ConnectionError:
        print("连接错误")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
    return None

# ============================================================================
# 第七部分：实用封装
# ============================================================================

class HttpClient:
    """HTTP 客户端封装"""
    
    def __init__(self, base_url=None, default_headers=None):
        self.base_url = base_url or ''
        self.session = requests.Session()
        if default_headers:
            self.session.headers.update(default_headers)
    
    def _url(self, endpoint):
        """构建完整 URL"""
        return f"{self.base_url}{endpoint}"
    
    def get(self, endpoint, params=None):
        """GET 请求"""
        return self.session.get(self._url(endpoint), params=params)
    
    def post(self, endpoint, data=None, json=None):
        """POST 请求"""
        return self.session.post(self._url(endpoint), data=data, json=json)
    
    def put(self, endpoint, json=None):
        """PUT 请求"""
        return self.session.put(self._url(endpoint), json=json)
    
    def delete(self, endpoint):
        """DELETE 请求"""
        return self.session.delete(self._url(endpoint))
    
    def close(self):
        """关闭会话"""
        self.session.close()

# 使用示例
client = HttpClient(
    base_url='https://httpbin.org',
    default_headers={'User-Agent': 'PythonLearning/1.0'}
)

response = client.get('/get', params={'test': 'value'})
print(f"客户端请求: {response.status_code}")

client.close()

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. GET 和 POST 请求
2. 参数和请求头设置
3. 文件上传
4. 响应处理
5. 会话管理
6. 超时和异常处理

➡️ 下一节：REST API
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HTTP请求模块学习完成！")
    print("=" * 60)