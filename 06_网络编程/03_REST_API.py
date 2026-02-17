#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：REST API
学习目标：掌握 RESTful API 的调用和设计
PyCharm 技巧：学习 API 响应数据分析
============================================================================
"""

import requests
import json
from typing import Optional, Dict, Any

# ============================================================================
# 第一部分：REST API 基础概念
# ============================================================================
"""
【概念讲解】
REST (Representational State Transfer) 是一种 API 设计风格。

RESTful API 特点：
1. 使用 HTTP 动词表示操作（GET/POST/PUT/DELETE）
2. 使用 URL 表示资源
3. 无状态
4. 返回 JSON/XML 数据

HTTP 动词对应 CRUD：
- GET    → Read（读取）
- POST   → Create（创建）
- PUT    → Update（更新）
- DELETE → Delete（删除）
"""

# ============================================================================
# 第二部分：调用公开 API
# ============================================================================

# ----------------------------------------------------------------------------
# JSONPlaceholder（测试 API）
# ----------------------------------------------------------------------------

BASE_URL = "https://jsonplaceholder.typicode.com"

# 获取所有帖子（GET）
response = requests.get(f"{BASE_URL}/posts")
posts = response.json()
print(f"帖子数量: {len(posts)}")
print(f"第一个帖子: {posts[0]}")

# 获取单个帖子
response = requests.get(f"{BASE_URL}/posts/1")
post = response.json()
print(f"帖子详情: {post}")

# 获取帖子的评论
response = requests.get(f"{BASE_URL}/posts/1/comments")
comments = response.json()
print(f"评论数量: {len(comments)}")

# 创建帖子（POST）
new_post = {
    "title": "我的新帖子",
    "body": "这是帖子内容",
    "userId": 1
}
response = requests.post(f"{BASE_URL}/posts", json=new_post)
created_post = response.json()
print(f"创建的帖子 ID: {created_post['id']}")

# 更新帖子（PUT）
update_data = {
    "id": 1,
    "title": "更新后的标题",
    "body": "更新后的内容",
    "userId": 1
}
response = requests.put(f"{BASE_URL}/posts/1", json=update_data)
print(f"更新响应: {response.status_code}")

# 删除帖子（DELETE）
response = requests.delete(f"{BASE_URL}/posts/1")
print(f"删除响应: {response.status_code}")

# ============================================================================
# 第三部分：API 客户端封装
# ============================================================================

class APIClient:
    """REST API 客户端"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # 设置默认请求头
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        self.session.headers.update(headers)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """发送请求"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status_code": response.status_code}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """GET 请求"""
        return self._request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: Dict) -> Dict:
        """POST 请求"""
        return self._request('POST', endpoint, data=data)
    
    def put(self, endpoint: str, data: Dict) -> Dict:
        """PUT 请求"""
        return self._request('PUT', endpoint, data=data)
    
    def patch(self, endpoint: str, data: Dict) -> Dict:
        """PATCH 请求"""
        return self._request('PATCH', endpoint, data=data)
    
    def delete(self, endpoint: str) -> Dict:
        """DELETE 请求"""
        return self._request('DELETE', endpoint)

# 使用 API 客户端
client = APIClient("https://jsonplaceholder.typicode.com")

# 获取用户列表
users = client.get("/users")
print(f"用户数量: {len(users)}")

# 获取单个用户
user = client.get("/users/1")
print(f"用户名: {user.get('name')}")

# ============================================================================
# 第四部分：分页和过滤
# ============================================================================

# 分页参数
params = {
    '_page': 1,      # 页码
    '_limit': 5      # 每页数量
}
response = requests.get(f"{BASE_URL}/posts", params=params)
print(f"分页数据: {len(response.json())} 条")

# 排序
params = {'_sort': 'id', '_order': 'desc'}
response = requests.get(f"{BASE_URL}/posts", params=params)
posts = response.json()
print(f"排序后第一条 ID: {posts[0]['id']}")

# 过滤
params = {'userId': 1}
response = requests.get(f"{BASE_URL}/posts", params=params)
user_posts = response.json()
print(f"用户1的帖子: {len(user_posts)} 条")

# ============================================================================
# 第五部分：错误处理最佳实践
# ============================================================================

def call_api(url: str, method: str = 'GET', **kwargs) -> tuple:
    """
    调用 API 并返回 (成功标志, 数据/错误信息)
    """
    try:
        response = requests.request(method, url, **kwargs)
        
        # 检查 HTTP 状态码
        if response.status_code == 200:
            return True, response.json()
        elif response.status_code == 201:
            return True, response.json()
        elif response.status_code == 204:
            return True, None
        elif response.status_code == 400:
            return False, "请求参数错误"
        elif response.status_code == 401:
            return False, "未授权"
        elif response.status_code == 403:
            return False, "禁止访问"
        elif response.status_code == 404:
            return False, "资源不存在"
        elif response.status_code == 500:
            return False, "服务器错误"
        else:
            return False, f"未知错误: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "请求超时"
    except requests.exceptions.ConnectionError:
        return False, "连接错误"
    except requests.exceptions.RequestException as e:
        return False, f"请求异常: {e}"
    except json.JSONDecodeError:
        return False, "响应解析错误"

# 使用
success, result = call_api(f"{BASE_URL}/posts/1")
if success:
    print(f"获取成功: {result['title']}")
else:
    print(f"获取失败: {result}")

# ============================================================================
# 第六部分：速率限制
# ============================================================================

import time

class RateLimitedClient:
    """带速率限制的 API 客户端"""
    
    def __init__(self, base_url: str, requests_per_second: int = 5):
        self.base_url = base_url
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
    
    def get(self, endpoint: str, params: Optional[Dict] = None):
        """带速率限制的 GET 请求"""
        # 计算需要等待的时间
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_request_time = time.time()
        return requests.get(f"{self.base_url}{endpoint}", params=params)

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. REST API 概念
2. HTTP 动词与 CRUD 对应
3. API 客户端封装
4. 分页、排序、过滤
5. 错误处理
6. 速率限制

➡️ 下一节：爬虫入门
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REST API 模块学习完成！")
    print("=" * 60)
