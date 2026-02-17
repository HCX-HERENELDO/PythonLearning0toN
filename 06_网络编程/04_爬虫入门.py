#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：爬虫入门
学习目标：掌握网页数据抓取的基本方法
PyCharm 技巧：学习分析网页结构
============================================================================
"""

# 需要安装: pip install requests beautifulsoup4
import requests
from bs4 import BeautifulSoup
import json
import time
import re

# ============================================================================
# 第一部分：爬虫基础概念
# ============================================================================
"""
【概念讲解】
网络爬虫是一种自动抓取网页数据的程序。

基本步骤：
1. 发送 HTTP 请求获取网页
2. 解析 HTML 内容
3. 提取所需数据
4. 存储数据

注意事项：
- 遵守 robots.txt 规则
- 设置合理的请求间隔
- 添加 User-Agent 标识
- 不要对服务器造成压力
"""

# ============================================================================
# 第二部分：发送请求获取网页
# ============================================================================

# 模拟浏览器请求
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# 获取网页
url = "https://example.com"
response = requests.get(url, headers=headers)

# 检查响应
if response.status_code == 200:
    print("请求成功")
    html_content = response.text
else:
    print(f"请求失败: {response.status_code}")

# ============================================================================
# 第三部分：BeautifulSoup 解析
# ============================================================================

# 示例 HTML
sample_html = """
<!DOCTYPE html>
<html>
<head>
    <title>示例网页</title>
</head>
<body>
    <div class="container">
        <h1 id="main-title">欢迎学习爬虫</h1>
        <p class="description">这是一个示例网页</p>
        
        <ul class="list">
            <li class="item">项目1</li>
            <li class="item">项目2</li>
            <li class="item">项目3</li>
        </ul>
        
        <div class="products">
            <div class="product" data-id="1">
                <h3>商品A</h3>
                <span class="price">99.9</span>
            </div>
            <div class="product" data-id="2">
                <h3>商品B</h3>
                <span class="price">199.9</span>
            </div>
        </div>
        
        <a href="/page1">链接1</a>
        <a href="/page2">链接2</a>
    </div>
</body>
</html>
"""

# 创建 BeautifulSoup 对象
soup = BeautifulSoup(sample_html, 'html.parser')

# ----------------------------------------------------------------------------
# 查找元素
# ----------------------------------------------------------------------------

# 通过标签名查找
title = soup.find('title')
print(f"标题: {title.text}")

# 通过 id 查找
main_title = soup.find(id='main-title')
print(f"主标题: {main_title.text}")

# 通过 class 查找
description = soup.find(class_='description')
print(f"描述: {description.text}")

# 使用 select (CSS 选择器)
items = soup.select('li.item')
print(f"列表项数量: {len(items)}")
for item in items:
    print(f"  - {item.text}")

# ----------------------------------------------------------------------------
# 获取属性
# ----------------------------------------------------------------------------

# 获取链接
links = soup.find_all('a')
for link in links:
    href = link.get('href')
    text = link.text
    print(f"链接: {text} -> {href}")

# 获取 data 属性
products = soup.select('.product')
for product in products:
    data_id = product.get('data-id')
    name = product.find('h3').text
    price = product.find(class_='price').text
    print(f"商品 {data_id}: {name}, 价格: {price}")

# ============================================================================
# 第四部分：实际爬虫示例
# ============================================================================

def crawl_quotes():
    """爬取名言网站示例"""
    base_url = "https://quotes.toscrape.com"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    quotes_data = []
    
    # 只爬取第一页作为示例
    response = requests.get(base_url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找所有名言
        quotes = soup.find_all('div', class_='quote')
        
        for quote in quotes:
            text = quote.find('span', class_='text').text
            author = quote.find('small', class_='author').text
            tags = [tag.text for tag in quote.find_all('a', class_='tag')]
            
            quotes_data.append({
                'text': text,
                'author': author,
                'tags': tags
            })
    
    return quotes_data

# 爬取名言（需要网络）
try:
    quotes = crawl_quotes()
    print(f"爬取到 {len(quotes)} 条名言")
    if quotes:
        print(f"第一条: {quotes[0]['text'][:50]}...")
except Exception as e:
    print(f"爬取失败: {e}")

# ============================================================================
# 第五部分：数据存储
# ============================================================================

def save_to_json(data, filename):
    """保存到 JSON 文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存到 {filename}")

def save_to_csv(data, filename):
    """保存到 CSV 文件"""
    import csv
    
    if not data:
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"数据已保存到 {filename}")

# ============================================================================
# 第六部分：爬虫类封装
# ============================================================================

class WebScraper:
    """网页爬虫类"""
    
    def __init__(self, delay=1.0):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.delay = delay
        self.last_request = 0
    
    def get(self, url):
        """发送 GET 请求"""
        # 请求间隔控制
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        
        self.last_request = time.time()
        response = self.session.get(url)
        response.raise_for_status()
        return response
    
    def get_soup(self, url):
        """获取 BeautifulSoup 对象"""
        response = self.get(url)
        return BeautifulSoup(response.text, 'html.parser')
    
    def extract_text(self, soup, selector):
        """提取文本"""
        elements = soup.select(selector)
        return [e.text.strip() for e in elements]
    
    def extract_links(self, soup, selector='a'):
        """提取链接"""
        links = []
        for a in soup.select(selector):
            href = a.get('href')
            text = a.text.strip()
            if href:
                links.append({'text': text, 'href': href})
        return links

# ============================================================================
# 第七部分：爬虫注意事项
# ============================================================================

"""
【爬虫伦理与规范】

1. 检查 robots.txt
   访问 /robots.txt 查看网站爬虫规则

2. 设置 User-Agent
   标识你的爬虫身份

3. 控制请求频率
   避免对服务器造成压力

4. 尊重网站条款
   某些网站禁止爬虫

5. 数据使用规范
   不要侵犯版权和隐私
"""

def check_robots_txt(base_url):
    """检查 robots.txt"""
    robots_url = f"{base_url}/robots.txt"
    try:
        response = requests.get(robots_url)
        if response.status_code == 200:
            print(f"robots.txt 内容:\n{response.text[:500]}")
        else:
            print("未找到 robots.txt")
    except Exception as e:
        print(f"检查失败: {e}")

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. 爬虫基本概念
2. 发送 HTTP 请求
3. BeautifulSoup 解析
4. 元素查找和属性提取
5. 数据存储
6. 爬虫伦理规范

➡️ 下一节：异步编程
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("爬虫入门模块学习完成！")
    print("=" * 60)