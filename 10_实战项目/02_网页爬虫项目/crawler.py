#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
项目名称：网页爬虫项目
项目描述：爬取网站数据并进行分析存储
学习目标：综合运用网络编程、数据处理知识
============================================================================
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

# ============================================================================
# 项目结构
# ============================================================================
"""
02_网页爬虫项目/
├── crawler.py          # 主程序
├── config.py           # 配置文件
├── models.py           # 数据模型
├── storage.py          # 数据存储
└── data/               # 数据目录
    ├── quotes.json
    └── quotes.csv
"""

# ============================================================================
# 配置
# ============================================================================

class Config:
    """爬虫配置"""
    
    # 目标网站
    BASE_URL = "https://quotes.toscrape.com"
    
    # 请求设置
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    TIMEOUT = 10
    DELAY = 0.5  # 请求间隔（秒）
    
    # 存储设置
    DATA_DIR = "data"
    JSON_FILE = "quotes.json"
    CSV_FILE = "quotes.csv"

# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class Quote:
    """名言数据模型"""
    text: str
    author: str
    tags: List[str]
    author_url: Optional[str] = None
    crawled_at: str = ""
    
    def __post_init__(self):
        if not self.crawled_at:
            self.crawled_at = datetime.now().isoformat()

# ============================================================================
# 爬虫类
# ============================================================================

class QuotesCrawler:
    """名言爬虫"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.session = requests.Session()
        self.session.headers.update(self.config.HEADERS)
        self.last_request_time = 0
        
        # 确保数据目录存在
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
    
    def _request(self, url: str) -> Optional[requests.Response]:
        """发送请求（带速率限制）"""
        # 速率限制
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.DELAY:
            time.sleep(self.config.DELAY - elapsed)
        
        self.last_request_time = time.time()
        
        try:
            response = self.session.get(
                url,
                timeout=self.config.TIMEOUT
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            return None
    
    def parse_quote(self, quote_element) -> Quote:
        """解析单个名言"""
        text = quote_element.find('span', class_='text').text
        author = quote_element.find('small', class_='author').text
        tags = [tag.text for tag in quote_element.find_all('a', class_='tag')]
        
        author_link = quote_element.find('a')
        author_url = None
        if author_link:
            author_url = self.config.BASE_URL + author_link.get('href', '')
        
        return Quote(
            text=text,
            author=author,
            tags=tags,
            author_url=author_url
        )
    
    def crawl_page(self, page: int = 1) -> List[Quote]:
        """爬取单页"""
        url = f"{self.config.BASE_URL}/page/{page}/"
        print(f"正在爬取: {url}")
        
        response = self._request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        quote_elements = soup.find_all('div', class_='quote')
        
        quotes = [self.parse_quote(q) for q in quote_elements]
        print(f"  获取到 {len(quotes)} 条名言")
        
        return quotes
    
    def crawl_all(self, max_pages: int = 10) -> List[Quote]:
        """爬取所有页面"""
        all_quotes = []
        
        for page in range(1, max_pages + 1):
            quotes = self.crawl_page(page)
            if not quotes:
                print(f"第 {page} 页没有数据，停止爬取")
                break
            all_quotes.extend(quotes)
        
        print(f"\n总共爬取 {len(all_quotes)} 条名言")
        return all_quotes

# ============================================================================
# 数据存储
# ============================================================================

class DataStorage:
    """数据存储类"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_json(self, quotes: List[Quote], filename: str):
        """保存为 JSON"""
        filepath = os.path.join(self.data_dir, filename)
        data = [asdict(q) for q in quotes]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到 {filepath}")
    
    def save_csv(self, quotes: List[Quote], filename: str):
        """保存为 CSV"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'author', 'tags', 'author_url'])
            writer.writeheader()
            
            for quote in quotes:
                row = asdict(quote)
                row['tags'] = ', '.join(quote.tags)  # 列表转字符串
                writer.writerow(row)
        
        print(f"数据已保存到 {filepath}")
    
    def load_json(self, filename: str) -> List[dict]:
        """加载 JSON"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

# ============================================================================
# 数据分析
# ============================================================================

class QuoteAnalyzer:
    """名言分析器"""
    
    def __init__(self, quotes: List[Quote]):
        self.quotes = quotes
    
    def count_by_author(self) -> dict:
        """按作者统计"""
        counts = {}
        for quote in self.quotes:
            counts[quote.author] = counts.get(quote.author, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_all_tags(self) -> List[str]:
        """获取所有标签"""
        tags = []
        for quote in self.quotes:
            tags.extend(quote.tags)
        return list(set(tags))
    
    def count_by_tag(self) -> dict:
        """按标签统计"""
        counts = {}
        for quote in self.quotes:
            for tag in quote.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def search(self, keyword: str) -> List[Quote]:
        """搜索名言"""
        return [
            q for q in self.quotes
            if keyword.lower() in q.text.lower()
        ]
    
    def report(self):
        """生成报告"""
        print("\n" + "=" * 50)
        print("数据分析报告")
        print("=" * 50)
        
        print(f"\n总名言数: {len(self.quotes)}")
        print(f"作者数: {len(self.count_by_author())}")
        print(f"标签数: {len(self.get_all_tags())}")
        
        print("\n名言最多的作者 TOP 5:")
        for author, count in list(self.count_by_author().items())[:5]:
            print(f"  - {author}: {count} 条")
        
        print("\n最热门标签 TOP 5:")
        for tag, count in list(self.count_by_tag().items())[:5]:
            print(f"  - {tag}: {count} 次")

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("=" * 50)
    print("名言爬虫项目")
    print("=" * 50)
    
    # 创建爬虫
    crawler = QuotesCrawler(Config())
    
    # 爬取数据
    quotes = crawler.crawl_all(max_pages=5)
    
    # 保存数据
    storage = DataStorage(Config.DATA_DIR)
    storage.save_json(quotes, Config.JSON_FILE)
    storage.save_csv(quotes, Config.CSV_FILE)
    
    # 分析数据
    analyzer = QuoteAnalyzer(quotes)
    analyzer.report()
    
    # 搜索示例
    print("\n搜索 'life' 的名言:")
    results = analyzer.search('life')
    for quote in results[:3]:
        print(f"  - {quote.author}: {quote.text[:50]}...")

if __name__ == "__main__":
    main()