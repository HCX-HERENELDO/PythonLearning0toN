#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
项目名称：简单博客系统
项目目标：综合运用 Flask + SQLAlchemy 创建 Web 应用
学习要点：
    - Flask Web 框架基础
    - SQLAlchemy ORM
    - RESTful API 设计
    - 前后端交互
PyCharm 技巧：HTTP Client 测试 API
============================================================================
"""

from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

# ============================================================================
# 应用配置
# ============================================================================

app = Flask(__name__)

# 数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'

db = SQLAlchemy(app)

# ============================================================================
# 数据模型
# ============================================================================

class Post(db.Model):
    """文章模型"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def __repr__(self):
        return f'<Post {self.title}>'


class Comment(db.Model):
    """评论模型"""
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # 关联文章
    post = db.relationship('Post', backref=db.backref('comments', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'post_id': self.post_id,
            'content': self.content,
            'author': self.author,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# ============================================================================
# HTML 模板
# ============================================================================

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>我的博客</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
               line-height: 1.6; background: #f5f5f5; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; margin-bottom: 20px; }
        .post { background: white; padding: 20px; margin-bottom: 20px; 
                border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .post h2 { color: #2c3e50; margin-bottom: 10px; }
        .post-meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 15px; }
        .post-content { color: #34495e; }
        .btn { display: inline-block; padding: 8px 16px; background: #3498db; 
               color: white; text-decoration: none; border-radius: 4px; }
        .btn:hover { background: #2980b9; }
        .header { display: flex; justify-content: space-between; align-items: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📝 我的博客</h1>
            <a href="/new" class="btn">写文章</a>
        </div>
        
        {% for post in posts %}
        <article class="post">
            <h2>{{ post.title }}</h2>
            <div class="post-meta">
                作者: {{ post.author }} | 
                发布时间: {{ post.created_at.strftime('%Y-%m-%d %H:%M') }}
            </div>
            <div class="post-content">
                {{ post.content[:200] }}{% if post.content|length > 200 %}...{% endif %}
            </div>
            <p><a href="/post/{{ post.id }}">阅读全文</a></p>
        </article>
        {% endfor %}
        
        {% if not posts %}
        <p style="text-align: center; color: #7f8c8d;">暂无文章，快来写一篇吧！</p>
        {% endif %}
    </div>
</body>
</html>
"""

NEW_POST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>写文章 - 我的博客</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               line-height: 1.6; background: #f5f5f5; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #333; }
        input[type="text"], textarea { width: 100%; padding: 10px; border: 1px solid #ddd;
                border-radius: 4px; font-size: 16px; }
        textarea { min-height: 300px; resize: vertical; }
        .btn { padding: 10px 20px; background: #3498db; color: white; border: none;
               border-radius: 4px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #2980b9; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #3498db; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← 返回首页</a>
        <h1>写文章</h1>
        
        <form method="POST" action="/api/posts">
            <div class="form-group">
                <label>标题</label>
                <input type="text" name="title" required placeholder="请输入标题">
            </div>
            <div class="form-group">
                <label>作者</label>
                <input type="text" name="author" required placeholder="请输入作者名">
            </div>
            <div class="form-group">
                <label>内容</label>
                <textarea name="content" required placeholder="请输入文章内容"></textarea>
            </div>
            <button type="submit" class="btn">发布文章</button>
        </form>
    </div>
</body>
</html>
"""

POST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ post.title }} - 我的博客</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               line-height: 1.6; background: #f5f5f5; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .post { background: white; padding: 30px; border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; margin-bottom: 15px; }
        .post-meta { color: #7f8c8d; margin-bottom: 25px; padding-bottom: 15px;
                     border-bottom: 1px solid #eee; }
        .post-content { color: #34495e; white-space: pre-wrap; }
        .comments { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }
        .comment { background: #f9f9f9; padding: 15px; margin-bottom: 15px; border-radius: 4px; }
        .comment-meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 5px; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #3498db; }
        .form-group { margin-bottom: 10px; }
        input[type="text"], textarea { width: 100%; padding: 8px; border: 1px solid #ddd;
                border-radius: 4px; }
        textarea { min-height: 80px; }
        .btn { padding: 8px 16px; background: #3498db; color: white; border: none;
               border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← 返回首页</a>
        
        <article class="post">
            <h1>{{ post.title }}</h1>
            <div class="post-meta">
                作者: {{ post.author }} | 
                发布时间: {{ post.created_at.strftime('%Y-%m-%d %H:%M') }}
            </div>
            <div class="post-content">{{ post.content }}</div>
        </article>
        
        <div class="comments">
            <h3>评论 ({{ comments|length }})</h3>
            
            {% for comment in comments %}
            <div class="comment">
                <div class="comment-meta">
                    {{ comment.author }} - {{ comment.created_at.strftime('%Y-%m-%d %H:%M') }}
                </div>
                <div>{{ comment.content }}</div>
            </div>
            {% endfor %}
            
            <h4>发表评论</h4>
            <form method="POST" action="/api/posts/{{ post.id }}/comments">
                <div class="form-group">
                    <input type="text" name="author" placeholder="昵称" required>
                </div>
                <div class="form-group">
                    <textarea name="content" placeholder="评论内容" required></textarea>
                </div>
                <button type="submit" class="btn">发表评论</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

# ============================================================================
# 路由 - 页面
# ============================================================================

@app.route('/')
def home():
    """首页"""
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template_string(HOME_TEMPLATE, posts=posts)

@app.route('/new')
def new_post():
    """写文章页面"""
    return render_template_string(NEW_POST_TEMPLATE)

@app.route('/post/<int:post_id>')
def view_post(post_id):
    """文章详情页"""
    post = Post.query.get_or_404(post_id)
    comments = Comment.query.filter_by(post_id=post_id).order_by(Comment.created_at).all()
    return render_template_string(POST_TEMPLATE, post=post, comments=comments)

# ============================================================================
# 路由 - API
# ============================================================================

@app.route('/api/posts', methods=['GET', 'POST'])
def api_posts():
    """文章 API"""
    if request.method == 'GET':
        # 获取文章列表
        posts = Post.query.order_by(Post.created_at.desc()).all()
        return jsonify([post.to_dict() for post in posts])
    
    elif request.method == 'POST':
        # 创建文章
        data = request.form if request.form else request.get_json()
        
        post = Post(
            title=data.get('title'),
            content=data.get('content'),
            author=data.get('author')
        )
        
        db.session.add(post)
        db.session.commit()
        
        # 判断是表单提交还是 API 请求
        if request.form:
            return f'''
            <script>
                alert('文章发布成功！');
                window.location.href = '/';
            </script>
            '''
        
        return jsonify(post.to_dict()), 201

@app.route('/api/posts/<int:post_id>', methods=['GET', 'PUT', 'DELETE'])
def api_post(post_id):
    """单个文章 API"""
    post = Post.query.get_or_404(post_id)
    
    if request.method == 'GET':
        return jsonify(post.to_dict())
    
    elif request.method == 'PUT':
        data = request.get_json()
        post.title = data.get('title', post.title)
        post.content = data.get('content', post.content)
        db.session.commit()
        return jsonify(post.to_dict())
    
    elif request.method == 'DELETE':
        db.session.delete(post)
        db.session.commit()
        return '', 204

@app.route('/api/posts/<int:post_id>/comments', methods=['GET', 'POST'])
def api_comments(post_id):
    """评论 API"""
    if request.method == 'GET':
        comments = Comment.query.filter_by(post_id=post_id).all()
        return jsonify([c.to_dict() for c in comments])
    
    elif request.method == 'POST':
        data = request.form if request.form else request.get_json()
        
        comment = Comment(
            post_id=post_id,
            content=data.get('content'),
            author=data.get('author')
        )
        
        db.session.add(comment)
        db.session.commit()
        
        if request.form:
            return f'''
            <script>
                alert('评论成功！');
                window.location.href = '/post/{post_id}';
            </script>
            '''
        
        return jsonify(comment.to_dict()), 201

# ============================================================================
# 初始化数据库
# ============================================================================

def init_db():
    """初始化数据库"""
    db.create_all()
    
    # 添加示例数据
    if Post.query.count() == 0:
        sample_posts = [
            Post(
                title='欢迎来到我的博客',
                content='这是我的第一篇博客文章。\n\n我正在学习 Python 和 Flask，这是一个练手项目。\n\n希望大家喜欢！',
                author='博主'
            ),
            Post(
                title='Python 学习笔记',
                content='今天学习了 Python 的基础知识：\n\n1. 变量和数据类型\n2. 条件语句和循环\n3. 函数定义\n\nPython 真是一门优雅的语言！',
                author='博主'
            ),
        ]
        
        for post in sample_posts:
            db.session.add(post)
        
        db.session.commit()
        print("✓ 示例数据已添加")

# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    with app.app_context():
        init_db()
    
    print("\n" + "=" * 50)
    print("博客系统启动成功！")
    print("=" * 50)
    print("访问地址: http://127.0.0.1:5000")
    print("API 文档: http://127.0.0.1:5000/api/posts")
    print("=" * 50 + "\n")
    
    app.run(debug=True, port=5000)


# ============================================================================
# API 测试说明
# ============================================================================
"""
使用 PyCharm HTTP Client 测试 API：

### 获取所有文章
GET http://127.0.0.1:5000/api/posts

### 创建文章
POST http://127.0.0.1:5000/api/posts
Content-Type: application/json

{
    "title": "测试文章",
    "content": "这是测试内容",
    "author": "测试用户"
}

### 获取单篇文章
GET http://127.0.0.1:5000/api/posts/1

### 更新文章
PUT http://127.0.0.1:5000/api/posts/1
Content-Type: application/json

{
    "title": "更新后的标题"
}

### 删除文章
DELETE http://127.0.0.1:5000/api/posts/1
"""
