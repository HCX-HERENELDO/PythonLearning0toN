#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：ORM基础
学习目标：掌握 SQLAlchemy ORM 的使用
PyCharm 技巧：学习数据库模型设计
============================================================================
"""

# 需要安装: pip install sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# ============================================================================
# 第一部分：ORM 概念
# ============================================================================
"""
【概念讲解】
ORM (Object-Relational Mapping) 对象关系映射
将数据库表映射为 Python 类，将记录映射为对象。

优势：
- 面向对象操作数据库
- 不需要写 SQL 语句
- 数据库无关性
- 代码更易维护
"""

# ============================================================================
# 第二部分：定义模型
# ============================================================================

# 创建基类
Base = declarative_base()

# ----------------------------------------------------------------------------
# 定义用户模型
# ----------------------------------------------------------------------------

class User(Base):
    """用户模型"""
    __tablename__ = 'users'
    
    # 字段定义
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, comment='用户名')
    email = Column(String(100), unique=True, comment='邮箱')
    age = Column(Integer, default=0, comment='年龄')
    created_at = Column(DateTime, default=datetime.now, comment='创建时间')
    
    # 关系（一个用户有多篇文章）
    articles = relationship("Article", back_populates="author")
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}')>"

# ----------------------------------------------------------------------------
# 定义文章模型
# ----------------------------------------------------------------------------

class Article(Base):
    """文章模型"""
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False, comment='标题')
    content = Column(String(5000), comment='内容')
    views = Column(Integer, default=0, comment='浏览量')
    
    # 外键
    author_id = Column(Integer, ForeignKey('users.id'), comment='作者ID')
    
    # 关系
    author = relationship("User", back_populates="articles")
    
    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title}')>"

# ============================================================================
# 第三部分：数据库连接
# ============================================================================

# 创建数据库引擎（使用 SQLite）
engine = create_engine('sqlite:///example.db', echo=True)

# 创建所有表
Base.metadata.create_all(engine)

# 创建会话工厂
Session = sessionmaker(bind=engine)

# ============================================================================
# 第四部分：CRUD 操作
# ============================================================================

# 创建会话
session = Session()

# ----------------------------------------------------------------------------
# Create - 创建记录
# ----------------------------------------------------------------------------

# 创建用户
user1 = User(name='张三', email='zhangsan@example.com', age=25)
user2 = User(name='李四', email='lisi@example.com', age=30)
user3 = User(name='王五', email='wangwu@example.com', age=28)

session.add_all([user1, user2, user3])
session.commit()

print(f"创建用户: {user1}, {user2}, {user3}")

# 创建文章
article1 = Article(title='Python入门', content='Python基础教程', author=user1)
article2 = Article(title='SQLAlchemy详解', content='ORM框架使用', author=user1)
article3 = Article(title='数据库设计', content='数据库设计原则', author=user2)

session.add_all([article1, article2, article3])
session.commit()

print(f"创建文章: {article1}, {article2}, {article3}")

# ----------------------------------------------------------------------------
# Read - 查询记录
# ----------------------------------------------------------------------------

# 查询所有用户
users = session.query(User).all()
print(f"所有用户: {users}")

# 条件查询
user = session.query(User).filter_by(name='张三').first()
print(f"查询张三: {user}")

# 多条件查询
users = session.query(User).filter(User.age >= 25, User.age < 30).all()
print(f"年龄25-30的用户: {users}")

# 排序
users = session.query(User).order_by(User.age.desc()).all()
print(f"按年龄降序: {[u.name for u in users]}")

# 分页
users = session.query(User).limit(2).offset(1).all()
print(f"分页查询: {[u.name for u in users]}")

# 统计
count = session.query(User).count()
print(f"用户总数: {count}")

# 关联查询
articles = session.query(Article).join(User).filter(User.name == '张三').all()
print(f"张三的文章: {[a.title for a in articles]}")

# ----------------------------------------------------------------------------
# Update - 更新记录
# ----------------------------------------------------------------------------

# 方式1：修改属性后提交
user = session.query(User).filter_by(name='张三').first()
user.age = 26
session.commit()
print(f"更新后年龄: {user.age}")

# 方式2：批量更新
session.query(User).filter(User.age == 28).update({'age': 29})
session.commit()

# ----------------------------------------------------------------------------
# Delete - 删除记录
# ----------------------------------------------------------------------------

# 删除单条
# user_to_delete = session.query(User).filter_by(name='王五').first()
# session.delete(user_to_delete)
# session.commit()

# 批量删除
# session.query(User).filter(User.age > 50).delete()
# session.commit()

# ============================================================================
# 第五部分：关系操作
# ============================================================================

# 访问关系属性
user = session.query(User).filter_by(name='张三').first()
print(f"张三的文章数: {len(user.articles)}")

for article in user.articles:
    print(f"  - {article.title}")

# 通过文章访问作者
article = session.query(Article).first()
print(f"文章作者: {article.author.name}")

# ============================================================================
# 第六部分：高级查询
# ============================================================================

from sqlalchemy import func, or_, and_, like

# 聚合函数
avg_age = session.query(func.avg(User.age)).scalar()
max_age = session.query(func.max(User.age)).scalar()
print(f"平均年龄: {avg_age}, 最大年龄: {max_age}")

# 分组统计
from sqlalchemy import func
result = session.query(
    User.age,
    func.count(User.id)
).group_by(User.age).all()
print(f"年龄分组统计: {result}")

# OR 条件
users = session.query(User).filter(
    or_(User.name == '张三', User.name == '李四')
).all()
print(f"张三或李四: {[u.name for u in users]}")

# LIKE 查询
users = session.query(User).filter(
    User.email.like('%example.com')
).all()
print(f"邮箱匹配: {[u.email for u in users]}")

# ============================================================================
# 第七部分：事务处理
# ============================================================================

try:
    # 开始事务
    user = User(name='测试用户', email='test@example.com')
    session.add(user)
    
    # 模拟错误
    # raise Exception("模拟错误")
    
    session.commit()
except Exception as e:
    session.rollback()
    print(f"事务回滚: {e}")

# ============================================================================
# 清理
# ============================================================================

session.close()

# 删除数据库文件
import os
if os.path.exists('example.db'):
    os.remove('example.db')

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. ORM 概念和优势
2. SQLAlchemy 模型定义
3. 一对多关系
4. CRUD 操作
5. 高级查询
6. 事务处理

➡️ 下一节：MongoDB
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ORM基础模块学习完成！")
    print("=" * 60)