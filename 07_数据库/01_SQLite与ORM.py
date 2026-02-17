#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：数据库基础
学习目标：掌握 Python 操作数据库的基本方法
PyCharm 技巧：学习数据库工具的使用
============================================================================
"""

# ============================================================================
# 第一部分：SQLite 数据库
# ============================================================================
"""
【概念讲解】
SQLite 是一个轻量级的嵌入式数据库，不需要服务器。
Python 内置 sqlite3 模块，开箱即用。
适合小型应用和学习使用。
"""

import sqlite3
from pathlib import Path

# ----------------------------------------------------------------------------
# 创建数据库和表
# ----------------------------------------------------------------------------

# 连接数据库（不存在则创建）
db_path = Path("example.db")
conn = sqlite3.connect(db_path)

# 创建游标
cursor = conn.cursor()

# 创建表
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# 提交更改
conn.commit()

print("数据库和表创建成功！")

# ----------------------------------------------------------------------------
# 插入数据
# ----------------------------------------------------------------------------

# 插入单条数据
cursor.execute('''
INSERT INTO users (name, email, age) VALUES (?, ?, ?)
''', ('张三', 'zhangsan@example.com', 25))

# 使用命名参数
cursor.execute('''
INSERT INTO users (name, email, age) VALUES (:name, :email, :age)
''', {'name': '李四', 'email': 'lisi@example.com', 'age': 30})

# 批量插入
users = [
    ('王五', 'wangwu@example.com', 28),
    ('赵六', 'zhaoliu@example.com', 22),
    ('钱七', 'qianqi@example.com', 35),
]

cursor.executemany('''
INSERT INTO users (name, email, age) VALUES (?, ?, ?)
''', users)

conn.commit()
print(f"插入了 {cursor.rowcount} 条记录")

# ----------------------------------------------------------------------------
# 查询数据
# ----------------------------------------------------------------------------

# 查询所有数据
cursor.execute('SELECT * FROM users')
all_users = cursor.fetchall()
print(f"所有用户: {len(all_users)} 条")

# 查询特定条件
cursor.execute('SELECT name, age FROM users WHERE age > ?', (25,))
adults = cursor.fetchall()
print(f"年龄>25的用户: {adults}")

# 获取一条记录
cursor.execute('SELECT * FROM users WHERE name = ?', ('张三',))
user = cursor.fetchone()
print(f"张三的信息: {user}")

# 获取指定数量
cursor.execute('SELECT * FROM users ORDER BY age DESC')
youngest = cursor.fetchmany(3)
print(f"最年轻的3人: {youngest}")

# ----------------------------------------------------------------------------
# 更新数据
# ----------------------------------------------------------------------------

cursor.execute('''
UPDATE users SET age = ? WHERE name = ?
''', (26, '张三'))

conn.commit()
print(f"更新了 {cursor.rowcount} 条记录")

# ----------------------------------------------------------------------------
# 删除数据
# ----------------------------------------------------------------------------

cursor.execute('DELETE FROM users WHERE name = ?', ('钱七',))
conn.commit()
print(f"删除了 {cursor.rowcount} 条记录")

# ----------------------------------------------------------------------------
# 使用上下文管理器
# ----------------------------------------------------------------------------

# 推荐使用 with 语句
with sqlite3.connect('example.db') as conn:
    conn.row_factory = sqlite3.Row  # 返回字典形式
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users')
    for row in cursor:
        print(f"用户: {row['name']}, 年龄: {row['age']}")

# ============================================================================
# 第二部分：SQLAlchemy ORM
# ============================================================================
"""
【概念讲解】
ORM（对象关系映射）将数据库表映射为 Python 类。
SQLAlchemy 是 Python 最流行的 ORM 库。

安装：pip install sqlalchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# ----------------------------------------------------------------------------
# 定义模型
# ----------------------------------------------------------------------------

Base = declarative_base()

class User(Base):
    """用户模型"""
    __tablename__ = 'orm_users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True)
    age = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<User(name='{self.name}', email='{self.email}')>"

# ----------------------------------------------------------------------------
# 创建表
# ----------------------------------------------------------------------------

# 创建引擎
engine = create_engine('sqlite:///orm_example.db', echo=False)

# 创建表
Base.metadata.create_all(engine)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()

# ----------------------------------------------------------------------------
# CRUD 操作
# ----------------------------------------------------------------------------

# 创建（Create）
user1 = User(name='张三', email='zhangsan@example.com', age=25)
user2 = User(name='李四', email='lisi@example.com', age=30)

session.add(user1)
session.add(user2)
session.commit()
print(f"创建用户: {user1}, ID: {user1.id}")

# 查询（Read）
# 查询所有
users = session.query(User).all()
print(f"所有用户: {users}")

# 条件查询
user = session.query(User).filter_by(name='张三').first()
print(f"查找张三: {user}")

# 复杂查询
adults = session.query(User).filter(User.age >= 25).order_by(User.age.desc()).all()
print(f"年龄>=25的用户: {adults}")

# 更新（Update）
user = session.query(User).filter_by(name='张三').first()
user.age = 26
session.commit()
print(f"更新后的年龄: {user.age}")

# 删除（Delete）
user_to_delete = session.query(User).filter_by(name='李四').first()
if user_to_delete:
    session.delete(user_to_delete)
    session.commit()
    print("已删除李四")

# 关闭会话
session.close()

# ============================================================================
# 第三部分：MySQL 数据库
# ============================================================================
"""
【概念讲解】
MySQL 是最流行的关系型数据库之一。
使用 PyMySQL 连接 MySQL。

安装：pip install pymysql
"""

# ----------------------------------------------------------------------------
# MySQL 连接示例
# ----------------------------------------------------------------------------

import pymysql

# 连接配置（示例）
config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',
    'database': 'test_db',
    'charset': 'utf8mb4'
}

# 使用 try-except 确保连接关闭
def mysql_example():
    """MySQL 操作示例"""
    # 实际使用时取消注释
    # conn = pymysql.connect(**config)
    # try:
    #     with conn.cursor() as cursor:
    #         # 创建表
    #         cursor.execute('''
    #         CREATE TABLE IF NOT EXISTS products (
    #             id INT AUTO_INCREMENT PRIMARY KEY,
    #             name VARCHAR(100),
    #             price DECIMAL(10, 2),
    #             stock INT DEFAULT 0
    #         )
    #         ''')
    #         
    #         # 插入数据
    #         cursor.execute(
    #             'INSERT INTO products (name, price, stock) VALUES (%s, %s, %s)',
    #             ('iPhone', 999.99, 100)
    #         )
    #         
    #         # 查询
    #         cursor.execute('SELECT * FROM products')
    #         results = cursor.fetchall()
    #         print(results)
    #         
    #     conn.commit()
    # finally:
    #     conn.close()
    pass

# SQLAlchemy 连接 MySQL
def sqlalchemy_mysql_example():
    """SQLAlchemy 连接 MySQL 示例"""
    # 连接字符串格式
    # mysql+pymysql://user:password@host:port/database
    # engine = create_engine('mysql+pymysql://root:password@localhost:3306/test_db')
    pass

# ============================================================================
# 第四部分：数据库最佳实践
# ============================================================================

# ----------------------------------------------------------------------------
# 使用连接池
# ----------------------------------------------------------------------------

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# 配置连接池
engine = create_engine(
    'sqlite:///pooled.db',
    poolclass=QueuePool,
    pool_size=5,         # 连接池大小
    max_overflow=10,     # 最大溢出连接
    pool_timeout=30,     # 获取连接超时
    pool_recycle=3600    # 连接回收时间
)

# ----------------------------------------------------------------------------
# 事务处理
# ----------------------------------------------------------------------------

def transfer_money(from_id, to_id, amount):
    """转账事务示例"""
    session = Session()
    try:
        # 扣款
        from_user = session.query(User).get(from_id)
        # 假设有 balance 字段
        # from_user.balance -= amount
        
        # 收款
        to_user = session.query(User).get(to_id)
        # to_user.balance += amount
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"转账失败: {e}")
        return False
    finally:
        session.close()

# ----------------------------------------------------------------------------
# 批量操作优化
# ----------------------------------------------------------------------------

def bulk_insert_users(users_data):
    """批量插入优化"""
    session = Session()
    try:
        # 使用 bulk_insert_mappings 比 循环 add 快很多
        session.bulk_insert_mappings(User, users_data)
        session.commit()
    finally:
        session.close()

# 示例数据
users_data = [
    {'name': f'用户{i}', 'email': f'user{i}@example.com', 'age': 20 + i}
    for i in range(100)
]
# bulk_insert_users(users_data)

# ============================================================================
# 第五部分：数据库工具类封装
# ============================================================================

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, db_url='sqlite:///app.db'):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """获取会话"""
        return self.Session()
    
    def add_user(self, name, email, age):
        """添加用户"""
        session = self.get_session()
        try:
            user = User(name=name, email=email, age=age)
            session.add(user)
            session.commit()
            return user
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_user_by_name(self, name):
        """根据名称获取用户"""
        session = self.get_session()
        try:
            return session.query(User).filter_by(name=name).first()
        finally:
            session.close()
    
    def get_all_users(self):
        """获取所有用户"""
        session = self.get_session()
        try:
            return session.query(User).all()
        finally:
            session.close()
    
    def update_user_age(self, name, new_age):
        """更新用户年龄"""
        session = self.get_session()
        try:
            user = session.query(User).filter_by(name=name).first()
            if user:
                user.age = new_age
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def delete_user(self, name):
        """删除用户"""
        session = self.get_session()
        try:
            user = session.query(User).filter_by(name=name).first()
            if user:
                session.delete(user)
                session.commit()
                return True
            return False
        finally:
            session.close()

# 使用示例
db = DatabaseManager('sqlite:///app.db')
user = db.add_user('测试用户', 'test@example.com', 25)
print(f"添加用户: {user}")

# ============================================================================
# 练习题
# ============================================================================
"""
【练习1】SQLite 操作
1. 创建一个学生成绩数据库
2. 实现增删改查功能
3. 统计班级平均分

【练习2】ORM 实践
1. 定义订单和商品模型
2. 建立一对多关系
3. 实现订单查询功能

【练习3】数据库封装
1. 封装一个通用的数据库操作类
2. 支持 CRUD 操作
3. 支持事务处理
"""

# ============================================================================
# 清理测试数据库
# ============================================================================

import os

# 关闭所有连接
engine.dispose()

# 删除测试数据库文件
for db_file in ['example.db', 'orm_example.db', 'pooled.db', 'app.db']:
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"已删除 {db_file}")

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. SQLite 数据库操作
2. SQLAlchemy ORM 使用
3. MySQL 连接配置
4. 数据库连接池
5. 事务处理
6. 批量操作优化
7. 数据库工具类封装

🔧 PyCharm 技巧：
1. Database 工具窗口连接数据库
2. 执行 SQL 查询
3. 查看表结构和数据
4. 数据导出导入

➡️ 下一节：MongoDB 数据库
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("数据库基础模块学习完成！")
    print("=" * 60)
