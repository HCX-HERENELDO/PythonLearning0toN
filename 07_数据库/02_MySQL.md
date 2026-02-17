# MySQL 数据库

## 目录
1. [连接 MySQL](#1-连接-mysql)
2. [CRUD 操作](#2-crud-操作)
3. [事务处理](#3-事务处理)

---

## 1. 连接 MySQL

### 安装驱动

```bash
pip install pymysql
```

### 连接数据库

```python
import pymysql

# 连接配置
config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'your_password',
    'database': 'test_db',
    'charset': 'utf8mb4'
}

# 连接数据库
connection = pymysql.connect(**config)

try:
    with connection.cursor() as cursor:
        # 执行 SQL
        cursor.execute("SELECT VERSION()")
        result = cursor.fetchone()
        print(f"MySQL 版本: {result}")
finally:
    connection.close()
```

---

## 2. CRUD 操作

### 创建表

```python
create_table_sql = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

with connection.cursor() as cursor:
    cursor.execute(create_table_sql)
    connection.commit()
```

### 插入数据

```python
# 插入单条
insert_sql = "INSERT INTO users (name, email, age) VALUES (%s, %s, %s)"
with connection.cursor() as cursor:
    cursor.execute(insert_sql, ('张三', 'zhangsan@example.com', 25))
    connection.commit()

# 批量插入
users = [
    ('李四', 'lisi@example.com', 30),
    ('王五', 'wangwu@example.com', 28),
]
with connection.cursor() as cursor:
    cursor.executemany(insert_sql, users)
    connection.commit()
```

### 查询数据

```python
# 查询所有
select_sql = "SELECT * FROM users"
with connection.cursor() as cursor:
    cursor.execute(select_sql)
    results = cursor.fetchall()
    for row in results:
        print(row)

# 条件查询
select_sql = "SELECT * FROM users WHERE age > %s"
with connection.cursor() as cursor:
    cursor.execute(select_sql, (25,))
    results = cursor.fetchall()
```

### 更新数据

```python
update_sql = "UPDATE users SET age = %s WHERE name = %s"
with connection.cursor() as cursor:
    cursor.execute(update_sql, (26, '张三'))
    connection.commit()
```

### 删除数据

```python
delete_sql = "DELETE FROM users WHERE name = %s"
with connection.cursor() as cursor:
    cursor.execute(delete_sql, ('王五',))
    connection.commit()
```

---

## 3. 事务处理

```python
try:
    with connection.cursor() as cursor:
        # 开始事务（自动）
        
        # 操作 1
        cursor.execute(
            "UPDATE accounts SET balance = balance - 100 WHERE id = 1"
        )
        
        # 操作 2
        cursor.execute(
            "UPDATE accounts SET balance = balance + 100 WHERE id = 2"
        )
        
        # 提交事务
        connection.commit()
        
except Exception as e:
    # 回滚事务
    connection.rollback()
    print(f"事务失败: {e}")
```

---

## 使用 SQLAlchemy 连接 MySQL

```python
from sqlalchemy import create_engine

# 连接字符串
url = "mysql+pymysql://root:password@localhost:3306/test_db"

# 创建引擎
engine = create_engine(url, echo=True)

# 测试连接
with engine.connect() as conn:
    result = conn.execute("SELECT 1")
    print(result.fetchone())
```

---

## 练习题

1. 创建一个学生成绩管理表
2. 实现学生信息的增删改查
3. 使用事务实现转账功能
