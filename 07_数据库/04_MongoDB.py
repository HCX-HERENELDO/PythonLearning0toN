#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：MongoDB
学习目标：掌握 MongoDB 文档数据库的操作
PyCharm 技巧：学习文档数据库设计
============================================================================
"""

# 需要安装: pip install pymongo
# MongoDB 是 NoSQL 文档数据库，需要先安装 MongoDB 服务

# ============================================================================
# 第一部分：MongoDB 概念
# ============================================================================
"""
【概念讲解】
MongoDB 是面向文档的 NoSQL 数据库。

核心概念：
- Database：数据库
- Collection：集合（类似表）
- Document：文档（类似记录）
- Field：字段

与关系型数据库对比：
┌─────────────────┬─────────────────┐
│ 关系型数据库     │ MongoDB         │
├─────────────────┼─────────────────┤
│ Database        │ Database        │
│ Table           │ Collection      │
│ Row             │ Document        │
│ Column          │ Field           │
│ Primary Key     │ _id             │
└─────────────────┴─────────────────┘

MongoDB 特点：
- 文档以 JSON/BSON 格式存储
- 无固定模式（Schema-less）
- 支持嵌套文档和数组
- 高性能、易扩展
"""

# ============================================================================
# 第二部分：连接 MongoDB
# ============================================================================

try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    import datetime
    
    # 连接 MongoDB
    # 默认连接本地 MongoDB，端口 27017
    client = MongoClient('mongodb://localhost:27017/')
    
    # 查看所有数据库
    print(f"数据库列表: {client.list_database_names()}")
    
    # 获取/创建数据库
    db = client['python_learning']
    
    # 获取/创建集合
    users_collection = db['users']
    
    MONGODB_AVAILABLE = True
    
except Exception as e:
    print(f"MongoDB 连接失败: {e}")
    print("请确保 MongoDB 服务已启动")
    MONGODB_AVAILABLE = False

# ============================================================================
# 第三部分：CRUD 操作
# ============================================================================

if MONGODB_AVAILABLE:
    
    # ----------------------------------------------------------------------------
    # Create - 插入文档
    # ----------------------------------------------------------------------------
    
    # 插入单个文档
    user = {
        "name": "张三",
        "email": "zhangsan@example.com",
        "age": 25,
        "city": "北京",
        "interests": ["Python", "AI", "数据库"],
        "created_at": datetime.datetime.now()
    }
    
    result = users_collection.insert_one(user)
    print(f"插入文档 ID: {result.inserted_id}")
    
    # 插入多个文档
    users = [
        {
            "name": "李四",
            "email": "lisi@example.com",
            "age": 30,
            "city": "上海",
            "interests": ["Java", "Web"],
            "created_at": datetime.datetime.now()
        },
        {
            "name": "王五",
            "email": "wangwu@example.com",
            "age": 28,
            "city": "广州",
            "interests": ["Python", "数据分析"],
            "created_at": datetime.datetime.now()
        }
    ]
    
    result = users_collection.insert_many(users)
    print(f"插入 {len(result.inserted_ids)} 个文档")
    
    # ----------------------------------------------------------------------------
    # Read - 查询文档
    # ----------------------------------------------------------------------------
    
    # 查询所有
    all_users = users_collection.find()
    print("所有用户:")
    for user in all_users:
        print(f"  - {user['name']}: {user['email']}")
    
    # 条件查询
    user = users_collection.find_one({"name": "张三"})
    print(f"查询张三: {user}")
    
    # 多条件查询
    users = users_collection.find({
        "age": {"$gte": 25, "$lte": 30}
    })
    print("年龄 25-30 的用户:")
    for user in users:
        print(f"  - {user['name']}: {user['age']}")
    
    # 投影（只返回指定字段）
    users = users_collection.find(
        {"city": "北京"},
        {"name": 1, "email": 1, "_id": 0}
    )
    print("北京用户（只返回姓名和邮箱）:")
    for user in users:
        print(f"  - {user}")
    
    # 排序和限制
    users = users_collection.find().sort("age", -1).limit(2)
    print("年龄最大的2个用户:")
    for user in users:
        print(f"  - {user['name']}: {user['age']}")
    
    # 统计
    count = users_collection.count_documents({"city": "北京"})
    print(f"北京用户数: {count}")
    
    # ----------------------------------------------------------------------------
    # Update - 更新文档
    # ----------------------------------------------------------------------------
    
    # 更新单个文档
    result = users_collection.update_one(
        {"name": "张三"},
        {"$set": {"age": 26, "city": "深圳"}}
    )
    print(f"更新了 {result.modified_count} 个文档")
    
    # 更新多个文档
    result = users_collection.update_many(
        {"age": {"$gt": 25}},
        {"$inc": {"age": 1}}  # 年龄加1
    )
    print(f"更新了 {result.modified_count} 个文档")
    
    # 添加数组元素
    result = users_collection.update_one(
        {"name": "张三"},
        {"$push": {"interests": "机器学习"}}
    )
    
    # ----------------------------------------------------------------------------
    # Delete - 删除文档
    # ----------------------------------------------------------------------------
    
    # 删除单个文档
    # result = users_collection.delete_one({"name": "测试用户"})
    # print(f"删除了 {result.deleted_count} 个文档")
    
    # 删除多个文档
    # result = users_collection.delete_many({"age": {"$gt": 50}})
    # print(f"删除了 {result.deleted_count} 个文档")

# ============================================================================
# 第四部分：查询操作符
# ============================================================================

"""
【常用查询操作符】

比较操作符：
$eq   - 等于
$ne   - 不等于
$gt   - 大于
$gte  - 大于等于
$lt   - 小于
$lte  - 小于等于
$in   - 在列表中
$nin  - 不在列表中

逻辑操作符：
$and  - 与
$or   - 或
$not  - 非
$nor  - 或非

数组操作符：
$all  - 包含所有元素
$elemMatch - 匹配数组元素
$size - 数组长度

示例：
{"age": {"$gte": 18, "$lte": 60}}     # 18-60岁
{"city": {"$in": ["北京", "上海"]}}    # 北京或上海
{"interests": {"$all": ["Python"]}}   # 兴趣包含Python
"""

if MONGODB_AVAILABLE:
    # 复杂查询示例
    users = users_collection.find({
        "$and": [
            {"age": {"$gte": 25}},
            {"interests": {"$in": ["Python"]}}
        ]
    })
    print("年龄>=25且对Python感兴趣的用户:")
    for user in users:
        print(f"  - {user['name']}")

# ============================================================================
# 第五部分：聚合管道
# ============================================================================

if MONGODB_AVAILABLE:
    
    # 分组统计
    pipeline = [
        {"$group": {
            "_id": "$city",
            "count": {"$sum": 1},
            "avg_age": {"$avg": "$age"}
        }},
        {"$sort": {"count": -1}}
    ]
    
    results = users_collection.aggregate(pipeline)
    print("按城市分组统计:")
    for result in results:
        print(f"  - {result['_id']}: {result['count']}人, 平均年龄{result['avg_age']:.1f}")
    
    # 投影和过滤
    pipeline = [
        {"$match": {"age": {"$gte": 25}}},
        {"$project": {
            "name": 1,
            "age": 1,
            "interests_count": {"$size": "$interests"}
        }}
    ]
    
    results = users_collection.aggregate(pipeline)
    print("25岁以上的用户及其兴趣数量:")
    for result in results:
        print(f"  - {result['name']}: {result['interests_count']}个兴趣")

# ============================================================================
# 第六部分：索引
# ============================================================================

if MONGODB_AVAILABLE:
    
    # 创建索引
    users_collection.create_index([("email", 1)], unique=True)  # 唯一索引
    users_collection.create_index([("name", 1)])  # 普通索引
    users_collection.create_index([("age", -1)])  # 降序索引
    
    # 查看索引
    indexes = users_collection.list_indexes()
    print("索引列表:")
    for index in indexes:
        print(f"  - {index['name']}")
    
    # 删除索引
    # users_collection.drop_index("name_1")

# ============================================================================
# 第七部分：清理
# ============================================================================

if MONGODB_AVAILABLE:
    # 清空集合
    users_collection.delete_many({})
    
    # 删除集合
    # db.drop_collection('users')
    
    # 关闭连接
    client.close()

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. MongoDB 概念和特点
2. 连接和基本操作
3. CRUD 操作
4. 查询操作符
5. 聚合管道
6. 索引创建

➡️ 下一章：深度学习
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MongoDB 模块学习完成！")
    print("=" * 60)
