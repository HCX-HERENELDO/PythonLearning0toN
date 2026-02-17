#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：CSV处理
学习目标：掌握 CSV 文件的读写操作
PyCharm 技巧：学习 CSV 文件预览
============================================================================
"""

import csv
from pathlib import Path

# ============================================================================
# 第一部分：CSV 基础
# ============================================================================
"""
【概念讲解】
CSV (Comma-Separated Values) 是一种简单的表格数据格式。
Python 的 csv 模块可以处理 CSV 文件。
"""

# ----------------------------------------------------------------------------
# 写入 CSV 文件
# ----------------------------------------------------------------------------

# 示例数据
students = [
    {"name": "张三", "age": 20, "score": 85, "city": "北京"},
    {"name": "李四", "age": 22, "score": 92, "city": "上海"},
    {"name": "王五", "age": 21, "score": 78, "city": "广州"},
]

# 方式1：使用 writer
with open("students.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # 写入表头
    writer.writerow(["姓名", "年龄", "成绩", "城市"])
    
    # 写入数据
    for student in students:
        writer.writerow([student["name"], student["age"], 
                        student["score"], student["city"]])

print("CSV 文件已写入")

# 方式2：使用 DictWriter
with open("students_dict.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["name", "age", "score", "city"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    # 写入表头
    writer.writeheader()
    
    # 写入数据
    writer.writerows(students)

# ----------------------------------------------------------------------------
# 读取 CSV 文件
# ----------------------------------------------------------------------------

# 方式1：使用 reader
print("\n使用 reader 读取:")
with open("students.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    
    # 获取表头
    header = next(reader)
    print(f"表头: {header}")
    
    # 读取数据
    for row in reader:
        print(f"数据: {row}")

# 方式2：使用 DictReader
print("\n使用 DictReader 读取:")
with open("students_dict.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        print(f"{row['name']}: {row['score']}分")

# ============================================================================
# 第二部分：处理特殊情况
# ============================================================================

# ----------------------------------------------------------------------------
# 自定义分隔符
# ----------------------------------------------------------------------------

data = [
    ["产品", "价格", "数量"],
    ["苹果", "5.5", "100"],
    ["香蕉", "3.2", "200"],
]

# 使用分号分隔
with open("products.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerows(data)

# 读取时指定分隔符
with open("products.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=";")
    for row in reader:
        print(row)

# ----------------------------------------------------------------------------
# 处理引号和特殊字符
# ----------------------------------------------------------------------------

special_data = [
    ["姓名", "描述"],
    ["张三", "他是\"好人\""],
    ["李四", "喜欢,编程"],
]

with open("special.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerows(special_data)

# ============================================================================
# 第三部分：实际应用
# ============================================================================

# ----------------------------------------------------------------------------
# 数据分析
# ----------------------------------------------------------------------------

def analyze_csv(filepath):
    """分析 CSV 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        scores = []
        for row in reader:
            scores.append(int(row["score"]))
        
        return {
            "count": len(scores),
            "average": sum(scores) / len(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "min": min(scores) if scores else 0
        }

stats = analyze_csv("students_dict.csv")
print(f"\n成绩统计: {stats}")

# ----------------------------------------------------------------------------
# 数据转换
# ----------------------------------------------------------------------------

def csv_to_json(csv_path, json_path):
    """CSV 转 JSON"""
    import json
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"已转换 {len(data)} 条记录")

csv_to_json("students_dict.csv", "students.json")

# ============================================================================
# 第四部分：使用 pandas 处理 CSV
# ============================================================================

# 如果安装了 pandas
try:
    import pandas as pd
    
    # 读取 CSV
    df = pd.read_csv("students_dict.csv")
    print(f"\nPandas 读取:\n{df}")
    
    # 基本统计
    print(f"\n统计:\n{df.describe()}")
    
    # 筛选
    filtered = df[df["score"] >= 80]
    print(f"\n80分以上:\n{filtered}")
    
except ImportError:
    print("\nPandas 未安装，跳过此部分")

# ============================================================================
# 清理测试文件
# ============================================================================

for f in ["students.csv", "students_dict.csv", "products.csv", 
          "special.csv", "students.json"]:
    if Path(f).exists():
        Path(f).unlink()

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. csv.writer 和 csv.reader
2. csv.DictWriter 和 csv.DictReader
3. 自定义分隔符和引号处理
4. CSV 与其他格式转换
5. 使用 pandas 处理 CSV

➡️ 下一节：序列化
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CSV处理模块学习完成！")
    print("=" * 60)
