#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：文件路径操作
学习目标：掌握 os 和 pathlib 模块进行路径操作
PyCharm 技巧：学习文件路径自动补全
============================================================================
"""

import os
from pathlib import Path

# ============================================================================
# 第一部分：os 模块
# ============================================================================
"""
【概念讲解】
os 模块提供了与操作系统交互的功能，包括文件和目录操作。
"""

# ----------------------------------------------------------------------------
# 当前目录操作
# ----------------------------------------------------------------------------

# 获取当前工作目录
current_dir = os.getcwd()
print(f"当前目录: {current_dir}")

# 改变工作目录
# os.chdir("/path/to/directory")

# ----------------------------------------------------------------------------
# 路径拼接
# ----------------------------------------------------------------------------

# 使用 os.path.join（推荐）
path1 = os.path.join("folder", "subfolder", "file.txt")
print(f"拼接路径: {path1}")

# 在 Windows 上自动使用反斜杠
path2 = os.path.join("data", "users", "profile.json")
print(f"跨平台路径: {path2}")

# ----------------------------------------------------------------------------
# 路径分解
# ----------------------------------------------------------------------------

# 获取目录名
dir_name = os.path.dirname("/home/user/documents/file.txt")
print(f"目录名: {dir_name}")

# 获取文件名
base_name = os.path.basename("/home/user/documents/file.txt")
print(f"文件名: {base_name}")

# 分割扩展名
name, ext = os.path.splitext("document.pdf")
print(f"名称: {name}, 扩展名: {ext}")

# ----------------------------------------------------------------------------
# 路径信息
# ----------------------------------------------------------------------------

test_path = "C:/Users/Hereneldo/PycharmProjects/PythonLearning/README.md"

# 判断路径是否存在
print(f"路径存在: {os.path.exists(test_path)}")

# 判断是否为文件
print(f"是文件: {os.path.isfile(test_path)}")

# 判断是否为目录
print(f"是目录: {os.path.isdir(test_path)}")

# 获取文件大小（字节）
if os.path.exists(test_path):
    size = os.path.getsize(test_path)
    print(f"文件大小: {size} 字节")

# 获取绝对路径
abs_path = os.path.abspath(".")
print(f"绝对路径: {abs_path}")

# ============================================================================
# 第二部分：pathlib 模块（推荐）
# ============================================================================
"""
【概念讲解】
pathlib 是 Python 3.4+ 引入的现代路径操作模块，
使用面向对象的方式处理路径，更加直观和易用。
"""

# ----------------------------------------------------------------------------
# 创建 Path 对象
# ----------------------------------------------------------------------------

# 当前目录
p1 = Path.cwd()
print(f"当前目录 (pathlib): {p1}")

# 用户主目录
p2 = Path.home()
print(f"主目录: {p2}")

# 从字符串创建
p3 = Path("data/users/profile.json")
print(f"创建路径: {p3}")

# ----------------------------------------------------------------------------
# 路径拼接（使用 / 运算符）
# ----------------------------------------------------------------------------

# 非常直观的路径拼接
base = Path("project")
data_file = base / "data" / "input.txt"
print(f"拼接路径: {data_file}")

# 可以混合字符串和 Path 对象
config = Path("config") / "settings.json"
print(f"配置文件: {config}")

# ----------------------------------------------------------------------------
# 路径属性
# ----------------------------------------------------------------------------

sample_path = Path("C:/Users/Hereneldo/PycharmProjects/PythonLearning/README.md")

# 各部分属性
print(f"文件名: {sample_path.name}")        # README.md
print(f"文件名(无扩展): {sample_path.stem}")  # README
print(f"扩展名: {sample_path.suffix}")       # .md
print(f"父目录: {sample_path.parent}")       # 上级目录
print(f"父目录的父目录: {sample_path.parent.parent}")

# 所有部分
print(f"路径部分: {sample_path.parts}")

# ----------------------------------------------------------------------------
# 路径判断
# ----------------------------------------------------------------------------

p = Path(".")

print(f"存在: {p.exists()}")
print(f"是文件: {p.is_file()}")
print(f"是目录: {p.is_dir()}")

# ----------------------------------------------------------------------------
# 文件操作
# ----------------------------------------------------------------------------

# 创建目录
new_dir = Path("test_directory")
new_dir.mkdir(exist_ok=True)  # exist_ok=True 避免已存在报错
print(f"创建目录: {new_dir}")

# 创建多级目录
deep_dir = Path("test_directory/a/b/c")
deep_dir.mkdir(parents=True, exist_ok=True)
print(f"创建多级目录: {deep_dir}")

# 创建文件
test_file = Path("test_directory/test.txt")
test_file.write_text("Hello, pathlib!", encoding="utf-8")
print(f"写入文件: {test_file}")

# 读取文件
content = test_file.read_text(encoding="utf-8")
print(f"文件内容: {content}")

# 删除文件
test_file.unlink()
print("文件已删除")

# 删除目录
import shutil
shutil.rmtree("test_directory")
print("目录已删除")

# ============================================================================
# 第三部分：遍历目录
# ============================================================================

# ----------------------------------------------------------------------------
# 使用 os.walk
# ----------------------------------------------------------------------------

def list_files_os(directory):
    """使用 os.walk 遍历目录"""
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 2 * (level + 1)
        for file in files[:5]:  # 限制显示数量
            print(f"{sub_indent}{file}")

# ----------------------------------------------------------------------------
# 使用 Path.rglob
# ----------------------------------------------------------------------------

def list_files_pathlib(directory):
    """使用 pathlib 遍历目录"""
    base = Path(directory)
    
    # 遍历所有 Python 文件
    for py_file in base.rglob("*.py"):
        print(py_file.relative_to(base))

# 只遍历一层
def list_immediate(path):
    """列出直接子项"""
    p = Path(path)
    for item in p.iterdir():
        if item.is_file():
            print(f"文件: {item.name}")
        else:
            print(f"目录: {item.name}/")

# ============================================================================
# 第四部分：实用函数
# ============================================================================

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_files_by_extension(directory, extension):
    """按扩展名获取文件"""
    return list(Path(directory).glob(f"*{extension}"))

def get_all_files(directory):
    """获取所有文件（递归）"""
    return [f for f in Path(directory).rglob("*") if f.is_file()]

def get_file_info(filepath):
    """获取文件信息"""
    p = Path(filepath)
    if not p.exists():
        return None
    
    stat = p.stat()
    return {
        "name": p.name,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "is_file": p.is_file(),
        "extension": p.suffix
    }

# 测试实用函数
ensure_dir("data/cache")
print("目录已确保存在")

# ============================================================================
# 第五部分：路径规范化
# ============================================================================

# 解析相对路径
relative = Path("../other_project/../current_project/./src")
resolved = relative.resolve()
print(f"解析后: {resolved}")

# 转换为正斜杠（跨平台）
win_path = Path("C:\\Users\\Documents")
posix_style = win_path.as_posix()
print(f"POSIX 风格: {posix_style}")

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. os 模块基本操作
2. os.path 路径处理
3. pathlib 面向对象路径操作
4. 目录遍历方法
5. 文件和目录操作

➡️ 下一节：JSON处理
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("文件路径操作模块学习完成！")
    print("=" * 60)
