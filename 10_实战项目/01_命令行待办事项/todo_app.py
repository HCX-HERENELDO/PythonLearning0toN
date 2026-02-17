#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
项目名称：命令行待办事项应用
项目目标：综合运用 Python 基础知识，创建一个实用的命令行工具
学习要点：
    - 变量、数据类型、条件语句、循环
    - 列表、字典操作
    - 函数定义和调用
    - 文件读写
    - 异常处理
PyCharm 技巧：断点调试、代码重构
============================================================================
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

# ============================================================================
# 数据模型
# ============================================================================

class TodoItem:
    """待办事项类"""
    
    def __init__(self, title: str, description: str = "", 
                 priority: str = "medium", completed: bool = False):
        """
        初始化待办事项
        
        参数:
            title: 标题
            description: 描述
            priority: 优先级
            completed: 是否完成
        """
        self.id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.title = title
        self.description = description
        self.priority = priority
        self.completed = completed
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "completed": self.completed,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TodoItem':
        """从字典创建实例"""
        item = cls(
            title=data["title"],
            description=data.get("description", ""),
            priority=data.get("priority", "medium"),
            completed=data.get("completed", False)
        )
        item.id = data["id"]
        item.created_at = data["created_at"]
        return item
    
    def __str__(self) -> str:
        """字符串表示"""
        status = "✓" if self.completed else "○"
        priority_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        priority_icon = priority_map.get(self.priority, "⚪")
        
        return f"[{status}] {priority_icon} {self.title} (ID: {self.id})"


# ============================================================================
# 数据管理
# ============================================================================

class TodoManager:
    """待办事项管理器"""
    
    def __init__(self, data_file: str = "todos.json"):
        """
        初始化管理器
        
        参数:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.todos: List[TodoItem] = []
        self.load()
    
    def load(self) -> None:
        """从文件加载数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.todos = [TodoItem.from_dict(item) for item in data]
                print(f"✓ 已加载 {len(self.todos)} 条待办事项")
            except json.JSONDecodeError:
                print("⚠ 数据文件格式错误，将创建新文件")
                self.todos = []
            except Exception as e:
                print(f"⚠ 加载数据失败: {e}")
                self.todos = []
        else:
            print("ℹ 创建新的数据文件")
            self.todos = []
    
    def save(self) -> None:
        """保存数据到文件"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump([item.to_dict() for item in self.todos], 
                         f, ensure_ascii=False, indent=2)
            print("✓ 数据已保存")
        except Exception as e:
            print(f"✗ 保存失败: {e}")
    
    def add(self, title: str, description: str = "", 
            priority: str = "medium") -> TodoItem:
        """添加新待办事项"""
        item = TodoItem(title, description, priority)
        self.todos.append(item)
        self.save()
        return item
    
    def remove(self, item_id: str) -> bool:
        """删除待办事项"""
        for i, item in enumerate(self.todos):
            if item.id == item_id:
                self.todos.pop(i)
                self.save()
                return True
        return False
    
    def toggle(self, item_id: str) -> Optional[TodoItem]:
        """切换完成状态"""
        for item in self.todos:
            if item.id == item_id:
                item.completed = not item.completed
                self.save()
                return item
        return None
    
    def get_by_id(self, item_id: str) -> Optional[TodoItem]:
        """根据 ID 获取待办事项"""
        for item in self.todos:
            if item.id == item_id:
                return item
        return None
    
    def list_all(self, show_completed: bool = True) -> List[TodoItem]:
        """列出所有待办事项"""
        if show_completed:
            return self.todos
        return [item for item in self.todos if not item.completed]
    
    def list_by_priority(self, priority: str) -> List[TodoItem]:
        """按优先级筛选"""
        return [item for item in self.todos if item.priority == priority]
    
    def search(self, keyword: str) -> List[TodoItem]:
        """搜索待办事项"""
        keyword = keyword.lower()
        return [
            item for item in self.todos 
            if keyword in item.title.lower() or keyword in item.description.lower()
        ]
    
    def clear_completed(self) -> int:
        """清除已完成的事项"""
        original_count = len(self.todos)
        self.todos = [item for item in self.todos if not item.completed]
        removed_count = original_count - len(self.todos)
        if removed_count > 0:
            self.save()
        return removed_count
    
    def statistics(self) -> Dict:
        """获取统计信息"""
        total = len(self.todos)
        completed = sum(1 for item in self.todos if item.completed)
        pending = total - completed
        
        high_priority = sum(1 for item in self.todos 
                           if item.priority == "high" and not item.completed)
        medium_priority = sum(1 for item in self.todos 
                             if item.priority == "medium" and not item.completed)
        low_priority = sum(1 for item in self.todos 
                          if item.priority == "low" and not item.completed)
        
        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "completion_rate": f"{(completed/total*100):.1f}%" if total > 0 else "0%",
            "by_priority": {
                "high": high_priority,
                "medium": medium_priority,
                "low": low_priority
            }
        }


# ============================================================================
# 用户界面
# ============================================================================

class TodoApp:
    """待办事项应用"""
    
    def __init__(self):
        """初始化应用"""
        self.manager = TodoManager()
        self.running = True
    
    def clear_screen(self) -> None:
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self) -> None:
        """打印标题"""
        print("\n" + "=" * 50)
        print("📋 待办事项管理器".center(46))
        print("=" * 50)
    
    def print_menu(self) -> None:
        """打印主菜单"""
        self.print_header()
        print("""
【主菜单】
  1. 查看所有待办事项
  2. 添加新待办事项
  3. 完成/取消完成
  4. 删除待办事项
  5. 搜索待办事项
  6. 按优先级筛选
  7. 查看统计信息
  8. 清除已完成事项
  0. 退出
        """)
    
    def print_todos(self, todos: List[TodoItem], title: str = "待办事项列表") -> None:
        """打印待办事项列表"""
        print(f"\n【{title}】")
        print("-" * 50)
        
        if not todos:
            print("暂无待办事项")
            return
        
        for i, item in enumerate(todos, 1):
            print(f"{i}. {item}")
            if item.description:
                print(f"   描述: {item.description}")
            print(f"   创建时间: {item.created_at}")
        
        print("-" * 50)
        print(f"共 {len(todos)} 条")
    
    def get_input(self, prompt: str, required: bool = True) -> str:
        """获取用户输入"""
        while True:
            value = input(prompt).strip()
            if value or not required:
                return value
            print("⚠ 此项为必填项，请重新输入")
    
    def select_priority(self) -> str:
        """选择优先级"""
        print("\n选择优先级:")
        print("  1. 高")
        print("  2. 中")
        print("  3. 低
        
        while True:
            choice = input("请选择 (1-3) [默认: 2]: ").strip()
            if not choice:
                return "medium"
            
            priority_map = {"1": "high", "2": "medium", "3": "low"}
            if choice in priority_map:
                return priority_map[choice]
            print("⚠ 无效选择，请重新输入")
    
    def select_todo(self, todos: List[TodoItem]) -> Optional[TodoItem]:
        """选择待办事项"""
        if not todos:
            print("暂无可选的待办事项")
            return None
        
        self.print_todos(todos, "选择待办事项")
        
        while True:
            choice = input("\n请输入序号 (输入 0 取消): ").strip()
            if choice == "0":
                return None
            
            try:
                index = int(choice) - 1
                if 0 <= index < len(todos):
                    return todos[index]
                print(f"⚠ 请输入 1-{len(todos)} 之间的数字")
            except ValueError:
                print("⚠ 请输入有效的数字")
    
    # ------------------------------------------------------------------------
    # 功能方法
    # ------------------------------------------------------------------------
    
    def view_all(self) -> None:
        """查看所有待办事项"""
        show_completed = input("显示已完成事项? (y/n) [默认: y]: ").strip().lower()
        show_completed = show_completed != 'n'
        
        todos = self.manager.list_all(show_completed)
        self.print_todos(todos)
        
        input("\n按回车键继续...")
    
    def add_todo(self) -> None:
        """添加新待办事项"""
        self.print_header()
        print("\n【添加新待办事项】")
        
        title = self.get_input("标题: ")
        description = self.get_input("描述 (可选): ", required=False)
        priority = self.select_priority()
        
        item = self.manager.add(title, description, priority)
        print(f"\n✓ 添加成功! ID: {item.id}")
        
        input("\n按回车键继续...")
    
    def toggle_todo(self) -> None:
        """切换完成状态"""
        todos = self.manager.list_all()
        item = self.select_todo(todos)
        
        if item:
            self.manager.toggle(item.id)
            status = "已完成" if not item.completed else "未完成"
            print(f"\n✓ 已标记为 {status}")
        
        input("\n按回车键继续...")
    
    def remove_todo(self) -> None:
        """删除待办事项"""
        todos = self.manager.list_all()
        item = self.select_todo(todos)
        
        if item:
            confirm = input(f"\n确定删除 '{item.title}'? (y/n): ").strip().lower()
            if confirm == 'y':
                self.manager.remove(item.id)
                print("✓ 删除成功")
        
        input("\n按回车键继续...")
    
    def search_todos(self) -> None:
        """搜索待办事项"""
        self.print_header()
        print("\n【搜索待办事项】")
        
        keyword = self.get_input("请输入搜索关键词: ")
        results = self.manager.search(keyword)
        
        self.print_todos(results, f"搜索结果: '{keyword}'")
        
        input("\n按回车键继续...")
    
    def filter_by_priority(self) -> None:
        """按优先级筛选"""
        priority = self.select_priority()
        
        priority_names = {"high": "高", "medium": "中", "low": "低"}
        todos = self.manager.list_by_priority(priority)
        
        self.print_todos(todos, f"优先级: {priority_names[priority]}")
        
        input("\n按回车键继续...")
    
    def show_statistics(self) -> None:
        """显示统计信息"""
        self.print_header()
        
        stats = self.manager.statistics()
        
        print("\n【统计信息】")
        print("-" * 50)
        print(f"  总计: {stats['total']} 条")
        print(f"  已完成: {stats['completed']} 条")
        print(f"  待处理: {stats['pending']} 条")
        print(f"  完成率: {stats['completion_rate']}")
        print()
        print("  待处理事项优先级分布:")
        print(f"    🔴 高优先级: {stats['by_priority']['high']} 条")
        print(f"    🟡 中优先级: {stats['by_priority']['medium']} 条")
        print(f"    🟢 低优先级: {stats['by_priority']['low']} 条")
        print("-" * 50)
        
        input("\n按回车键继续...")
    
    def clear_completed(self) -> None:
        """清除已完成事项"""
        count = self.manager.clear_completed()
        print(f"\n✓ 已清除 {count} 条已完成事项")
        
        input("\n按回车键继续...")
    
    # ------------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------------
    
    def run(self) -> None:
        """运行应用"""
        actions = {
            "1": self.view_all,
            "2": self.add_todo,
            "3": self.toggle_todo,
            "4": self.remove_todo,
            "5": self.search_todos,
            "6": self.filter_by_priority,
            "7": self.show_statistics,
            "8": self.clear_completed,
        }
        
        while self.running:
            self.print_menu()
            
            choice = input("请选择操作 (0-8): ").strip()
            
            if choice == "0":
                print("\n感谢使用，再见！")
                self.running = False
            elif choice in actions:
                actions[choice]()
            else:
                print("\n⚠ 无效选择，请重试")
                input("\n按回车键继续...")


# ============================================================================
# 程序入口
# ============================================================================

def main():
    """主函数"""
    print("\n欢迎使用待办事项管理器！")
    print("提示: 输入 0 可以返回上一级菜单\n")
    
    app = TodoApp()
    app.run()


if __name__ == "__main__":
    main()


# ============================================================================
# 练习建议
# ============================================================================
"""
【进阶练习】

1. 功能扩展:
   - 添加到期日期功能
   - 实现分类/标签功能
   - 支持导出为 CSV/Markdown
   - 添加提醒功能

2. 代码改进:
   - 添加单元测试
   - 使用 logging 替代 print
   - 实现数据备份功能
   - 添加撤销/重做功能

3. 学习要点:
   - 在 PyCharm 中设置断点调试
   - 使用 TODO 注释标记待办事项
   - 尝试重构代码，提高可维护性
   - 使用 Git 进行版本控制
"""
