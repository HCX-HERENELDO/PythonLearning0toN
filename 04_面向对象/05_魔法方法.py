#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：魔法方法
学习目标：掌握常用魔法方法，让类更强大
PyCharm 技巧：学习使用代码生成快速创建魔法方法
============================================================================
"""

# ============================================================================
# 第一部分：对象表示
# ============================================================================

class Person:
    """人员类"""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        """用户友好的字符串表示"""
        return f"Person({self.name}, {self.age}岁)"
    
    def __repr__(self):
        """开发者友好的字符串表示"""
        return f"Person(name='{self.name}', age={self.age})"
    
    def __format__(self, format_spec):
        """自定义格式化"""
        if format_spec == "detail":
            return f"姓名: {self.name}, 年龄: {self.age}"
        return str(self)

p = Person("张三", 25)

print(str(p))      # 调用 __str__
print(repr(p))     # 调用 __repr__
print(f"{p:detail}")  # 调用 __format__

# ============================================================================
# 第二部分：比较运算
# ============================================================================

class Student:
    """学生类"""
    
    def __init__(self, name, score):
        self.name = name
        self.score = score
    
    def __eq__(self, other):
        """相等比较"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.score == other.score
    
    def __lt__(self, other):
        """小于比较"""
        if not isinstance(other, Student):
            return NotImplemented
        return self.score < other.score
    
    def __le__(self, other):
        """小于等于"""
        return self == other or self < other
    
    def __gt__(self, other):
        """大于"""
        return not self <= other
    
    def __ge__(self, other):
        """大于等于"""
        return not self < other
    
    def __repr__(self):
        return f"Student({self.name}, {self.score})"

students = [
    Student("张三", 85),
    Student("李四", 92),
    Student("王五", 78),
]

# 排序
sorted_students = sorted(students)
print(f"按成绩排序: {sorted_students}")

# 比较
s1 = Student("A", 90)
s2 = Student("B", 90)
print(f"相等: {s1 == s2}")

# ============================================================================
# 第三部分：算术运算
# ============================================================================

class Vector:
    """二维向量"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """加法"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """减法"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """乘法（标量）"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """右乘"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """除法"""
        return Vector(self.x / scalar, self.y / scalar)
    
    def __neg__(self):
        """取负"""
        return Vector(-self.x, -self.y)
    
    def __abs__(self):
        """绝对值（模长）"""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(f"加法: {v1 + v2}")
print(f"减法: {v1 - v2}")
print(f"乘法: {v1 * 2}")
print(f"右乘: {2 * v1}")
print(f"模长: {abs(v1)}")

# ============================================================================
# 第四部分：容器协议
# ============================================================================

class Deck:
    """扑克牌组"""
    
    def __init__(self):
        suits = "♠♥♦♣"
        ranks = "A23456789TJQK"
        self.cards = [s + r for s in suits for r in ranks]
    
    def __len__(self):
        """长度"""
        return len(self.cards)
    
    def __getitem__(self, index):
        """索引访问"""
        return self.cards[index]
    
    def __setitem__(self, index, value):
        """索引设置"""
        self.cards[index] = value
    
    def __contains__(self, card):
        """成员检查"""
        return card in self.cards
    
    def __iter__(self):
        """迭代"""
        return iter(self.cards)

deck = Deck()

print(f"牌组数量: {len(deck)}")
print(f"第一张: {deck[0]}")
print(f"最后一张: {deck[-1]}")
print(f"'♠A' 在牌组中: {'♠A' in deck}")

# 支持切片
print(f"前3张: {deck[:3]}")

# 可迭代
for card in deck[:5]:
    print(card, end=" ")
print()

# ============================================================================
# 第五部分：可调用对象
# ============================================================================

class Multiplier:
    """乘法器"""
    
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        """使对象可调用"""
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(f"double(5) = {double(5)}")
print(f"triple(5) = {triple(5)}")
print(f"可调用: {callable(double)}")

# ============================================================================
# 第六部分：上下文管理器
# ============================================================================

class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name="Timer"):
        self.name = name
    
    def __enter__(self):
        """进入上下文"""
        import time
        self.start = time.time()
        print(f"{self.name} 开始...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        import time
        elapsed = time.time() - self.start
        print(f"{self.name} 结束，耗时: {elapsed:.4f}秒")
        return False  # 不抑制异常

with Timer("数据处理"):
    sum(range(1000000))

# ============================================================================
# 第七部分：属性访问控制
# ============================================================================

class ProtectedAttr:
    """属性访问控制示例"""
    
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        """访问不存在的属性时调用"""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' 没有属性 '{name}'")
    
    def __setattr__(self, name, value):
        """设置属性时调用"""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            print(f"设置 {name} = {value}")
            self._data[name] = value
    
    def __delattr__(self, name):
        """删除属性时调用"""
        if name in self._data:
            print(f"删除 {name}")
            del self._data[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' 没有属性 '{name}'")

obj = ProtectedAttr()
obj.name = "张三"  # 触发 __setattr__
print(obj.name)    # 触发 __getattr__

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. __str__ 和 __repr__
2. 比较运算 (__eq__, __lt__ 等)
3. 算术运算 (__add__, __mul__ 等)
4. 容器协议 (__len__, __getitem__ 等)
5. 可调用对象 __call__
6. 上下文管理器 __enter__, __exit__
7. 属性访问控制

🔧 PyCharm 技巧：
1. Alt+Insert → Override Methods
2. Ctrl+O 快速重写
3. Live Templates 快速生成

➡️ 下一节：属性装饰器
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("魔法方法模块学习完成！")
    print("=" * 60)
