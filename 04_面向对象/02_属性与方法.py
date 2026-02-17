#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：属性与方法
学习目标：理解类属性、实例属性和各种方法的区别
PyCharm 技巧：学习使用 Structure 面板查看类结构
============================================================================
"""

# ============================================================================
# 第一部分：类属性与实例属性
# ============================================================================

class Student:
    """学生类"""
    
    # 类属性 - 所有实例共享
    school = "清华大学"
    count = 0
    
    def __init__(self, name, age):
        # 实例属性 - 每个实例独有
        self.name = name
        self.age = age
        Student.count += 1

# 创建实例
s1 = Student("张三", 20)
s2 = Student("李四", 22)

# 访问类属性
print(f"学校: {Student.school}")
print(f"学生数: {Student.count}")

# 访问实例属性
print(f"s1 姓名: {s1.name}")
print(f"s2 姓名: {s2.name}")

# 通过实例访问类属性
print(f"s1 学校: {s1.school}")

# 修改类属性
Student.school = "北京大学"
print(f"修改后 s1 学校: {s1.school}")  # 所有实例都改变

# 通过实例"修改"类属性（实际是创建新实例属性）
s1.school = "复旦大学"  # 创建了实例属性，遮蔽类属性
print(f"s1 实例学校: {s1.school}")    # 复旦大学（实例属性）
print(f"s2 类学校: {s2.school}")      # 北京大学（类属性）
print(f"类属性: {Student.school}")    # 北京大学

# ============================================================================
# 第二部分：实例方法、类方法、静态方法
# ============================================================================

class Counter:
    """计数器类"""
    
    total = 0  # 类属性
    
    def __init__(self):
        self.value = 0
        Counter.total += 1
    
    # 实例方法 - 可以访问实例和类
    def increment(self):
        """增加计数"""
        self.value += 1
        return self.value
    
    # 类方法 - 只能访问类属性
    @classmethod
    def get_total(cls):
        """获取总数"""
        return cls.total
    
    # 静态方法 - 不能访问实例或类属性
    @staticmethod
    def description():
        """描述"""
        return "这是一个计数器类"

# 使用
c1 = Counter()
c2 = Counter()

# 实例方法
print(f"c1.increment(): {c1.increment()}")
print(f"c1.increment(): {c1.increment()}")

# 类方法
print(f"Counter.get_total(): {Counter.get_total()}")

# 静态方法
print(f"Counter.description(): {Counter.description()}")

# ============================================================================
# 第三部分：私有属性和私有方法
# ============================================================================

class BankAccount:
    """银行账户"""
    
    def __init__(self, owner, balance):
        self.owner = owner
        self._balance = balance      # 受保护属性（约定）
        self.__secret = "password"   # 私有属性（名称修饰）
    
    # 公有方法
    def deposit(self, amount):
        """存款"""
        if amount > 0:
            self._balance += amount
            return True
        return False
    
    # 受保护方法
    def _validate(self, amount):
        """验证金额"""
        return amount > 0
    
    # 私有方法
    def __log_transaction(self, amount):
        """记录交易（内部使用）"""
        print(f"交易记录: {amount}")

account = BankAccount("张三", 1000)

# 公有属性
print(f"户主: {account.owner}")

# 受保护属性（可以访问，但不推荐）
print(f"余额: {account._balance}")

# 私有属性（无法直接访问）
# print(account.__secret)  # AttributeError

# 通过名称修饰访问（不推荐）
print(f"私有属性: {account._BankAccount__secret}")

# ============================================================================
# 第四部分：@property 装饰器
# ============================================================================

class Temperature:
    """温度类"""
    
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """获取摄氏度"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """设置摄氏度"""
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """获取华氏度（只读）"""
        return self._celsius * 9/5 + 32

temp = Temperature(25)
print(f"摄氏度: {temp.celsius}°C")
print(f"华氏度: {temp.fahrenheit}°F")

temp.celsius = 30
print(f"修改后: {temp.celsius}°C")

# temp.fahrenheit = 100  # 错误：只读属性

# ============================================================================
# 第五部分：特殊属性
# ============================================================================

class MyClass:
    """示例类"""
    
    pass

obj = MyClass()

# __dict__ - 查看实例属性
obj.x = 1
obj.y = 2
print(f"实例属性: {obj.__dict__}")

# __class__ - 查看类
print(f"类: {obj.__class__}")

# __doc__ - 文档字符串
print(f"文档: {MyClass.__doc__}")

# __name__ - 类名
print(f"类名: {MyClass.__name__}")

# __bases__ - 父类
print(f"父类: {MyClass.__bases__}")

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. 类属性 vs 实例属性
2. 实例方法、类方法、静态方法
3. 私有属性和私有方法
4. @property 装饰器
5. 特殊属性

➡️ 下一节：继承
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("属性与方法模块学习完成！")
    print("=" * 60)
