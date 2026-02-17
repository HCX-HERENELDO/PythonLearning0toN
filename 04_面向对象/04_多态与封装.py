#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：封装
学习目标：理解封装的概念和实现方式
PyCharm 技巧：学习使用代码检查
============================================================================
"""

# ============================================================================
# 第一部分：封装的概念
# ============================================================================
"""
【概念讲解】
封装是将数据（属性）和操作数据的方法包装在一起，
并隐藏内部实现细节，只暴露必要的接口。

封装的好处：
1. 隐藏实现细节
2. 保护数据安全
3. 提供统一的访问接口
4. 便于维护和修改
"""

# ============================================================================
# 第二部分：访问控制
# ============================================================================

class BankAccount:
    """银行账户（封装示例）"""
    
    def __init__(self, owner, initial_balance=0):
        self.owner = owner                    # 公有属性
        self._balance = initial_balance       # 受保护属性
        self.__account_id = self._generate_id()  # 私有属性
    
    def _generate_id(self):
        """生成账户ID（受保护方法）"""
        import random
        return f"ACC{random.randint(10000, 99999)}"
    
    def deposit(self, amount):
        """存款（公有接口）"""
        if self._validate_amount(amount):
            self._balance += amount
            self._log_transaction("存款", amount)
            return True
        return False
    
    def withdraw(self, amount):
        """取款（公有接口）"""
        if self._validate_amount(amount) and amount <= self._balance:
            self._balance -= amount
            self._log_transaction("取款", amount)
            return True
        return False
    
    def get_balance(self):
        """获取余额"""
        return self._balance
    
    def _validate_amount(self, amount):
        """验证金额（受保护方法）"""
        return isinstance(amount, (int, float)) and amount > 0
    
    def __log_transaction(self, transaction_type, amount):
        """记录交易（私有方法）"""
        print(f"[{transaction_type}] 金额: {amount}, 余额: {self._balance}")

# 使用
account = BankAccount("张三", 1000)
account.deposit(500)
account.withdraw(200)
print(f"当前余额: {account.get_balance()}")

# ============================================================================
# 第三部分：属性访问器
# ============================================================================

class Person:
    """人员类（使用属性访问器）"""
    
    def __init__(self, name, age):
        self._name = name
        self._age = age
    
    # Getter
    @property
    def name(self):
        return self._name
    
    # Setter
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("姓名必须是字符串")
        if len(value) < 2:
            raise ValueError("姓名至少2个字符")
        self._name = value
    
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, value):
        if not isinstance(value, int):
            raise TypeError("年龄必须是整数")
        if value < 0 or value > 150:
            raise ValueError("年龄必须在0-150之间")
        self._age = value
    
    def __repr__(self):
        return f"Person(name='{self._name}', age={self._age})"

person = Person("张三", 25)
print(f"人员信息: {person}")

person.age = 26  # 通过 setter 设置
print(f"修改后年龄: {person.age}")

# ============================================================================
# 第四部分：使用描述符
# ============================================================================

class ValidatedAttribute:
    """验证描述符"""
    
    def __init__(self, name, validator, error_msg):
        self.name = name
        self.validator = validator
        self.error_msg = error_msg
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(self.error_msg)
        obj.__dict__[self.name] = value

def is_positive(value):
    return isinstance(value, (int, float)) and value > 0

def is_non_empty_string(value):
    return isinstance(value, str) and len(value.strip()) > 0

class Product:
    """产品类（使用描述符）"""
    
    name = ValidatedAttribute("name", is_non_empty_string, "名称不能为空")
    price = ValidatedAttribute("price", is_positive, "价格必须为正数")
    
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    def __repr__(self):
        return f"Product(name='{self.name}', price={self.price})"

product = Product("Python教程", 99.9)
print(f"产品: {product}")

# product.price = -10  # ValueError

# ============================================================================
# 第五部分：封装设计原则
# ============================================================================

"""
【封装设计原则】

1. 最小化公有接口
   - 只暴露必要的方法
   - 其他都设为私有或受保护

2. 数据隐藏
   - 属性设为私有
   - 通过方法访问

3. 验证输入
   - 在 setter 中验证
   - 确保数据有效性

4. 提供便捷方法
   - 常用操作封装为方法
   - 简化使用
"""

class Stack:
    """栈（封装良好的示例）"""
    
    def __init__(self):
        self._items = []
    
    def push(self, item):
        """入栈"""
        self._items.append(item)
    
    def pop(self):
        """出栈"""
        if not self.is_empty():
            return self._items.pop()
        raise IndexError("栈为空")
    
    def peek(self):
        """查看栈顶"""
        if not self.is_empty():
            return self._items[-1]
        raise IndexError("栈为空")
    
    def is_empty(self):
        """是否为空"""
        return len(self._items) == 0
    
    def size(self):
        """大小"""
        return len(self._items)

stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(f"栈顶: {stack.peek()}")
print(f"出栈: {stack.pop()}")
print(f"大小: {stack.size()}")

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. 封装的概念和好处
2. 访问控制（公有、受保护、私有）
3. 属性访问器（getter/setter）
4. 描述符
5. 封装设计原则

➡️ 下一节：多态与封装
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("封装模块学习完成！")
    print("=" * 60)
