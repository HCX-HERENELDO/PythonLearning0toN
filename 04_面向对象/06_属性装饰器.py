#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：属性装饰器
学习目标：掌握 @property 装饰器的使用
PyCharm 技巧：学习快速生成属性
============================================================================
"""

# ============================================================================
# 第一部分：@property 基础
# ============================================================================
"""
【概念讲解】
@property 装饰器将方法转换为只读属性。
可以实现对属性访问的控制和验证。
"""

class Circle:
    """圆形类"""
    
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """半径（只读属性）"""
        return self._radius
    
    @property
    def area(self):
        """面积（计算属性）"""
        import math
        return math.pi * self._radius ** 2
    
    @property
    def circumference(self):
        """周长（计算属性）"""
        import math
        return 2 * math.pi * self._radius

c = Circle(5)

print(f"半径: {c.radius}")
print(f"面积: {c.area:.2f}")
print(f"周长: {c.circumference:.2f}")

# c.radius = 10  # 错误：只读属性

# ============================================================================
# 第二部分：getter、setter、deleter
# ============================================================================

class Temperature:
    """温度类（摄氏度）"""
    
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """获取摄氏度"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """设置摄氏度（带验证）"""
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """删除属性"""
        print("删除温度值")
        self._celsius = 0
    
    @property
    def fahrenheit(self):
        """华氏度（只读）"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """通过华氏度设置"""
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)

print(f"摄氏度: {temp.celsius}°C")
print(f"华氏度: {temp.fahrenheit}°F")

temp.celsius = 30
print(f"修改后: {temp.celsius}°C")

temp.fahrenheit = 100
print(f"华氏度设置后: {temp.celsius}°C")

# temp.celsius = -300  # 报错

del temp.celsius
print(f"删除后: {temp.celsius}°C")

# ============================================================================
# 第三部分：只读属性和延迟计算
# ============================================================================

class LazyProperty:
    """延迟计算属性"""
    
    def __init__(self, func):
        self.func = func
        self.attr_name = func.__name__
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # 计算并缓存结果
        value = self.func(obj)
        setattr(obj, self.attr_name, value)
        return value

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, data):
        self.data = data
    
    @property
    def count(self):
        """计数（每次都计算）"""
        print("计算 count")
        return len(self.data)
    
    @LazyProperty
    def expensive_result(self):
        """耗时计算（只计算一次）"""
        print("执行耗时计算...")
        import time
        time.sleep(0.1)
        return sum(x ** 2 for x in self.data)

processor = DataProcessor(range(10))

print(f"第一次 count: {processor.count}")
print(f"第二次 count: {processor.count}")

print(f"第一次 expensive_result: {processor.expensive_result}")
print(f"第二次 expensive_result: {processor.expensive_result}")  # 使用缓存

# ============================================================================
# 第四部分：属性验证
# ============================================================================

class Person:
    """人员类（属性验证）"""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    @property
    def name(self):
        return self._name
    
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

try:
    p = Person("张三", 25)
    print(f"姓名: {p.name}, 年龄: {p.age}")
    
    # p.age = -5  # 报错
    # p.name = "A"  # 报错
except ValueError as e:
    print(f"错误: {e}")

# ============================================================================
# 第五部分：描述符
# ============================================================================

class ValidatedAttribute:
    """验证描述符"""
    
    def __init__(self, name, validator):
        self.name = name
        self.validator = validator
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"无效值: {value}")
        obj.__dict__[self.name] = value

def validate_positive(value):
    """验证正数"""
    return isinstance(value, (int, float)) and value > 0

def validate_string(value):
    """验证非空字符串"""
    return isinstance(value, str) and len(value) > 0

class Product:
    """产品类（使用描述符）"""
    
    name = ValidatedAttribute("name", validate_string)
    price = ValidatedAttribute("price", validate_positive)
    
    def __init__(self, name, price):
        self.name = name
        self.price = price

product = Product("Python教程", 99.9)
print(f"产品: {product.name}, 价格: {product.price}")

# product.price = -10  # 报错

# ============================================================================
# 第六部分：cached_property (Python 3.8+)
# ============================================================================

from functools import cached_property

class DataSet:
    """数据集"""
    
    def __init__(self, data):
        self.data = data
    
    @cached_property
    def statistics(self):
        """统计信息（缓存）"""
        print("计算统计信息...")
        return {
            "count": len(self.data),
            "sum": sum(self.data),
            "avg": sum(self.data) / len(self.data) if self.data else 0
        }

dataset = DataSet([1, 2, 3, 4, 5])

print(f"第一次: {dataset.statistics}")
print(f"第二次: {dataset.statistics}")  # 使用缓存

# 清除缓存
del dataset.statistics
print(f"清除后: {dataset.statistics}")  # 重新计算

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. @property 基本用法
2. getter、setter、deleter
3. 只读属性
4. 属性验证
5. 延迟计算
6. 描述符
7. @cached_property

🔧 PyCharm 技巧：
1. Alt+Insert → Property
2. 快速生成 getter/setter
3. 使用 Live Templates

➡️ 恭喜完成面向对象模块！
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("属性装饰器模块学习完成！")
    print("面向对象模块全部完成！")
    print("=" * 60)