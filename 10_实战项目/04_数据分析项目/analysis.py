#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
项目名称：数据分析项目
项目描述：使用 Pandas 进行数据分析
学习目标：综合运用数据处理、可视化知识
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# ============================================================================
# 配置
# ============================================================================

class Config:
    """项目配置"""
    DATA_DIR = "data"
    OUTPUT_DIR = "output"
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

# 确保目录存在
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 数据生成（模拟销售数据）
# ============================================================================

def generate_sales_data(n_records: int = 1000) -> pd.DataFrame:
    """生成模拟销售数据"""
    np.random.seed(42)
    
    # 日期范围
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_records)]
    
    # 产品类别
    categories = ['电子产品', '服装', '食品', '家居', '图书']
    
    # 城市列表
    cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安']
    
    # 生成数据
    data = {
        'order_id': range(1, n_records + 1),
        'date': dates,
        'category': np.random.choice(categories, n_records),
        'city': np.random.choice(cities, n_records),
        'quantity': np.random.randint(1, 10, n_records),
        'unit_price': np.random.uniform(10, 1000, n_records).round(2),
        'customer_id': np.random.randint(1001, 2001, n_records),
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    
    return df

# ============================================================================
# 数据分析类
# ============================================================================

class SalesAnalyzer:
    """销售数据分析器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._prepare_data()
    
    def _prepare_data(self):
        """数据预处理"""
        # 转换日期
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 提取日期组件
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['weekday'] = self.df['date'].dt.dayofweek
        self.df['weekday_name'] = self.df['date'].dt.day_name()
    
    def basic_stats(self):
        """基本统计"""
        print("\n" + "=" * 50)
        print("基本统计信息")
        print("=" * 50)
        
        print(f"\n数据概览:")
        print(f"  - 订单总数: {len(self.df)}")
        print(f"  - 时间范围: {self.df['date'].min()} 至 {self.df['date'].max()}")
        print(f"  - 总销售额: ¥{self.df['total_amount'].sum():,.2f}")
        print(f"  - 平均订单金额: ¥{self.df['total_amount'].mean():,.2f}")
        
        print(f"\n数值列统计:")
        print(self.df[['quantity', 'unit_price', 'total_amount']].describe())
    
    def category_analysis(self):
        """类别分析"""
        print("\n" + "=" * 50)
        print("类别分析")
        print("=" * 50)
        
        # 按类别统计
        category_stats = self.df.groupby('category').agg({
            'order_id': 'count',
            'quantity': 'sum',
            'total_amount': 'sum'
        }).rename(columns={
            'order_id': '订单数',
            'quantity': '销售数量',
            'total_amount': '销售额'
        })
        
        category_stats['销售额占比'] = category_stats['销售额'] / category_stats['销售额'].sum() * 100
        
        print(category_stats.sort_values('销售额', ascending=False))
        
        return category_stats
    
    def city_analysis(self):
        """城市分析"""
        print("\n" + "=" * 50)
        print("城市分析")
        print("=" * 50)
        
        city_stats = self.df.groupby('city').agg({
            'order_id': 'count',
            'total_amount': 'sum'
        }).rename(columns={
            'order_id': '订单数',
            'total_amount': '销售额'
        })
        
        print(city_stats.sort_values('销售额', ascending=False).head(5))
        
        return city_stats
    
    def time_analysis(self):
        """时间分析"""
        print("\n" + "=" * 50)
        print("时间分析")
        print("=" * 50)
        
        # 月度销售额
        monthly = self.df.groupby('month')['total_amount'].sum()
        print("\n月度销售额:")
        print(monthly)
        
        # 星期销售额
        weekday = self.df.groupby('weekday_name')['total_amount'].mean()
        print("\n各星期平均销售额:")
        print(weekday)
        
        return monthly, weekday

# ============================================================================
# 可视化类
# ============================================================================

class SalesVisualizer:
    """销售数据可视化"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_category_sales(self, df: pd.DataFrame):
        """类别销售额柱状图"""
        category_sales = df.groupby('category')['total_amount'].sum().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        category_sales.plot(kind='barh', ax=ax, color='steelblue')
        
        ax.set_xlabel('销售额 (元)')
        ax.set_ylabel('类别')
        ax.set_title('各产品类别销售额')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'category_sales.png'), dpi=100)
        plt.close()
        print("图表已保存: category_sales.png")
    
    def plot_monthly_trend(self, df: pd.DataFrame):
        """月度销售趋势"""
        monthly = df.groupby('month')['total_amount'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly.plot(kind='line', ax=ax, marker='o', color='steelblue', linewidth=2)
        
        ax.set_xlabel('月份')
        ax.set_ylabel('销售额 (元)')
        ax.set_title('月度销售趋势')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'monthly_trend.png'), dpi=100)
        plt.close()
        print("图表已保存: monthly_trend.png")
    
    def plot_city_pie(self, df: pd.DataFrame):
        """城市销售占比饼图"""
        city_sales = df.groupby('city')['total_amount'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        city_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
        
        ax.set_ylabel('')
        ax.set_title('各城市销售额占比')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'city_pie.png'), dpi=100)
        plt.close()
        print("图表已保存: city_pie.png")

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("=" * 50)
    print("销售数据分析项目")
    print("=" * 50)
    
    # 生成数据
    print("\n正在生成模拟数据...")
    df = generate_sales_data(1000)
    
    # 保存原始数据
    df.to_csv(os.path.join(Config.DATA_DIR, 'sales_data.csv'), index=False)
    print(f"数据已保存: {Config.DATA_DIR}/sales_data.csv")
    
    # 数据分析
    analyzer = SalesAnalyzer(df)
    analyzer.basic_stats()
    analyzer.category_analysis()
    analyzer.city_analysis()
    analyzer.time_analysis()
    
    # 数据可视化
    visualizer = SalesVisualizer(Config.OUTPUT_DIR)
    visualizer.plot_category_sales(df)
    visualizer.plot_monthly_trend(df)
    visualizer.plot_city_pie(df)
    
    print("\n" + "=" * 50)
    print("分析完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
