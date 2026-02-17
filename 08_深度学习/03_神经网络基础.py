#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：神经网络基础
学习目标：使用 PyTorch 构建和训练神经网络
PyCharm 技巧：学习查看模型结构
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 第一部分：神经网络模块
# ============================================================================
"""
【概念讲解】
PyTorch 的 nn 模块提供了构建神经网络的各种组件：
- nn.Linear：全连接层
- nn.Conv2d：卷积层
- nn.ReLU、nn.Sigmoid：激活函数
- nn.MaxPool2d：池化层
- nn.Dropout：Dropout 层
"""

# ----------------------------------------------------------------------------
# 激活函数
# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# 常用激活函数
x = torch.linspace(-5, 5, 100)

# ReLU
relu = nn.ReLU()
y_relu = relu(x)

# Sigmoid
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)

# Tanh
tanh = nn.Tanh()
y_tanh = tanh(x)

# LeakyReLU
leaky_relu = nn.LeakyReLU(0.1)
y_leaky = leaky_relu(x)

print("激活函数定义完成")

# ============================================================================
# 第二部分：构建神经网络
# ============================================================================

# ----------------------------------------------------------------------------
# 使用 nn.Sequential
# ----------------------------------------------------------------------------

# 简单的全连接网络
model_sequential = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

print(f"Sequential 模型:\n{model_sequential}")

# ----------------------------------------------------------------------------
# 自定义神经网络类
# ----------------------------------------------------------------------------

class NeuralNetwork(nn.Module):
    """自定义神经网络"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out

# 创建模型
model = NeuralNetwork(input_size=784, hidden_size=256, num_classes=10)
print(f"\n自定义模型:\n{model}")

# 查看模型参数
print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# 第三部分：损失函数和优化器
# ============================================================================

# ----------------------------------------------------------------------------
# 常用损失函数
# ----------------------------------------------------------------------------

# 交叉熵损失（分类任务）
criterion_ce = nn.CrossEntropyLoss()

# 均方误差损失（回归任务）
criterion_mse = nn.MSELoss()

# 二元交叉熵损失（二分类）
criterion_bce = nn.BCELoss()

# ----------------------------------------------------------------------------
# 常用优化器
# ----------------------------------------------------------------------------

# SGD（随机梯度下降）
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam（自适应矩估计）
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# AdamW（带权重衰减的 Adam）
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# ============================================================================
# 第四部分：训练循环
# ============================================================================

# ----------------------------------------------------------------------------
# 准备数据
# ----------------------------------------------------------------------------

# 生成模拟数据
num_samples = 1000
X = torch.randn(num_samples, 784)
y = torch.randint(0, 10, (num_samples,))

# 创建数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----------------------------------------------------------------------------
# 训练函数
# ----------------------------------------------------------------------------

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    """训练模型"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 训练
print("\n开始训练:")
train_model(model, dataloader, criterion_ce, optimizer_adam, num_epochs=5)

# ============================================================================
# 第五部分：模型评估
# ============================================================================

def evaluate_model(model, dataloader):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

accuracy = evaluate_model(model, dataloader)
print(f"\n测试准确率: {accuracy:.2f}%")

# ============================================================================
# 第六部分：卷积神经网络（CNN）
# ============================================================================

class CNN(nn.Module):
    """卷积神经网络"""
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

cnn_model = CNN()
print(f"\nCNN 模型:\n{cnn_model}")

# ============================================================================
# 第七部分：模型保存与加载
# ============================================================================

# 保存模型
torch.save(model.state_dict(), 'model.pth')
print("\n模型已保存")

# 加载模型
loaded_model = NeuralNetwork(input_size=784, hidden_size=256, num_classes=10)
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
print("模型已加载")

# 清理
import os
if os.path.exists('model.pth'):
    os.remove('model.pth')

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. nn 模块的常用组件
2. 激活函数的使用
3. 自定义神经网络类
4. 损失函数和优化器
5. 训练循环的实现
6. 模型评估
7. CNN 构建

➡️ 下一节：模型训练实战
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("神经网络基础模块学习完成！")
    print("=" * 60)
