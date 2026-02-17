#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：PyTorch 基础
学习目标：掌握 PyTorch 张量操作和自动求导机制
PyCharm 技巧：学习使用 GPU 加速调试
============================================================================
"""

# ============================================================================
# 第一部分：PyTorch 简介
# ============================================================================
"""
【概念讲解】
PyTorch 是 Facebook 开发的深度学习框架，特点：
1. 动态计算图 - 更灵活，便于调试
2. GPU 加速 - 支持 CUDA
3. Pythonic - 与 NumPy 类似的 API
4. 生态丰富 - TorchVision、TorchText 等

安装：
CPU: pip install torch torchvision
GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""

import torch
import numpy as np

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 第二部分：张量（Tensor）
# ============================================================================
"""
【概念讲解】
张量是 PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，
但支持 GPU 加速和自动求导。
"""

# ----------------------------------------------------------------------------
# 创建张量
# ----------------------------------------------------------------------------

# 从列表创建
x = torch.tensor([1, 2, 3, 4, 5])
print(f"一维张量: {x}")
print(f"形状: {x.shape}")
print(f"数据类型: {x.dtype}")

# 从 NumPy 创建
np_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(np_array)
print(f"从 NumPy 创建: {tensor_from_numpy}")

# 创建特定形状的张量
zeros = torch.zeros(3, 4)  # 全零
ones = torch.ones(2, 3)    # 全一
random = torch.rand(2, 3)  # 随机 [0, 1)
randn = torch.randn(2, 3)  # 标准正态分布

print(f"全零张量:\n{zeros}")
print(f"随机张量:\n{random}")

# 创建序列
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # 均匀分布

print(f"arange: {arange}")
print(f"linspace: {linspace}")

# 指定数据类型
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
long_tensor = torch.tensor([1, 2, 3], dtype=torch.long)

print(f"float32: {float_tensor.dtype}")
print(f"long: {long_tensor.dtype}")

# ----------------------------------------------------------------------------
# 张量操作
# ----------------------------------------------------------------------------

# 索引和切片
x = torch.arange(12).reshape(3, 4)
print(f"二维张量:\n{x}")
print(f"第一行: {x[0]}")
print(f"第一列: {x[:, 0]}")
print(f"切片: {x[0:2, 1:3]}")

# 形状操作
x = torch.arange(12)
print(f"原始形状: {x.shape}")

# reshape - 改变形状
x_reshaped = x.reshape(3, 4)
print(f"reshape: {x_reshaped.shape}")

# view - 共享内存的 reshape
x_view = x.view(2, 6)
print(f"view: {x_view.shape}")

# squeeze - 去除大小为1的维度
x = torch.randn(1, 3, 1, 4)
print(f"squeeze 前: {x.shape}")
x_squeezed = x.squeeze()
print(f"squeeze 后: {x_squeezed.shape}")

# unsqueeze - 添加维度
x = torch.randn(3, 4)
x_unsqueezed = x.unsqueeze(0)  # 在第0维添加
print(f"unsqueeze 后: {x_unsqueezed.shape}")

# 转置
x = torch.randn(2, 3)
print(f"转置前:\n{x}")
print(f"转置后:\n{x.T}")

# 矩阵乘法
a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.mm(a, b)  # 或 a @ b
print(f"矩阵乘法结果形状: {c.shape}")

# ----------------------------------------------------------------------------
# 数学运算
# ----------------------------------------------------------------------------

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 基本运算
print(f"加法: {x + y}")
print(f"减法: {x - y}")
print(f"乘法: {x * y}")
print(f"除法: {x / y}")

# 数学函数
x = torch.tensor([0.0, 0.5, 1.0])
print(f"sin: {torch.sin(x)}")
print(f"cos: {torch.cos(x)}")
print(f"exp: {torch.exp(x)}")
print(f"log: {torch.log(x + 1)}")  # 避免 log(0)
print(f"sqrt: {torch.sqrt(x)}")

# 聚合操作
x = torch.randn(3, 4)
print(f"求和: {x.sum()}")
print(f"均值: {x.mean()}")
print(f"最大值: {x.max()}")
print(f"最小值: {x.min()}")

# 沿维度聚合
print(f"按行求和: {x.sum(dim=1)}")
print(f"按列求均值: {x.mean(dim=0)}")

# ============================================================================
# 第三部分：自动求导（Autograd）
# ============================================================================
"""
【概念讲解】
Autograd 是 PyTorch 的自动微分引擎，可以自动计算梯度。
只需要设置 requires_grad=True，PyTorch 会跟踪所有操作。
"""

# ----------------------------------------------------------------------------
# 基本自动求导
# ----------------------------------------------------------------------------

# 创建需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
print(f"x: {x}")
print(f"requires_grad: {x.requires_grad}")

# 定义计算
y = x ** 2
z = y + 3

print(f"y = x^2 = {y}")
print(f"z = y + 3 = {z}")

# 反向传播计算梯度
z.backward()

# dz/dx = dz/dy * dy/dx = 1 * 2x = 2x = 4
print(f"梯度 dz/dx: {x.grad}")

# ----------------------------------------------------------------------------
# 多变量求导
# ----------------------------------------------------------------------------

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

z = x * y + x ** 2
z.backward()

print(f"dz/dx = y + 2x = {x.grad}")  # y + 2x = 2 + 2 = 4
print(f"dz/dy = x = {y.grad}")       # x = 1

# ----------------------------------------------------------------------------
# 停止梯度跟踪
# ----------------------------------------------------------------------------

x = torch.tensor([1.0], requires_grad=True)

# 方法1：detach()
y = x ** 2
y_detached = y.detach()  # 创建不需要梯度的新张量

# 方法2：torch.no_grad()
with torch.no_grad():
    z = x ** 2
    print(f"no_grad 中的 z.requires_grad: {z.requires_grad}")

# ============================================================================
# 第四部分：GPU 加速
# ============================================================================

# ----------------------------------------------------------------------------
# 设备管理
# ----------------------------------------------------------------------------

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 将张量移动到 GPU
x = torch.randn(3, 3)
x_gpu = x.to(device)
print(f"GPU 张量设备: {x_gpu.device}")

# 在 GPU 上进行运算
if torch.cuda.is_available():
    y_gpu = torch.randn(3, 3).to(device)
    z_gpu = x_gpu + y_gpu
    print(f"GPU 运算结果:\n{z_gpu}")
    
    # 移回 CPU
    z_cpu = z_gpu.cpu()
    # 或 z_cpu = z_gpu.to('cpu')

# ----------------------------------------------------------------------------
# 性能对比
# ----------------------------------------------------------------------------

import time

def matrix_multiply_test(device, size=1000):
    """矩阵乘法性能测试"""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    c = a @ b
    
    # 计时
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        c = a @ b
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    return elapsed

# CPU 测试
cpu_time = matrix_multiply_test(torch.device('cpu'))
print(f"CPU 时间: {cpu_time:.4f}秒")

# GPU 测试
if torch.cuda.is_available():
    gpu_time = matrix_multiply_test(torch.device('cuda'))
    print(f"GPU 时间: {gpu_time:.4f}秒")
    print(f"GPU 加速比: {cpu_time / gpu_time:.1f}x")

# ============================================================================
# 第五部分：数据加载
# ============================================================================

from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------
# 自定义数据集
# ----------------------------------------------------------------------------

class CustomDataset(Dataset):
    """自定义数据集示例"""
    
    def __init__(self, size=100):
        # 生成示例数据
        self.x = torch.randn(size, 10)
        self.y = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 创建数据集和数据加载器
dataset = CustomDataset(size=100)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"数据集大小: {len(dataset)}")
print(f"批次数: {len(dataloader)}")

# 遍历数据
for batch_x, batch_y in dataloader:
    print(f"批次 x 形状: {batch_x.shape}")
    print(f"批次 y 形状: {batch_y.shape}")
    break

# ============================================================================
# 第六部分：简单神经网络
# ============================================================================

import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------------
# 定义神经网络
# ----------------------------------------------------------------------------

class SimpleNet(nn.Module):
    """简单神经网络"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 创建模型
model = SimpleNet(input_size=10, hidden_size=20, num_classes=2)
print(f"模型结构:\n{model}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------------------------------------------------------
# 训练循环
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
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 训练
print("\n开始训练:")
train_model(model, dataloader, criterion, optimizer, num_epochs=5)

# ----------------------------------------------------------------------------
# 保存和加载模型
# ----------------------------------------------------------------------------

# 保存模型
torch.save(model.state_dict(), 'simple_net.pth')
print("模型已保存")

# 加载模型
loaded_model = SimpleNet(input_size=10, hidden_size=20, num_classes=2)
loaded_model.load_state_dict(torch.load('simple_net.pth'))
loaded_model.eval()
print("模型已加载")

# 清理
import os
if os.path.exists('simple_net.pth'):
    os.remove('simple_net.pth')

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. PyTorch 安装和环境配置
2. 张量的创建和操作
3. 自动求导机制
4. GPU 加速使用
5. 数据集和数据加载器
6. 神经网络定义和训练
7. 模型保存和加载

🔧 PyCharm 技巧：
1. 使用 CUDA 调试
2. Variables 面板查看张量
3. Structure 面板查看模型结构
4. 使用 Scientific Mode 查看图表

➡️ 下一节：神经网络深入
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PyTorch 基础模块学习完成！")
    print("=" * 60)
