#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：实战案例 - MNIST 手写数字识别
学习目标：使用 PyTorch 完成图像分类项目
PyCharm 技巧：学习完整的深度学习项目开发流程
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# 项目概述
# ============================================================================
"""
【项目描述】
MNIST 是经典的手写数字识别数据集：
- 60,000 张训练图片
- 10,000 张测试图片
- 图片大小：28x28 灰度图
- 类别：0-9 共10个数字

目标：构建 CNN 模型，实现高精度数字识别
"""

# ============================================================================
# 第一部分：配置和准备
# ============================================================================

# 配置
class Config:
    # 数据
    batch_size = 64
    num_workers = 0  # Windows 设为 0
    
    # 模型
    num_classes = 10
    
    # 训练
    num_epochs = 10
    learning_rate = 0.001
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"使用设备: {Config.device}")

# ============================================================================
# 第二部分：数据准备
# ============================================================================

# 数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

# 下载数据集
print("下载 MNIST 数据集...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 划分训练集和验证集
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=Config.num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.batch_size,
    num_workers=Config.num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.batch_size,
    num_workers=Config.num_workers
)

print(f"数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

# 可视化样本
def show_samples(loader, num_samples=8):
    """显示样本图片"""
    images, labels = next(iter(loader))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
    for i in range(num_samples):
        img = images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=100)
    plt.close()
    print("样本图片已保存到 mnist_samples.png")

show_samples(train_loader)

# ============================================================================
# 第三部分：定义模型
# ============================================================================

class MNISTClassifier(nn.Module):
    """MNIST 分类器 - CNN"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 7 -> 3
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 创建模型
model = MNISTClassifier(Config.num_classes).to(Config.device)
print(f"\n模型结构:\n{model}")

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

# ============================================================================
# 第四部分：训练配置
# ============================================================================

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# ============================================================================
# 第五部分：训练函数
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), correct / total

# ============================================================================
# 第六部分：训练循环
# ============================================================================

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

print("\n开始训练:")
for epoch in range(Config.num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, Config.device
    )
    
    # 验证
    val_loss, val_acc = evaluate(
        model, val_loader, criterion, Config.device
    )
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # 记录
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_mnist_model.pth')
    
    # 打印进度
    print(f"Epoch [{epoch+1}/{Config.num_epochs}] "
          f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
          f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

# ============================================================================
# 第七部分：评估和可视化
# ============================================================================

# 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curve')
axes[0].legend()

axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig('mnist_training.png', dpi=100)
plt.close()
print("\n训练曲线已保存到 mnist_training.png")

# 测试最佳模型
checkpoint = torch.load('best_mnist_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = evaluate(model, test_loader, criterion, Config.device)
print(f"\n测试结果: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")

# ============================================================================
# 第八部分：预测示例
# ============================================================================

def predict_and_show(model, loader, device, num_samples=10):
    """预测并显示结果"""
    model.eval()
    images, labels = next(iter(loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 显示
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f'Pred: {predicted[i].item()}, True: {labels[i]}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=100)
    plt.close()
    print("预测结果已保存到 mnist_predictions.png")

predict_and_show(model, test_loader, Config.device)

# ============================================================================
# 清理
# ============================================================================

for f in ['best_mnist_model.pth', 'mnist_samples.png', 'mnist_training.png', 'mnist_predictions.png']:
    if os.path.exists(f):
        os.remove(f)

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 项目总结：
1. 完整的深度学习项目流程
2. CNN 模型设计
3. 数据预处理和增强
4. 训练和验证
5. 模型保存和加载
6. 结果可视化

🎉 恭喜完成深度学习模块！
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MNIST 实战案例学习完成！")
    print("=" * 60)
