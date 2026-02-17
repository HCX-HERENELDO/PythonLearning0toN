#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
项目名称：图像分类模型
项目描述：使用 PyTorch 构建图像分类 CNN 模型
学习目标：综合运用深度学习知识完成实际项目
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
from datetime import datetime

# ============================================================================
# 配置
# ============================================================================

class Config:
    """项目配置"""
    # 数据
    batch_size = 64
    num_workers = 0  # Windows 设为 0
    
    # 模型
    num_classes = 10
    
    # 训练
    num_epochs = 15
    learning_rate = 0.001
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出
    output_dir = "output"
    model_path = "cifar10_model.pth"

# ============================================================================
# 数据准备
# ============================================================================

def prepare_data():
    """准备 CIFAR-10 数据集"""
    print("准备 CIFAR-10 数据集...")
    
    # 数据增强（训练集）
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 测试集变换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 下载并加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size,
        shuffle=True, num_workers=Config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size,
        shuffle=False, num_workers=Config.num_workers
    )
    
    # CIFAR-10 类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")
    
    return train_loader, test_loader, classes

# ============================================================================
# 模型定义
# ============================================================================

class CIFAR10Classifier(nn.Module):
    """CIFAR-10 分类器 - CNN"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 卷积层
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
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
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), correct / total

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("=" * 50)
    print("CIFAR-10 图像分类项目")
    print("=" * 50)
    print(f"使用设备: {Config.device}")
    
    # 准备数据
    train_loader, test_loader, classes = prepare_data()
    
    # 创建模型
    model = CIFAR10Classifier(Config.num_classes).to(Config.device)
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_acc = 0.0
    
    # 创建输出目录
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 训练循环
    print("\n开始训练:")
    for epoch in range(Config.num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, Config.device
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 
                      os.path.join(Config.output_dir, Config.model_path))
        
        print(f"Epoch [{epoch+1}/{Config.num_epochs}] "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Test: {test_loss:.4f}/{test_acc:.4f}")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['test_acc'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, 'training_curve.png'), dpi=100)
    plt.close()
    
    print(f"\n训练完成!")
    print(f"最佳测试准确率: {best_acc:.4f}")
    print(f"模型已保存: {Config.output_dir}/{Config.model_path}")

if __name__ == "__main__":
    main()
