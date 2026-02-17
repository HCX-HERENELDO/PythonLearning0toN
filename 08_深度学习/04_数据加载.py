#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：数据加载
学习目标：掌握 PyTorch 的 Dataset 和 DataLoader
PyCharm 技巧：学习数据管道调试
============================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# ============================================================================
# 第一部分：数据加载概念
# ============================================================================
"""
【概念讲解】
PyTorch 使用 Dataset 和 DataLoader 进行数据处理：

Dataset：
- 存储样本和标签
- 实现 __len__ 和 __getitem__ 方法

DataLoader：
- 批量加载数据
- 打乱数据顺序
- 多进程加载

数据加载流程：
原始数据 → Dataset → DataLoader → 训练循环
"""

# ============================================================================
# 第二部分：自定义 Dataset
# ============================================================================

# ----------------------------------------------------------------------------
# 简单 Dataset 示例
# ----------------------------------------------------------------------------

class SimpleDataset(Dataset):
    """简单的数据集示例"""
    
    def __init__(self, data, labels):
        """
        初始化数据集
        
        Args:
            data: 特征数据
            labels: 标签数据
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return self.data[idx], self.labels[idx]

# 创建数据集
data = np.random.randn(100, 10)  # 100个样本，每个10维
labels = np.random.randint(0, 3, 100)  # 3个类别

dataset = SimpleDataset(data, labels)

print(f"数据集大小: {len(dataset)}")
print(f"第一个样本: {dataset[0]}")

# ----------------------------------------------------------------------------
# 文件数据集示例
# ----------------------------------------------------------------------------

import os
from PIL import Image

class ImageDataset(Dataset):
    """图像数据集"""
    
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: 图像目录
            transform: 数据变换
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# ============================================================================
# 第三部分：DataLoader
# ============================================================================

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,      # 批量大小
    shuffle=True,       # 是否打乱
    num_workers=0,      # 进程数（Windows 建议0）
    drop_last=False     # 是否丢弃不完整批次
)

print(f"\nDataLoader:")
print(f"批次数: {len(dataloader)}")

# 遍历数据
for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"批次 {batch_idx}: 数据形状 {batch_data.shape}, 标签形状 {batch_labels.shape}")
    if batch_idx >= 2:
        break

# ============================================================================
# 第四部分：数据预处理
# ============================================================================

import torchvision.transforms as transforms

# ----------------------------------------------------------------------------
# 常用变换
# ----------------------------------------------------------------------------

# 组合变换
transform = transforms.Compose([
    # 图像变换
    transforms.Resize((224, 224)),          # 调整大小
    transforms.RandomHorizontalFlip(),       # 随机水平翻转
    transforms.RandomRotation(10),           # 随机旋转
    transforms.ColorJitter(                  # 颜色抖动
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),                   # 转为张量
    transforms.Normalize(                    # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------------------------------------------------------
# 自定义变换
# ----------------------------------------------------------------------------

class CustomTransform:
    """自定义变换"""
    
    def __init__(self, scale=1.0):
        self.scale = scale
    
    def __call__(self, x):
        return x * self.scale

# ============================================================================
# 第五部分：数据集划分
# ============================================================================

# 划分训练集、验证集、测试集
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\n数据集划分:")
print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")
print(f"测试集: {len(test_dataset)}")

# 创建各自的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# ============================================================================
# 第六部分：内置数据集
# ============================================================================

# ----------------------------------------------------------------------------
# MNIST 数据集
# ----------------------------------------------------------------------------

try:
    from torchvision import datasets
    
    # 下载并加载 MNIST
    mnist_train = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    mnist_test = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    print(f"\nMNIST 数据集:")
    print(f"训练集: {len(mnist_train)}")
    print(f"测试集: {len(mnist_test)}")
    print(f"图像形状: {mnist_train[0][0].shape}")
    
except Exception as e:
    print(f"MNIST 下载失败: {e}")

# ----------------------------------------------------------------------------
# 其他常用数据集
# ----------------------------------------------------------------------------

"""
torchvision.datasets 提供的数据集：
- CIFAR10/100
- ImageNet
- COCO
- VOC
- FashionMNIST
- etc.
"""

# ============================================================================
# 第七部分：数据加载技巧
# ============================================================================

# ----------------------------------------------------------------------------
# 处理类别不平衡
# ----------------------------------------------------------------------------

from torch.utils.data import WeightedRandomSampler

# 计算类别权重
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]

# 创建加权采样器
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

balanced_loader = DataLoader(
    dataset,
    batch_size=16,
    sampler=sampler
)

# ----------------------------------------------------------------------------
# 多进程加载（Linux/Mac）
# ----------------------------------------------------------------------------

# Windows 下 num_workers 应设为 0
# Linux/Mac 可以设置 > 0 加速加载
multi_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # 多进程
    pin_memory=True  # 锁页内存，加速 GPU 传输
)

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. Dataset 和 DataLoader 概念
2. 自定义 Dataset
3. DataLoader 参数
4. 数据预处理
5. 数据集划分
6. 内置数据集
7. 类别不平衡处理

➡️ 下一节：训练与验证
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("数据加载模块学习完成！")
    print("=" * 60)
