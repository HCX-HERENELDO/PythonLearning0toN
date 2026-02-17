#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
模块名称：训练与验证
学习目标：掌握 PyTorch 模型训练和验证流程
PyCharm 技巧：学习监控训练过程
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============================================================================
# 第一部分：训练流程概述
# ============================================================================
"""
【概念讲解】
模型训练的标准流程：
1. 准备数据
2. 定义模型
3. 定义损失函数
4. 定义优化器
5. 训练循环
   - 前向传播
   - 计算损失
   - 反向传播
   - 更新参数
6. 验证和测试
"""

# ============================================================================
# 第二部分：准备数据和模型
# ============================================================================

# 生成模拟数据
np.random.seed(42)
torch.manual_seed(42)

# 训练数据
X_train = torch.randn(800, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()

# 验证数据
X_val = torch.randn(100, 10)
y_val = (X_val[:, 0] + X_val[:, 1] > 0).long()

# 测试数据
X_test = torch.randn(100, 10)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).long()

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 定义模型
class SimpleClassifier(nn.Module):
    """简单分类器"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# 创建模型
model = SimpleClassifier(10, 64, 2)
print(f"模型结构:\n{model}")

# ============================================================================
# 第三部分：训练函数
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
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
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# ============================================================================
# 第四部分：完整训练循环
# ============================================================================

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练历史
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# 训练循环
num_epochs = 20
best_val_acc = 0.0

print("\n开始训练:")
for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # 验证
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # 更新学习率
    scheduler.step()
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    # 打印进度
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ============================================================================
# 第五部分：可视化训练过程
# ============================================================================

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("\n训练曲线已保存到 training_history.png")
    
except ImportError:
    print("matplotlib 未安装，跳过可视化")

# ============================================================================
# 第六部分：测试和评估
# ============================================================================

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 测试
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"\n测试结果: Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")

# ----------------------------------------------------------------------------
# 详细评估
# ----------------------------------------------------------------------------

from collections import Counter

def evaluate_details(model, dataloader, device):
    """详细评估"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    return all_preds, all_labels

preds, labels = evaluate_details(model, test_loader, device)

# 打印分类报告
print("\n分类结果:")
print(f"预测分布: {Counter(preds)}")
print(f"真实分布: {Counter(labels)}")

# ============================================================================
# 第七部分：训练技巧
# ============================================================================

"""
【训练技巧总结】

1. 数据相关：
   - 数据增强
   - 归一化
   - 类别平衡

2. 模型相关：
   - 权重初始化
   - Dropout
   - Batch Normalization

3. 优化相关：
   - 学习率调度
   - 梯度裁剪
   - 权重衰减

4. 正则化：
   - L1/L2 正则化
   - Early Stopping
   - 模型集成

5. 监控：
   - TensorBoard
   - 学习曲线
   - 验证集监控
"""

# 清理
import os
if os.path.exists('best_model.pth'):
    os.remove('best_model.pth')
if os.path.exists('training_history.png'):
    os.remove('training_history.png')

# ============================================================================
# 本节小结
# ============================================================================
"""
✅ 掌握的知识点：
1. 训练流程
2. 训练函数封装
3. 验证函数
4. 完整训练循环
5. 学习率调度
6. 模型保存和加载
7. 训练可视化
8. 模型评估

➡️ 下一节：实战案例
"""

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("训练与验证模块学习完成！")
    print("=" * 60)
