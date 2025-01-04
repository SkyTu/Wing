import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys
import numpy as np
import struct 

def read_file(filename):
    with open(filename, 'rb') as f:
        file_data = f.read()
    numbers = np.frombuffer(file_data, dtype=np.int64)  # dtype根据数据类型调整
    return numbers

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        
        # 第一层卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        # Flatten层
        self.flatten = nn.Flatten()
        
        # 全连接层
        self.fc5 = nn.Linear(64 , 10)  # 假设输入图像是32x32大小的

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.flatten(x)
        x = self.fc5(x)
        
        return x


def init_weights(model, weight_list, fraction):
    # 获取权重列表中的索引
    idx = 0

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            # 获取该参数的形状
            shape = param.shape
            # 计算当前参数需要的权重数量
            num_weights = np.prod(shape)

            # 获取对应的权重列表的部分，并调整为相应的形状
            weights = np.array(weight_list[idx: idx + num_weights]/(2**fraction)).reshape(shape)
            print(weights)
            # 将权重赋值到模型参数
            param.data.copy_(torch.tensor(weights, dtype=param.dtype))

            # 更新索引
            idx += num_weights

    print(f"Total number of weights set: {idx}")

def export_weights_to_u64_dat(model, fraction, filename='weights_u64.dat'):
    weight_list = []

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Exporting {name}")

            # 获取权重并转换为定点数表示
            weight = param.data.cpu().numpy()  # 转为 NumPy 数组（如果模型在 GPU 上）
            
            # 将权重转换为定点数表示，乘以 2^fraction
            weight = weight * (2 ** fraction)

            # 将定点数权重转换为 u64 类型并展平
            weight_u64 = weight.astype(np.uint64).flatten()

            # 添加到权重列表
            weight_list.append(weight_u64)

    # 将所有权重组合成一个大的 NumPy 数组
    weight_array = np.concatenate(weight_list)

    # 保存为二进制文件 (.dat)
    with open(filename, 'wb') as f:
        f.write(weight_array.tobytes())  # 将 u64 数组转换为字节并保存

    print(f"Model weights saved as fixed-point representation to '{filename}'")


def load_data(data_dir, batch_size=64):
    # 定义转换：先将图像转为张量，再归一化到 [0, 1] 范围内
    transform = transforms.Compose([
        transforms.ToTensor(),  # 自动将图像从 [0, 255] 映射到 [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 对每个通道进行归一化
    ])
    
    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 测试
filename = 'weights_u64.dat'
weights = read_file(filename)
# Usage example
data_dir = "./datasets/cifar"  # Set this to your actual data directory
train_loader, test_loader = load_data(data_dir, 64)
model = CNN3()
# 假设这是你的模型
fraction = 24  # 定点数的分数部分位数
# export_weights_to_u64_dat(model, fraction)
init_weights(model, weights, fraction)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.015625, momentum=0.90625)

# Training loop (for demonstration)
for epoch in range(2):  # Set to your desired number of epochs
    model.train()
    cnt = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        cnt += 1
        if cnt % 10 == 0:
            accuracy = evaluate(model, test_loader)
            print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}, Test Accuracy: {accuracy:.2f}%")
        if cnt > 460:
            break

    # Evaluate accuracy on test set after each epoch
    
    print(f"Test Accuracy: {accuracy:.2f}%")