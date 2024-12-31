import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys
import numpy as np
import struct 

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        # 定义各层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()

        # FC layers: Fully connected layers
        self.fc4 = nn.Linear(256, 128)  # Assuming the output of the flatten layer is 256
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(128, 10)  # 10 output classes for classification

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        
        x = self.fc4(x)
        x = self.relu4(x)
        
        x = self.fc5(x)
        return x


def read_file(filename):
    with open(filename, 'rb') as f:
        file_data = f.read()
    numbers = np.frombuffer(file_data, dtype=np.int64)  # dtype根据数据类型调整
    print(len(numbers))
    return numbers

# 测试
filename = '../wing/weights/CNN2.dat'
weights = read_file(filename)


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
            # print(weights)
            # 将权重赋值到模型参数
            param.data.copy_(torch.tensor(weights, dtype=param.dtype))

            # 更新索引
            idx += num_weights

    print(f"Total number of weights set: {idx}")

def load_data(data_dir, batch_size=128):
    # 定义简单的转换，仅将图像从 [0, 255] 范围转换为 [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()  # 自动将图像从 [0, 255] 映射到 [0, 1]
    ])
    
    # 加载训练集和测试集
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

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

f = 24

# Usage example
data_dir = "./datasets/mnist"  # Set this to your actual data directory
train_loader, test_loader = load_data(data_dir, 128)
model = CNN2()
init_weights(model, weights, 24)

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
            print(f"Epoch {epoch + 1}, Batch: {cnt}, Loss: {loss.item()}")
            accuracy = evaluate(model, test_loader)
            with open(f'output/CNN2-2e-46b/accuracy.txt', 'a') as f:
                f.write(f'{accuracy:.2f}\n')
        if cnt > 460:
            break

    # Evaluate accuracy on test set after each epoch
    
    print(f"Test Accuracy: {accuracy:.2f}%")