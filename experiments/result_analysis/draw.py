# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties

# # 定义读取文件的函数
# def read_accuracy(file_path):
#     accuracies = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             accuracies.append(float(line.strip()))  # 假设每行是一个浮点数
#     return accuracies

# # 读取两个路径下的accuracy.txt
# wing = "../wing/output/P0/training/loss/CNN2-2e-46b/accuracy.txt"
# orca = "../orca/output/P0/training/loss/CNN2-2e-46b/accuracy.txt"
# clear = "../cleartext/output/CNN2-2e-46b/accuracy.txt"

# wing_accuracy = read_accuracy(wing)
# orca_accuracy = read_accuracy(orca)
# cleartext_accuracy = read_accuracy(clear)

# # 创建迭代次数列表，假设每次读取的准确率是10次迭代后的结果
# iterations = [i * 10 for i in range(1, len(wing_accuracy) + 1)]

# # 创建折线图
# plt.figure(figsize=(10, 6))

# # 绘制红色实线（第一组数据）
# plt.plot(iterations, wing_accuracy, 'r-', label='Wing', linewidth=2)

# # 绘制黑色虚线（第二组数据）
# plt.plot(iterations, orca_accuracy, 'k--', label='Orca', linewidth=2)

# # 绘制黑色虚线（第二组数据）
# plt.plot(iterations, cleartext_accuracy, 'b--', label='Clear', linewidth=2)

# # 设置图表标题和标签
# # plt.title('Iteration vs Accuracy')
# plt.xlabel('Number of iterations', fontsize=18, fontweight='bold')
# plt.ylabel('Accuracy', fontsize=18, fontweight='bold')

# # 创建字体属性对象，用于图例
# font_properties = FontProperties(weight='bold',size=18)

# # 显示图例，并将图例字体设置为粗体
# plt.legend(prop=font_properties)
# # 显示网格
# plt.grid(True)

# # 保存图像为result.png
# plt.savefig('result.png', dpi=1200)

# # 显示图形
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 定义读取文件的函数
def read_accuracy(file_path):
    accuracies = []
    cnt = 0
    with open(file_path, 'r') as file:
        for line in file:
            accuracies.append(float(line.strip()))  # 假设每行是一个浮动数
            cnt += 1
            if(cnt >= 46):
                break
    return accuracies

# 读取两个路径下的accuracy.txt
wing = "../wing/output/P0/training/loss/CNN2-2e-46b-wing/accuracy.txt"
orca = "../orca/output/P0/training/loss/CNN2-2e-46b/accuracy.txt"
piranha = "../wing/output/P0/training/loss/CNN2-2e-46b-piranha/accuracy.txt"
clear = "../cleartext/output/CNN2-2e-46b/accuracy.txt"

wing_accuracy = read_accuracy(wing)
piranha_accuracy = read_accuracy(piranha)
orca_accuracy = read_accuracy(orca)
cleartext_accuracy = read_accuracy(clear)

# 创建迭代次数列表，假设每次读取的准确率是10次迭代后的结果
iterations = [i * 10 for i in range(1, len(wing_accuracy) + 1)]

# 创建主图
plt.figure(figsize=(10, 6))

# 绘制完整的准确率变化曲线
plt.plot(iterations, wing_accuracy, 'r-', label='Wing', linewidth=2)
plt.plot(iterations, orca_accuracy, 'k--', label='Orca', linewidth=2)
plt.plot(iterations, piranha_accuracy, 'g--', label='Piranha', linewidth=2)
plt.plot(iterations, cleartext_accuracy, 'b--', label='Clear', linewidth=2)


# 设置图表标题和标签
plt.xlabel('Number of iterations', fontsize=18, fontweight='bold')
plt.ylabel('Accuracy', fontsize=18, fontweight='bold')

# 创建字体属性对象，用于图例
font_properties = FontProperties(weight='bold', size=14)

# 显示图例，并将图例字体设置为粗体
plt.legend(prop=font_properties)

# 创建放大镜视图区域
# 放大显示最后100个batch的部分
print(len(wing_accuracy), iterations)
start_idx = len(wing_accuracy) - 10  # 确保数据范围不会越界
print(start_idx)
# 创建放大镜区域并将其放在右下角
ax = plt.gca()
axins = ax.inset_axes(bounds=[0.55, 0.5, 0.4, 0.3])  # 调整位置和大小


# 绘制放大镜区域中的数据（最后100个batch的准确率）
axins.plot(iterations[start_idx:], wing_accuracy[start_idx:], 'r-', linewidth=2)
axins.plot(iterations[start_idx:], orca_accuracy[start_idx:], 'k--', linewidth=2)
axins.plot(iterations[start_idx:], piranha_accuracy[start_idx:], 'g--', linewidth=2)
axins.plot(iterations[start_idx:], cleartext_accuracy[start_idx:], 'b--', linewidth=2)

# 设置放大镜区域的坐标轴范围
axins.set_xlim(iterations[start_idx], iterations[-1])
axins.set_ylim(min(wing_accuracy[start_idx:]) * 0.99, max(wing_accuracy[start_idx:]) * 1.01)

axins.grid(True)

# 显示网格
plt.grid(True)

# 保存图像为result_with_zoom.png
plt.savefig('mnist accuracy cnn2.png', dpi=1200)

# 显示图形
plt.show()

