import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建三维坐标轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义三维向量
# origin = np.array([0, 0, 0])  # 向量起点
vector = np.array([0,0,1])
print(vector.shape)
axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 向量值

# 使用旋转矩阵对此向量进行旋转
R = np.array([[0.99926102, -0.01277727, -0.03625129],
               [-0.03217815, 0.23776585, -0.97078882],
               [0.02102333, 0.97123793, 0.237179]])

vector2 = np.matmul(R, vector)
print(vector2.shape)
# vector3 = np.matmul(vector2,R)
# print(vector3.shape)
# print(R.T)
# vector4 = np.matmul(R.T,vector2)
# print(vector4)
# # 绘制三维向量
#
# axis1 = np.dot(np.linalg.inv(R), axis)[0]
# print(axis1)
# # vector3 = np.matmul(vector2, R)
# # print(vector2)
# # print(vector3)
#
# ax.quiver(0, 0, 0, axis1[0][0], axis1[0][1], axis1[0][2], color='r', arrow_length_ratio=0.08)
# ax.quiver(0, 0, 0, axis1[1][0], axis1[1][1], axis1[1][2], color='g', arrow_length_ratio=0.08)
# ax.quiver(0, 0, 0, axis1[2][0], axis1[2][1], axis1[2][2], color='b', arrow_length_ratio=0.08)
#
# # 设置坐标轴范围
# ax.set_xlim([0, 3])
# ax.set_ylim([0, 3])
# ax.set_zlim([0, 3])
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # 显示图形
# plt.show()
