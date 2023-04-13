import time

import matplotlib.pyplot as plt
import open3d
import numpy as np
import pandas as pd
from pathlib import Path
import rotation
import file_utils

# 读取文件
BASE_DIR = Path('D:\DataSet\RIDI\\archive\data_publish_v2')
print(list(BASE_DIR.glob('*/*.txt')))
# 把所有txt文件的名字全部存入 all_files_df
# DataFrame是一个表型的数据结构
all_files_df = pd.DataFrame({'path': list(BASE_DIR.glob('*/*.txt'))})
all_files_df['exp_code'] = all_files_df['path'].map(lambda x: x.parent.stem)
all_files_df['activity'] = all_files_df['exp_code'].map(lambda x: '_'.join(x.split('_')[1:]))
all_files_df['person'] = all_files_df['exp_code'].map(lambda x: x.split('_')[0])
all_files_df['data_src'] = all_files_df['path'].map(lambda x: x.stem)

data_df = all_files_df.pivot_table(values='path',
                                   columns='data_src',
                                   index=['activity', 'person'],
                                   aggfunc='first'). \
    reset_index(). \
    dropna(axis=1)  # remove mostly empty columns
# data_df.to_csv(path_or_buf='D:\DataSet\RIDI\Projetct\data.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)
# print(data_df)

# 获取到第一行
cur_exp = data_df.iloc[0]
print(data_df)
# 字典类型 {“数据类型名称“, "数据DataFrame"}
dict_df = {k: file_utils.from_path_to_df(v)
           for k, v in cur_exp.iloc[2:].items()}

for k, v in dict_df.items():
    print(k, v.shape, '采样频率:{:2.1f}'.format(1 / (np.mean(v['timestamp_s'].diff()))))

# 获取ori  (ori的获取有问题)
print(dict_df['orientation'][1])
ori = np.column_stack((dict_df['orientation'][4], dict_df['orientation'][1], dict_df['orientation'][2], dict_df['orientation'][3]))
# 时间戳(n * 1)
timestamp_s = np.array(dict_df['acce']['timestamp_s'])
timestamp_s = timestamp_s.reshape(-1, 1)  # 转换为 (n, 1) 维度的数据
# print(timestamp_s.shape)

# 得到旋转矩阵
R = rotation.get_rotation_matrix_from_quaternion(ori)

acce_IMU = np.array(dict_df['acce'])[:, 0:3]
gyro_IMU = np.array(dict_df['gyro'])[:, 0:3]
# 把时间戳和转换坐标系后的数据封装
acce_World = rotation.imu_data_to_world_coords(acce_IMU, R)
gyro_glob = rotation.imu_data_to_world_coords(gyro_IMU,R)
acce_World = np.concatenate((timestamp_s, acce_World), axis=1)
gyro_glob = np.concatenate((timestamp_s, gyro_glob), axis=1)

# 绘制2d真实路径图
# fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
# ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], '+-', label='Actual Pose')
# ax1.legend()
# plt.show()
# 绘制3d真实路径图
# fig = plt.figure(figsize=(10, 10), dpi=300)
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], dict_df['pose']['z'], '+-', label='Actual Pose')
# ax1.legend()
# ax1.axis('equal')
# plt.show()



# # 绘制坐标转换后的acce
# x = acce_World[:, 1].reshape(-1, 1)
# y = acce_World[:, 2].reshape(-1, 1)
# z = acce_World[:, 3].reshape(-1, 1)
# fig, ax1 = plt.subplots(1, 1)
# ax1.plot(timestamp_s, x, label='X')
# ax1.plot(timestamp_s, y, label='Y')
# ax1.plot(timestamp_s, z, label='Z')
# ax1.legend()
# plt.show()

# 绘制转换后的gyro
# x = gyro_glob[:, 1].reshape(-1, 1)
# y = gyro_glob[:, 2].reshape(-1, 1)
# z = gyro_glob[:, 3].reshape(-1, 1)
# fig, ax1 = plt.subplots(1, 1)
# ax1.plot(timestamp_s, x, label='GYRO_X')
# ax1.plot(timestamp_s, y, label='GYRO_Y')
# ax1.plot(timestamp_s, z, label='GYRO_Z')
# ax1.legend()
# plt.show()

#
# # 原始的处于IMU坐标系下的acc数据
# dict_df['acce'].plot('timestamp_s')
# plt.show()
# # 原始的RIDI
# dict_df['gyro'].plot('timestamp_s')
# plt.show()


# 根据四元数 显示手机实时姿态
# vis = open3d.visualization.Visualizer()
# vis.create_window(window_name='phone', width=1440, height=1080)
# param = open3d.io.read_pinhole_camera_parameters("./resources/draw.json")
# # Geometry追加
# mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=15, origin=[-2, -2, -2])
#
# phone = open3d.io.read_point_cloud("./resources/iPhonex.pcd")
# # 填色
# phone.paint_uniform_color([1,0.5,0])
# vis.add_geometry(mesh_frame)
# vis.add_geometry(phone)
#
# quart = np.array([[0.421977 ,0.147928 ,0.271446 ,0.852273]]).T
# thQuart = phone.get_rotation_matrix_from_quaternion(quart)
# phone.rotate(thQuart)
# vis.run()  # user changes the view and press "q" to terminate

# vis.update_geometry(phone)
# vis.poll_events()
# vis.update_renderer()
# vis.capture_screen_image("./resources/iPhonex.jpg")

# 更新处理
# while True:
#     for i in range(0, len(ori),200):
#         quart = np.array([ori[i][3],ori[i][0], ori[i][1], ori[i][2]]).T
#         thQuart = phone.get_rotation_matrix_from_quaternion(quart)
#         phone.rotate(thQuart)
#         vis.update_geometry(phone)
#         vis.poll_events()
#         vis.update_renderer()
#         time.sleep(1)
#         vis.capture_screen_image("pic/temp_%04d.jpg" % i)
#         # 位置还原
#         quart = np.array([ori[i][3],-ori[i][0], -ori[i][1], -ori[i][2]]).T
#         thQuart = phone.get_rotation_matrix_from_quaternion(quart)
#         phone.rotate(thQuart)
