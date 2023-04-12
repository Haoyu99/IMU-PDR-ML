import time

import matplotlib.pyplot as plt
import open3d
import numpy as np
import pandas as pd
from pathlib import Path
import rotation
import FileUtils

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
# print(cur_exp)
# 字典类型 {“数据类型名称“, "数据DataFrame"}
dict_df = {k: FileUtils.from_path_to_df(v)
           for k, v in cur_exp.iloc[2:].items()}
for k, v in dict_df.items():
    print(k, v.shape, '采样频率:{:2.1f}'.format(1 / (np.mean(v['timestamp_s'].diff()))))

# 获取ori
ori = np.array([[dict_df['orientation'][i][j] for i in range(1, 5)] for j in range(len(dict_df['orientation'][1]))])
# print(ori)
# 时间戳(n * 1)
timestamp_s = np.array(dict_df['acce']['timestamp_s'])
timestamp_s = timestamp_s.reshape(-1, 1)  # 转换为 (n, 1) 维度的数据
print(timestamp_s.shape)

# 得到旋转矩阵
R = rotation.get_rotation_matrix_from_quaternion(ori)

acce_IMU = np.array(dict_df['acce'])[:, 0:3]

# 把时间戳和转换坐标系后的数据封装
acce_World = rotation.imu_data_to_world_coords(acce_IMU, R)
acce_World = np.concatenate((timestamp_s, acce_World), axis=1)
print(acce_World)
FileUtils.save_data_to_txt(acce_World, 'acce_world.txt')

# print(time)
# x = acce_World[:, 0]
# y = acce_World[:, 1]
# z = acce_World[:, 2]
# fig, ax1 = plt.subplots(1, 1)
# ax1.plot(time, x, label='X')
# ax1.plot(time, y, label='Y')
# ax1.plot(time, z, label='Z')
# ax1.legend()
# ax1.autoscale(axis=x) # 自适应 y 轴限制范围
# plt.show()


# # 原始的处于IMU坐标系下的acc数据
# dict_df['acce'].plot('timestamp_s')
# plt.show()

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
# quart = np.array([[0.78648,0.617316,-0.018206,-0.006167]]).T
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
