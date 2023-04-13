import numpy as np
import pandas as pd
from pathlib import Path
import rotation
import file_utils
# 1. 生成全局坐标系下的 acce 和 gyro
# 2. 将需要的数据封装为csv文件
# 数据的根路径
BASE_DIR = 'D:\DataSet\RIDI\\archive\data_publish_v2'
DIR = Path(BASE_DIR)
# 加载根路径下所有的文件夹内的txt文件
all_files_df = pd.DataFrame({'path': list(DIR.glob('*/*.txt'))})
# 新建一个列 名字是"exp_code" 代表这个文件所属的文件夹名字
all_files_df['exp_code'] = all_files_df['path'].map(lambda x: x.parent.stem)
# 新建一个列 放人名
all_files_df['person'] = all_files_df['exp_code'].map(lambda x: x.split('_')[0])
# 新建一个列 从exp_code中找到活动类型
all_files_df['activity'] = all_files_df['exp_code'].map(lambda x: '_'.join(x.split('_')[1:]))
# 新建一个列 是数据的位置
all_files_df['data_src'] = all_files_df['path'].map(lambda x: x.stem)

data_df = all_files_df.pivot_table(values='path',
                                   columns='data_src',
                                   index=['activity', 'person'],
                                   aggfunc='first').reset_index().dropna(axis=1)


# 将data_df存为csv格式的文件
# data_df.to_csv(path_or_buf='D:\data_df.csv', sep=',', na_rep='', float_format=None, columns=None, header=True,
#                index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
#                chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict',
#                storage_options=None)

# 处理acce_glob gyro_glob数据,写出txt文件
# for i in range(0, len(data_df)):
#     cur_exp = data_df.iloc[i]
#     print(cur_exp)
#     path_acce_glob = cur_exp['acce_glob']
#     path_gyro_glob = cur_exp['gyro_glob']
#     dict_df = {k: file_utils.from_path_to_df(v) for k, v in cur_exp.iloc[2:].items()}
#     ori = np.column_stack((dict_df['orientation'][4], dict_df['orientation'][1], dict_df['orientation'][2], dict_df['orientation'][3]))
#     timestamp_s = np.array(dict_df['acce']['timestamp_s'])
#     timestamp_s = timestamp_s.reshape(-1, 1)  # 转换为 (n, 1) 维度的数据
#     R = rotation.get_rotation_matrix_from_quaternion(ori)
#     acce_IMU = np.array(dict_df['acce'])[:, 0:3]
#     acce_World = rotation.imu_data_to_world_coords(acce_IMU, R)
#     acce_World = np.concatenate((timestamp_s, acce_World), axis=1)
#     file_utils.save_data_to_txt(acce_World, path_acce_glob)
#     gyro_IMU = np.array(dict_df['gyro'])[:, 0:3]
#     gyro_World = rotation.imu_data_to_world_coords(gyro_IMU, R)
#     gyro_World = np.concatenate((timestamp_s, gyro_World), axis=1)
#     file_utils.save_data_to_txt(gyro_World, path_gyro_glob)

# 写出文件 csv格式
for i in range(0, len(data_df)):
    cur_exp = data_df.iloc[i]
    output_dir = BASE_DIR + '\\' + cur_exp['person'] + '_' + cur_exp['activity']
    all_data = pd.DataFrame()
    acce_path = cur_exp['acce']
    acce_data = file_utils.from_path_to_df(acce_path)
    # 添加时间戳
    all_data['time'] = acce_data['timestamp_s']
    # 添加acce_raw 数据
    all_data['acce_x'] = acce_data['x']
    all_data['acce_y'] = acce_data['y']
    all_data['acce_z'] = acce_data['z']

    # 添加acce_glob 数据
    acce_glob_path = cur_exp['acce_glob']
    acce_glob_data = file_utils.from_path_to_df(acce_glob_path)
    all_data['acce_glob_x'] = acce_glob_data['x']
    all_data['acce_glob_y'] = acce_glob_data['y']
    all_data['acce_glob_z'] = acce_glob_data['z']

    # 添加gyro 数据
    gyro_path = cur_exp['gyro']
    gyro_data = file_utils.from_path_to_df(gyro_path)
    all_data['gyro_x'] = gyro_data['x']
    all_data['gyro_y'] = gyro_data['y']
    all_data['gyro_z'] = gyro_data['z']

    # 添加gyro_glob 数据
    gyro_glob_path = cur_exp['gyro_glob']
    gyro_glob_data = file_utils.from_path_to_df(gyro_glob_path)
    all_data['gyro_glob_x'] = gyro_glob_data['x']
    all_data['gyro_glob_y'] = gyro_glob_data['y']
    all_data['gyro_glob_z'] = gyro_glob_data['z']

    # 添加ori数据
    ori_path = cur_exp['orientation']
    ori_data = file_utils.from_path_to_df(ori_path)
    all_data['ori_w'] = ori_data[4]
    all_data['ori_x'] = ori_data[1]
    all_data['ori_y'] = ori_data[2]
    all_data['ori_z'] = ori_data[3]

    # 添加pose信息
    pose_path = cur_exp['pose']
    pose_data = file_utils.from_path_to_df(pose_path)
    all_data['pos_x'] = pose_data['x']
    all_data['pos_y'] = pose_data['y']
    all_data.to_csv(path_or_buf=output_dir + '/data.csv', sep=',', na_rep='', float_format=None, columns=None,
                    header=True,
                    index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None,
                    quotechar='"',
                    chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict',
                    storage_options=None)
    print(output_dir, 'is success')

# # 将处理后的数据写入对应的txt中
# # 获取第一个切片
# cur_exp = data_df.iloc[0]
# path_acce_glob = cur_exp['acce_glob']
#
# # 用一个字典类型 来装数据{k(数据类型),v(实际的数据)}
# dict_df = {k: FileUtils.from_path_to_df(v) for k, v in cur_exp.iloc[2:].items()}
# for k, v in dict_df.items():
#     if v is not None:
#         print(k, v.shape)
# # 获得四元数数据
# ori = np.array([[dict_df['orientation'][i][j] for i in range(1, 5)] for j in range(len(dict_df['orientation'][1]))])
# # 获得时间戳进行格式转化
# timestamp_s = np.array(dict_df['acce']['timestamp_s'])
# timestamp_s = timestamp_s.reshape(-1, 1)  # 转换为 (n, 1) 维度的数据
# # 从四元数转化成为旋转矩阵
# R = rotation.get_rotation_matrix_from_quaternion(ori)
# # 获取acce数据
# acce_IMU = np.array(dict_df['acce'])[:, 0:3]
#
# # 把时间戳和转换坐标系后的数据封装
# acce_World = rotation.imu_data_to_world_coords(acce_IMU, R)
# acce_World = np.concatenate((timestamp_s, acce_World), axis=1)
#
# FileUtils.save_data_to_txt(acce_World, path_acce_glob)
