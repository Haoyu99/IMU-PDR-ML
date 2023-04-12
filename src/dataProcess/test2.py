import numpy as np
import pandas as pd
from pathlib import Path
import rotation
import FileUtils

# 数据的根路径
BASE_DIR = Path('D:\DataSet\RIDI\\archive\data_publish_v2')
# 加载根路径下所有的文件夹内的txt文件
all_files_df = pd.DataFrame({'path': list(BASE_DIR.glob('*/*.txt'))})
# 新建一个列 名字是"exp_code" 代表这个文件所属的文件夹名字
all_files_df['exp_code'] = all_files_df['path'].map(lambda x: x.parent.stem)
# 新建一个列 放人名
all_files_df['person'] = all_files_df['exp_code'].map(lambda x: x.split('_')[0])
# 新建一个列 从exp_code中找到活动类型
all_files_df['activity'] = all_files_df['exp_code'].map(lambda x: x.split('_')[1])  # 按照下划线切割
# 新建一个列 是数据的位置
all_files_df['data_src'] = all_files_df['path'].map(lambda x: x.stem)

data_df = all_files_df.pivot_table(values='path',
                                   columns='data_src',
                                   index=['activity', 'person'],
                                   aggfunc='first').reset_index().dropna(axis=1)
print(len(data_df))



# data_df.to_csv(path_or_buf='D:\data_df.csv', sep=',', na_rep='', float_format=None, columns=None, header=True,
#                index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"',
#                chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict',
#                storage_options=None)

# 处理acce_glob数据
for i in range(1,len(data_df)-1):
    cur_exp = data_df.iloc[i]
    path_acce_glob = cur_exp['acce_glob']
    dict_df = {k: FileUtils.from_path_to_df(v) for k, v in cur_exp.iloc[2:].items()}
    ori = np.array([[dict_df['orientation'][i][j] for i in range(1, 5)] for j in range(len(dict_df['orientation'][1]))])
    timestamp_s = np.array(dict_df['acce']['timestamp_s'])
    timestamp_s = timestamp_s.reshape(-1, 1)  # 转换为 (n, 1) 维度的数据
    R = rotation.get_rotation_matrix_from_quaternion(ori)
    acce_IMU = np.array(dict_df['acce'])[:, 0:3]
    acce_World = rotation.imu_data_to_world_coords(acce_IMU, R)
    acce_World = np.concatenate((timestamp_s, acce_World), axis=1)
    FileUtils.save_data_to_txt(acce_World, path_acce_glob)

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