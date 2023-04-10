import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def from_path_to_df(in_path):
    """Read experiment txt files."""
    with in_path.open('r') as f:
        cur_df = pd.read_csv(f, sep='\s+', header=None, skiprows=1)
        if len(cur_df.columns) == 4:
            cur_df.columns = ['timestamp_ns', 'x', 'y', 'z']
        else:
            old_cols = cur_df.columns.tolist()
            old_cols[0] = 'timestamp_ns'
            if len(cur_df.columns) == 8:
                old_cols[1] = 'x'
                old_cols[2] = 'y'
                old_cols[3] = 'z'
            cur_df.columns = old_cols
        cur_df['timestamp_s'] = cur_df['timestamp_ns'] / 1.0e9
        return cur_df.drop('timestamp_ns', axis=1).sort_values('timestamp_s')


# 读取文件
BASE_DIR = Path('D:\DataSet\RIDI\\archive\data_publish_v2')
# 把所有txt文件的名字全部存入 all_files_df
# DataFrame是一个表型的数据结构
all_files_df = pd.DataFrame({'path': list(BASE_DIR.glob('*/*.txt'))})
print("一共有", all_files_df.size, "数据")
# 新增列
all_files_df['exp_code'] = all_files_df['path'].map(lambda x: x.parent.stem)
all_files_df['activity'] = all_files_df['exp_code'].map(lambda x: '_'.join(x.split('_')[1:]))
all_files_df['person'] = all_files_df['exp_code'].map(lambda x: x.split('_')[0])
all_files_df['data_src'] = all_files_df['path'].map(lambda x: x.stem)
print(all_files_df.values)

data_df = all_files_df.pivot_table(values='path',
                                   columns='data_src',
                                   index=['activity', 'person'],
                                   aggfunc='first'). \
    reset_index(). \
    dropna(axis=1)  # remove mostly empty columns
# data_df.to_csv(path_or_buf='D:\DataSet\RIDI\Projetct\data.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)
# print(data_df)

# # 获取到第一行
# cur_exp = data_df.iloc[0]
# print(cur_exp.iloc[2:])

# # 字典类型 {“数据类型名称“, "数据DataFrame"}
# dict_df = {k: from_path_to_df(v)
#            for k, v in cur_exp.iloc[2:].items()}
# for k, v in dict_df.items():
#     print(k, v.shape, 'Framerate:{:2.1f}'.format(1 / (np.mean(v['timestamp_s'].diff()))))
# # dict_df['pose'].plot('timestamp_s')
# # plt.show()

# # 显示真实的轨迹
# fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
# # ax1.plot(la_df['pos_x'], la_df['pos_y'], '.-', label='Integrated Position')
# ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], '+-', label='Actual Pose')
# ax1.legend()
# ax1.axis('equal')
# plt.show()
#
# fig = plt.figure(figsize=(10, 10), dpi=300)
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], dict_df['pose']['z'], label='Actual Pose')
# ax1.legend()
# ax1.axis('auto')
# plt.show()
