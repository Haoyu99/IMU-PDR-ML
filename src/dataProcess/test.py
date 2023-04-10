import time

import open3d
import numpy as np
import pandas as pd
from pathlib import Path

def from_path_to_df(in_path):
    """Read experiment txt files."""
    with in_path.open('r') as f:
        cur_df = pd.read_csv(f, sep='\s+', header=None, skiprows=1)
        if len(cur_df.columns)==4:
            cur_df.columns = ['timestamp_ns', 'x', 'y', 'z']
        else:
            old_cols = cur_df.columns.tolist()
            old_cols[0] = 'timestamp_ns'
            if len(cur_df.columns)==8:
                old_cols[1] = 'x'
                old_cols[2] = 'y'
                old_cols[3] = 'z'
            cur_df.columns = old_cols
        cur_df['timestamp_s'] = cur_df['timestamp_ns']/1.0e9
        return cur_df.drop('timestamp_ns', axis=1).sort_values('timestamp_s')

# 读取文件
# inputPath = r"D:\DataSet\RIDI\archive\data_publish_v2";
BASE_DIR = Path('D:\DataSet\RIDI\\archive\data_publish_v2')
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
                        aggfunc='first').\
    reset_index().\
    dropna(axis=1) # remove mostly empty columns
# data_df.to_csv(path_or_buf='D:\DataSet\RIDI\Projetct\data.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)
# print(data_df)

# 获取到第一行
cur_exp = data_df.iloc[31]

# 字典类型 {“数据类型名称“, "数据DataFrame"}
dict_df = {k: from_path_to_df(v)
           for k, v in cur_exp.iloc[2:].items()}
# for k, v in dict_df.items():
#     print(k, v.shape, 'Framerate:{:2.1f}'.format(1/(np.mean(v['timestamp_s'].diff()))))
# print(len(dict_df['orientation'][1]))
ori = []
for i in range(len(dict_df['orientation'][1])):
    ori.append([dict_df['orientation'][1][i], dict_df['orientation'][2][i], dict_df['orientation'][3][i], dict_df['orientation'][4][i]])
print(ori)


vis = open3d.visualization.Visualizer()
vis.create_window(
    window_name="Phone",
    width=1440, height=1080
)
ctr = vis.get_view_control()
param = open3d.io.read_pinhole_camera_parameters("D:\DataSet\RIDI\Projetct\source\draw.json")
# Geometry追加
mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
    size=15, origin=[-2, -2, -2])

phone = open3d.io.read_point_cloud("iPhonex.pcd")
phone.paint_uniform_color([1,0.5,0])
vis.add_geometry(mesh_frame)
vis.add_geometry(phone)
ctr.convert_from_pinhole_camera_parameters(param)

   # 更新处理
while True:
    for i in range(0, len(ori),200):
        quart = np.array([ori[i][3],ori[i][0], ori[i][1], ori[i][2]]).T
        thQuart = phone.get_rotation_matrix_from_quaternion(quart)
        phone.rotate(thQuart)
        vis.update_geometry(phone)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)
        vis.capture_screen_image("pic/temp_%04d.jpg" % i)
        # 位置还原
        quart = np.array([ori[i][3],-ori[i][0], -ori[i][1], -ori[i][2]]).T
        thQuart = phone.get_rotation_matrix_from_quaternion(quart)
        phone.rotate(thQuart)


