import os

import numpy as np
from datetime import datetime
import pandas as pd


def save_data_to_txt(data, filename):
    """
    将二维数组保存为 txt 文件，并在第一行输出生成时间。

    参数：
    -- data: numpy 数组，二维数组，形状为 (n, m)，n 行 m 列的数据
    -- filename: str，保存的文件名

    示例：
    save_data_to_txt(data, 'output.txt')
    """

    # 逐行输出并保存为 txt 文件
    # 获取当前时间
    current_time = datetime.now()

    # 获取当前时间的字符串表示

    with open(filename, 'w') as file:
        for i, row in enumerate(data):
            if i == 0:
                file.write(f'# Create Time :{current_time} Create By haoyu99  \n')
            # 将数据保存到 txt 文件
            np.savetxt(file, [row], fmt='%.6f')


def from_path_to_df(in_path):
    """
    从in_path读取txt文件，返回数据的dp类型
    :param in_path: 文件的真实路径
    :return: 返回一个DataFrame类型的数据
    """
    with in_path.open('r') as f:
        if os.path.getsize(in_path) == 0:
            print(f"Warning: Empty file found at {in_path}. Skipping.")
            return None
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