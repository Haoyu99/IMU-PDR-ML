from data_ridi import RIDIRawDataSequence,RIDIDataset,SequenceToSequenceDataset
import matplotlib.pyplot as plt
from data_utils import load_cached_sequences
from pathlib import Path
import pandas as pd

# 测试RIDIRawDataSequence类的加载
# data = RIDIRawDataSequence('D:\DataSet\RIDI\\archive\data_publish_v2\dan_bag1\processed',interval=200)
# print(data.get_aux().shape)


# 生成train_list 里面放的是每个用于训练的文件夹的名字
# BASE_DIR = 'D:\DataSet\RIDI\\archive\data_publish_v2'
# DIR = Path(BASE_DIR)
# train_list = pd.read_csv(BASE_DIR+'\list_train_publish_v2.txt')
# train_list = train_list['file_name']
# print(train_list)
# train_list.to_csv(BASE_DIR+'\\train_list.txt', sep='\t',index=False, header=False)
# 生成test_list 里面放的是每个用于训练的文件夹的名字
# test_list = pd.read_csv(BASE_DIR+'\list_test_publish_v2.txt')
# test_list = test_list['file_name']
# print(test_list)
# test_list.to_csv(BASE_DIR+'\\test_list.txt', sep='\t',index=False, header=False)


# 测试load_cached_sequences方法
# root_dir = 'D:\DataSet\RIDI\\archive\data_publish_v2'
# list_path = 'D:\DataSet\RIDI\\archive\data_publish_v2\\test_list2.txt'
root_dir = '/home/jiamingjie/zhanghaoyu/data'
list_path = '/home/jiamingjie/zhanghaoyu/data/test_list2.txt'
with open(list_path) as f:
    data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
seq_type = RIDIRawDataSequence
cache = root_dir+'\\cache'
# load_cached_sequences(seq_type,root_dir,data_list,cache)
# RIDIDataset 输入是[200 * 6 * 1] 输出[1 * 2 * 1]
# SeqToSeq 输入是[400 * 6 ] 输出[400 * 6 ]
data = SequenceToSequenceDataset(RIDIRawDataSequence,root_dir,data_list,cache,shuffle=False)
data.__getitem__(0)
print(data.index_map)
