from abc import ABC, abstractmethod
import os
import warnings
from os import path as osp
import h5py
import json
import numpy as np



class DataSequence(ABC):
    """
    An abstract interface for compiled sequence.
    一个抽象类 用于处理数据
    """

    def __init__(self, **kwargs):
        super(DataSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass


def load_cached_sequences(seq_type, root_dir, data_list, cache_path, **kwargs):
    """
    从缓存中加载数据
    :param seq_type: 数据类型DataSequence的子类
    :param root_dir: 根路径
    :param data_list: 数据列表
    :param cache_path: 缓存地址
    :param kwargs: 额外参数
    :return:
    """

    if cache_path is not None and cache_path not in ['none', 'invalid', 'None']:
        if not osp.isdir(cache_path):
            # cache_path 不存在则新建
            os.makedirs(cache_path)
        # 查看缓存的config
        if osp.exists(osp.join(cache_path, 'config.json')):
            info = json.load(open(osp.join(cache_path, 'config.json')))
            if info['feature_dim'] != seq_type.feature_dim or info['target_dim'] != seq_type.target_dim:
                warnings.warn('The cached dataset has different feature or target dimension. Ignore')
                cache_path = 'invalid'
            if info.get('aux_dim', 0) != seq_type.aux_dim:
                warnings.warn('The cached dataset has different auxiliary dimension. Ignore')
                cache_path = 'invalid'
        else:
            info = {'feature_dim': seq_type.feature_dim, 'target_dim': seq_type.target_dim,
                    'aux_dim': seq_type.aux_dim}
            json.dump(info, open(osp.join(cache_path, 'config.json'), 'w'))

    features_all, targets_all, aux_all = [], [], []
    for i in range(len(data_list)):
        if cache_path is not None and osp.exists(osp.join(cache_path, data_list[i] + '.hdf5')):
            with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5')) as f:
                feat = np.copy(f['feature'])
                targ = np.copy(f['target'])
                aux = np.copy(f['aux'])
        else:
            # 新建缓存的逻辑
            seq = seq_type(osp.join(root_dir, data_list[i]), **kwargs)
            feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
            if cache_path is not None and osp.isdir(cache_path):
                with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5'), 'x') as f:
                    f['feature'] = feat
                    f['target'] = targ
                    f['aux'] = aux
        print(data_list[i])
        features_all.append(feat)
        targets_all.append(targ)
        aux_all.append(aux)
    return features_all, targets_all, aux_all


