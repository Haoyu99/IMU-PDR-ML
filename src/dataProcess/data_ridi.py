from os import path as osp
from data_utils import DataSequence,load_cached_sequences
import numpy as np
import pandas
import random
from torch.utils.data import Dataset


class RIDIRawDataSequence(DataSequence):
    """
    DataSet: RIDI数据集
    Features : 三轴的加速度，三轴的陀螺仪
    target: 时间窗位移
    """
    feature_dim = 6
    target_dim = 2
    # aux 数据是 时间 四元数 以及真实位置
    aux_dim = 7


    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        # w是一个窗口值
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        """
        从指定的path中加载csv文件
        :param path:
        :return:
        """
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        print(path)

        if osp.exists(osp.join(path, 'processed/data.csv')):
            imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        else:
            print('fail to load, data.csv is not exist')

        ts = imu_all[['time']].values / 1e09  # 时间值变为以秒为单位
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        pos = imu_all[['pos_x', 'pos_y']].values
        quat = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

        # Use game rotation vector as device orientation.
        # init_tango_ori = quaternion.quaternion(*imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[0])
        # game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
        #
        # init_rotor = init_tango_ori * game_rv[0].conj()
        # ori = init_rotor * game_rv
        #
        # nz = np.zeros(ts.shape)
        # gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        # acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))
        #
        # gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
        # acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro, acce], axis=1)
        # 计算每个时间窗的的位移
        self.targets = pos[self.w:, :] - pos[:-self.w, :]
        self.gt_pos = pos
        self.orientations = quat
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)


class RIDIRawDataSequence2(DataSequence):
    """
    DataSet: RIDI数据集
    Features : 三轴的加速度，三轴的陀螺仪
    target: 真实的位置position(x,y)
    """
    feature_dim = 6
    target_dim = 2
    # aux 数据是 时间 四元数 以及真实位置
    aux_dim = 7


    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        # w是一个窗口值
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        """
        从指定的path中加载csv文件
        :param path:
        :return:
        """
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        print(path)

        if osp.exists(osp.join(path, 'processed/data.csv')):
            imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        else:
            print('fail to load, data.csv is not exist')

        ts = imu_all[['time']].values / 1e09  # 时间值变为以秒为单位
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        pos = imu_all[['pos_x', 'pos_y']].values
        quat = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values


        self.ts = ts
        self.features = np.concatenate([gyro, acce], axis=1)
        # 计算每个时间窗的的位移
        self.targets = pos
        self.gt_pos = pos
        self.orientations = quat
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)


class RIDIGlobalDataSequence(DataSequence):
    """
    DataSet: 通过转换坐标系的RIDI数据集
    Features : 三轴的加速度，三轴的陀螺仪
    target: 时间窗位移
    """
    feature_dim = 6
    target_dim = 2
    # aux 数据是 时间 四元数 以及真实位置
    aux_dim = 7


    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        # w是一个窗口值
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        """
        从指定的path中加载csv文件
        :param path:
        :return:
        """
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        print(path)

        if osp.exists(osp.join(path, 'processed/data.csv')):
            imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        else:
            print('fail to load, data.csv is not exist')

        ts = imu_all[['time']].values / 1e09  # 时间值变为以秒为单位
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        pos = imu_all[['pos_x', 'pos_y', 'pos_z']].values
        quat = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

        # Use game rotation vector as device orientation.
        # init_tango_ori = quaternion.quaternion(*imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[0])
        # game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
        #
        # init_rotor = init_tango_ori * game_rv[0].conj()
        # ori = init_rotor * game_rv
        #
        # nz = np.zeros(ts.shape)
        # gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        # acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))
        #
        # gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
        # acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro, acce], axis=1)
        # 计算每个时间窗的的位移
        self.targets = (pos[self.w:, :2] - pos[:-self.w, :2]) / (ts[self.w:] - ts[:-self.w])
        self.gt_pos = pos
        self.orientations = quat
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

class RIDIDataset(Dataset):
    """
    用于训练的RIDI数据集
    seq_type 是继承于DataSequence的具体实现类
    """
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(RIDIDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        # 从缓存中加载数据
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        # 把来自不同文件的数据加载到同一个list当中
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -2:])
            # 将target中的数据 切片[i,j] i代表文件id， j代表切片的索引值 每step_size切一次
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]
        print('lode success')
        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        # 默认 random_shift = 0
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))
        # 就是说每200个feature 生成一个target
        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        # if self.transform is not None:
        #     feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)
    def get_test_seq(self, i):
        return self.features[i].astype(np.float32), self.targets[i].astype(np.float32)


class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.window_size, **kwargs)

        # Optionally smooth the sequence 平滑序列
        # feat_sigma = kwargs.get('feature_sigma,', -1)
        # targ_sigma = kwargs.get('target_sigma,', -1)
        # # 如果sigma大于0，则对其进行高斯滤波
        # if feat_sigma > 0:
        #     self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        # if targ_sigma > 0:
        #     self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            # aux
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, -2:])
            # 计算一个合加速度
            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            # 移除异常点
            bad_data = velocity > max_norm
            # bad_data 是一个为维度和velocity一样的 布尔矩阵 大于max_norm的地方是ture 其余地方为false
            # j 从（400，数据总数，步长100）
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                # any 又有一个为真 就为真  如果没有坏点 则放入索引map中
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))
        # feat 的维度 ： 6 * 400
        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        # target 的维度 ： 6 * 400
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])


        # if self.transform is not None:
        #     feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)