import json
import os
import sys
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from model_temporal import LSTMSeqNetwork, BilinearLSTMSeqNetwork, TCNSeqNetwork
from utils import load_config, MSEAverageMeter
from data_ridi import SequenceToSequenceDataset,RIDIDataset,RIDIRawDataSequence
_input_channel, _output_channel = 6, 2
device = 'cpu'
# 定义训练中需要用到的参数
class GetArgs(dict):
    def __init__(self, *args, **kwargs):
        super(GetArgs, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = GetArgs(value)
        return value

info = {'type': 'lstm',
        'data_dir':'D:\DataSet\RIDI\\archive\data_publish_v2',
        'test_path':'D:\RoNIN\seen_subjects_test_set\\a000_7',
        'train_list':'D:\DataSet\RIDI\\archive\data_publish_v2\\train_list.txt',
        'cache_path':'D:\DataSet\RIDI\\archive\data_publish_v2\\cache',
        'feature_sigma':0.001,
        'target_sigma':0.0,
        'window_size':400,
        'step_size':100,
        'batch_size':128,
        'num_workers':1,
        'out_dir':'D:\DataSet\RIDI\\archive\data_publish_v2\\cache\\lstm_out',
        'device':'cpu',
        'dataset':'ridi',
        'layers':3,
        'layer_size':100,
        'val_list':None,
        'epochs':1000,
        'save_interval':20,
        'lr':0.003,
        'mode':'train',
        'continue_from':None,
        'model_path' : None,
        'fast_test':False,
        'show_plot':True,
        'seq_type': RIDIRawDataSequence
        }
args = GetArgs(info)

def get_dataset(args, **kwargs):
    with open(args.train_list) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    random_shift, shuffle, transforms, grv_only = 0, False, [], False
    print(data_list)
    dataset = SequenceToSequenceDataset(args.seq_type, args.data_dir, data_list, args.cache_path, args.step_size, args.window_size,
                                        random_shift=random_shift, transform=transforms, shuffle=shuffle,
                                        grv_only=grv_only, **kwargs)
    print("数据集获取成功")
    return dataset

class GlobalPosLoss(torch.nn.Module):
    def __init__(self, mode='full', history=None):
        """
        Calculate position loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalPosLoss, self).__init__()
        """
            regression_loss = torch.nn.MSELoss(reduction='none')  # none 不求平均 # 默认为mean #sum
            inputs = torch.tensor([1., 2.])
            target = torch.tensor([2., 5.])
            loss = regression_loss(inputs, target)
            print(loss)
            
            tensor([1., 9.])
        
        """
        #  输出的是各个位置元素相减之后的平方
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        assert mode in ['full', 'part']
        self.mode = mode
        if self.mode == 'part':
            assert history is not None
            self.history = history
        elif self.mode == 'full':
            self.history = 1
    # 这里定义的损失函数是用真实位置和估计位置进行计算均方根误差
    def forward(self, pred, targ):
        gt_pos = torch.cumsum(targ[:, 1:, ], 1)
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)
        # if self.mode == 'part':
        #     gt_pos = gt_pos[:, self.history:, :] - gt_pos[:, :-self.history, :]
        #     pred_pos = pred_pos[:, self.history:, :] - pred_pos[:, :-self.history, :]
        loss = self.mse_loss(pred_pos, gt_pos)
        return torch.mean(loss)

# 写出配置信息
def write_config(args, **kwargs):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "pytorch_global_position"
            if kwargs:
                values['kwargs'] = kwargs
            json.dump(values, f, sort_keys=True)


# 获取训练模型
def get_model(args, **kwargs):
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')

    if args.type == 'tcn':
        network = TCNSeqNetwork(_input_channel, _output_channel, args.kernel_size,
                                layer_channels=args.channels, **config)
        print("TCN Network. Receptive field: {} ".format(network.get_receptive_field()))
    elif args.type == 'lstm_bi':
        print("Bilinear LSTM Network")
        network = BilinearLSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                         lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)
    else:
        print("Simple LSTM Network")
        network = LSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                 lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)

    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    return network


def get_loss_function(history, args, **kwargs):

    config = {'mode': 'full'}
    criterion = GlobalPosLoss(**config)
    return criterion


def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset(args, **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True,pin_memory=True)
    end_t = time.time()

    print('Training set loaded. Time usage: {:.3f}s'.format(end_t - start_t))
    val_dataset, val_loader = None, None

    # 训练数据的设备
    global device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(device)
    # 输出文件的地址
    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        copyfile(args.train_list, osp.join(args.out_dir, "train_list"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.out_dir, "validation_list"))
        write_config(args, **kwargs)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    # 验证集
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_model(args, **kwargs).to(device)
    history = network.get_receptive_field() if args.type == 'tcn' else args.window_size // 2
    criterion = get_loss_function(history, args, **kwargs)
    # criterion = torch.nn.MSELoss()

    print(criterion)
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.75, verbose=True, eps=1e-12)
    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', False)

    log_file = None
    if args.out_dir:
        log_file = osp.join(args.out_dir, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.out_dir, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        with open(osp.join(str(Path(args.continue_from).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)

        if device.type == 'cpu':
            checkpoints = torch.load(args.continue_from, map_location=lambda storage, location: storage)
        else:
            checkpoints = torch.load(args.continue_from, map_location={model_data['device']: args.device})

        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    if kwargs.get('force_lr', False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)

    print("Starting from epoch {}".format(start_epoch))
    try:
        for epoch in range(start_epoch, args.epochs):
            log_line = ''
            network.train()
            # MSEAverageMeter的作用是计算均值
            train_vel = MSEAverageMeter(3, [2], _output_channel)
            train_loss = 0
            start_t = time.time()
            for bid, batch in enumerate(train_loader):
            # bid 指切片号 batch中包含数据
                # 每次获取的feat的规格是[batch_size * Windows_size * input_dim]
                feat, targ, _, _ = batch
                feat, targ = feat.to(device), targ.to(device)
                # 梯度清0
                optimizer.zero_grad()
                # 进行预测
                predicted = network(feat)
                #
                train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                # 计算损失值，通常使用交叉熵、均方误差等损失函数，这里的损失函数是通过
                loss = criterion(predicted, targ)
                train_loss += loss.cpu().detach().numpy()
                # 对损失值进行反向传播，计算参数的梯度
                loss.backward()
                # 使用优化器（optimizer）更新模型的参数
                optimizer.step()
                step += 1

            train_errs[epoch] = train_loss / train_mini_batches
            end_t = time.time()
            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss: {}, vel_loss {}/{:.6f}'.format(
                    epoch, end_t - start_t, train_errs[epoch], train_vel.get_channel_avg(), train_vel.get_total_avg()))
            log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch],
                                     *train_vel.get_channel_avg())

            saved_model = False

            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)
            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)

if __name__ == '__main__':
    train(args)



