import json
import os
import sys
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from model_temporal import LSTMSeqNetwork, BilinearLSTMSeqNetwork, TCNSeqNetwork
from IMUNet import IMUNet, BasicBlock1D, FCOutputModule
from model_resnet1d import ResNet1D
from utils import load_config, MSEAverageMeter
from data_ridi import SequenceToSequenceDataset, RIDIDataset, RIDIRawDataSequence, RIDIRawDataSequence2

_input_channel, _output_channel = 6, 2
device = 'cuda:0'


# 定义训练中需要用到的参数
class GetArgs(dict):
    def __init__(self, *args, **kwargs):
        super(GetArgs, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = GetArgs(value)
        return value


info = {'type': 'lstm_bi',
        'data_dir': 'D:\DataSet\RIDI\\archive\data_publish_v2',
        # 'data_dir': '/home/jiamingjie/zhanghaoyu/data',
        # 'train_list': '/home/jiamingjie/zhanghaoyu/data/train_handle.txt',
        'train_list': 'D:\DataSet\RIDI\\archive\list\\train.txt',
        # 'val_list': '/home/jiamingjie/zhanghaoyu/data/val_handle.txt',
        'val_list': 'D:\DataSet\RIDI\\archive\list\\val.txt',
        # 'test_list': '/home/jiamingjie/zhanghaoyu/data/test_list2.txt',
        'test_list': 'D:\DataSet\RIDI\\archive\list\\test.txt',

        # 'cache_path': '/home/jiamingjie/zhanghaoyu/data//cache',
        'cache_path': 'D:\DataSet\RIDI\\archive\\cache',

        # 'model_path': '/home/jiamingjie/zhanghaoyu/datacache/handle_out/checkpoints/checkpoint_best.pt',
        'model_path': 'D:\DataSet\RIDI\\archive/out/checkpoints/checkpoint_best.pt',
        'feature_sigma': 0.001,
        'target_sigma': 0.0,
        'window_size': 200,
        'step_size': 10,
        'batch_size': 128,
        'num_workers': 1,
        # 'out_dir': '/home/jiamingjie/zhanghaoyu/datacache/handle_out/',
        'out_dir': 'D:\DataSet\RIDI\\archive/out',
        'device': 'cuda:0',
        'dataset': 'ridi',
        'layers': 3,
        'layer_size': 200,
        'epochs': 300,
        'save_interval': 20,
        'lr': 0.0001,
        'mode': 'train',
        # 'continue_from': 'D:\DataSet\RIDI\\archive\data_publish_v2\cache\lstm_out\checkpoints\\icheckpoint_7979.pt',
        'continue_from': None,
        'fast_test': False,
        'show_plot': True,
        'seq_type': RIDIRawDataSequence,
        }
args = GetArgs(info)


def get_dataset(args, **kwargs):
    mode = kwargs.get('mode', 'train')
    random_shift, shuffle, transforms, grv_only = 0, False, [], False
    # 加载训练数据
    if mode == 'train':
        shuffle = True
        random_shift = args.step_size // 2
        with open(args.train_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            print(data_list)
    # 加载验证集数据
    elif mode == 'val':
        shuffle = True
        with open(args.val_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            print(data_list)
    # 加载测试数据
    else:
        shuffle = False
        with open(args.test_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            print(data_list)
    dataset = RIDIDataset(args.seq_type, args.data_dir, data_list, args.cache_path, args.step_size,
                                        args.window_size,
                                        random_shift=random_shift, transform=transforms, shuffle=shuffle,
                                        grv_only=grv_only, **kwargs)
    return dataset


class GlobalPosLoss(torch.nn.Module):
    def __init__(self):
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
        self.mse_loss = torch.nn.MSELoss()

        # assert mode in ['full', 'part']
        # self.mode = mode
        # if self.mode == 'part':
        #     assert history is not None
        #     self.history = history
        # elif self.mode == 'full':
        #     self.history = 1

    # 这里定义的损失函数是用真实位置和估计位置进行计算均方根误差
    def forward(self, pred, targ):
        # 使用位置计算 而不是使用每秒的位移
        # 做数据集的时候为什么不用pos直接输入？
        gt_pos = torch.cumsum(targ[:, 1:, ], 1)
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)
        loss = self.mse_loss(pred_pos, gt_pos)
        return loss


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
        # print("Bilinear LSTM Network")
        # network = BilinearLSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
        #                                  lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)
        _fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}
        # network = IMUNet(_input_channel, 2, BasicBlock1D, [2, 2, 2, 2],
        #                    base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
        network = IMUNet(_input_channel, _output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)

    else:
        print("Simple LSTM Network")
        network = LSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                 lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)

    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    print(network)
    return network


def get_loss_function():
    criterion = GlobalPosLoss()
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
    train_dataset = get_dataset(args, mode='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=True)
    end_t = time.time()

    print('训练集获取成功. 用时: {:.3f}s'.format(end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset(args, mode='val', **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('验证集加载成功')
    # 训练数据的设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('训练设备' + device.type)
    summary_writer = None
    # 输出文件的地址
    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        copyfile(args.train_list, osp.join(args.out_dir, "train_list.txt"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.out_dir, "validation_list.txt"))
        write_config(args, **kwargs)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    # 验证集
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_model(args, **kwargs).to(device)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    # history = network.get_receptive_field() if args.type == 'tcn' else args.window_size // 2
    # criterion = get_loss_function()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True, eps=1e-12)

    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', False)

    # log_file = None
    # if args.out_dir:
    #     log_file = osp.join(args.out_dir, 'logs', 'log.txt')
    #     if osp.exists(log_file):
    #         if args.continue_from is None:
    #             os.remove(log_file)
    #         else:
    #             copyfile(log_file, osp.join(args.out_dir, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        with open(osp.join(str(Path(args.continue_from).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)

        if device.type == 'cpu':
            checkpoints = torch.load(args.continue_from, map_location=lambda storage, location: storage)
        else:
            checkpoints = torch.load(args.continue_from, 'cuda:0')

        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    # if kwargs.get('force_lr', False):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)

    print("Starting from epoch {}".format(start_epoch))

    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.  获取初始的损失函数
    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)
    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device)
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        for epoch in range(start_epoch, args.epochs):
            log_line = ''
            network.train()
            # MSEAverageMeter的作用是计算均值
            # train_vel = MSEAverageMeter(3, [2], _output_channel)
            train_outs, train_targets = [], []
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
                train_outs.append(predicted.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                #
                # train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                # 计算损失值，通常使用交叉熵、均方误差等损失函数，这里是均方根
                loss = criterion(predicted, targ)
                loss = torch.mean(loss)
                # 对损失值进行反向传播，计算参数的梯度
                loss.backward()
                # 使用优化器（optimizer）更新模型的参数
                optimizer.step()
                step += 1
            # 计算每个batch的loss
            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)
            end_t = time.time()
            print('-------------------------')
            # print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
            #     epoch, end_t - start_t, train_losses, np.average(train_losses)))
            print('Epoch {}, time usage: {:.3f}s, average loss: {:.6f}, lr : {}'.format(
                epoch, end_t - start_t, np.average(train_losses), optimizer.param_groups[0]['lr']))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            # if not quiet_mode:
            #     print('-' * 25)
            #     # print('Epoch {}, time usage: {:.3f}s, loss: {}, vel_loss {}/{:.6f}'.format(
            #     #     epoch, end_t - start_t, train_errs[epoch], train_vel.get_channel_avg(), train_vel.get_total_avg()))
            #     print('Epoch {}, time usage: {:.3f}s, loss: {}, learningRate: {}'.format(
            #         epoch, end_t - start_t, train_errs[epoch], optimizer.state_dict()['param_groups'][0]['lr']))
            # log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch], )

            saved_model = False
            # 验证集
            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(network, val_loader, device)
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                # print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                print('Validation loss: {:.6f}'.format(avg_loss))
                scheduler.step(avg_loss)
                # val_vel = MSEAverageMeter(3, [2], _output_channel)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                    val_losses_all.append(avg_loss)
                    if avg_loss < best_val_loss:
                        best_val_loss = avg_loss
                        if args.out_dir and osp.isdir(args.out_dir):
                            model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_best.pt')
                            torch.save({'model_state_dict': network.state_dict(),
                                        'epoch': epoch,
                                        'optimizer_state_dict': optimizer.state_dict()}, model_path)
                            print('Model saved to ', model_path)

            total_epoch = epoch
            #     for bid, batch in enumerate(val_loader):
            #         feat, targ, _, _ = batch
            #         feat, targ = feat.to(device), targ.to(device)
            #         optimizer.zero_grad()
            #         pred = network(feat)
            #         # val_vel.add(pred.cpu().detach().numpy(), targ.cpu().detach().numpy())
            #         val_loss += criterion(pred, targ).cpu().detach().numpy()
            #     val_loss = val_loss / val_mini_batches
            #     # log_line = format_string(log_line, val_loss, *val_vel.get_channel_avg())
            #     log_line = format_string(log_line, val_loss)
            #     if not quiet_mode:
            #         # print('Validation loss: {} vel_loss: {}/{:.6f}'.format(val_loss, val_vel.get_channel_avg(),
            #         #                                                        val_vel.get_total_avg()))
            #         print('Validation loss: {} '.format(val_loss))
            #     if val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         saved_model = True
            #         if args.out_dir:
            #             model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
            #             torch.save({'model_state_dict': network.state_dict(),
            #                         'epoch': epoch,
            #                         'loss': train_errs[epoch],
            #                         'optimizer_state_dict': optimizer.state_dict()}, model_path)
            #             print('Best Validation Model saved to ', model_path)
            #     if use_scheduler:
            #         scheduler.step(val_loss)
            #
            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)
            #
            # if log_file:
            #     log_line += '\n'
            #     with open(log_file, 'a') as f:
            #         f.write(log_line)
            # if np.isnan(train_loss):
            #     print("Invalid value. Stopping training.")
            #     break
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)



def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    print(targets_all.shape)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')
    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


# 获取根据速度获取全局的位置
def recon_traj_with_preds_global(dataset, preds, seq_id=0, type='preds', **kwargs):

    if type == 'gt':
        pos = dataset.gt_pos[seq_id]
    else:
        start_pos = dataset.gt_pos[seq_id][0]
        preds[0] = start_pos
        pos = np.cumsum(preds,axis=0)
        # ts = dataset.ts[seq_id]
        # # Compute the global velocity from local velocity.
        # dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
        # pos = preds * dts
        # pos[0, :] = dataset.gt_pos[seq_id][0, :]
        # pos = np.cumsum(pos, axis=0)
    # veloc = preds
    # ori = dataset.orientations[seq_id]

    return pos

def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


# 测试
def test(args, **kwargs):
    global device, _output_channel
    import matplotlib.pyplot as plt
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path, map_location=args.device)
    network = get_model(args, **kwargs)
    print(network)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))
    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []
    seq_dataset = get_dataset(args, mode='test', **kwargs)
    seq_loader = DataLoader(seq_dataset, 1024, num_workers=args.num_workers, shuffle=False)
    ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int)
    targets, preds = run_test(network, seq_loader, device, True)
    print(targets.shape)
    print(preds.shape)
    losses = np.mean((targets - preds) ** 2, axis=0)
    preds_seq.append(preds)
    targets_seq.append(targets)
    losses_seq.append(losses)
    pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :2]
    print(pos_pred.shape)
    pos_gt = seq_dataset.gt_pos[0][:, :2]

    traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
    # ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
    # ate_all.append(ate)
    # rte_all.append(rte)
    # pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

    # print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

    train_dataset = get_dataset(args, mode='test', **kwargs)
    print(train_dataset.index_map)
    feat, tar = train_dataset.get_test_seq(0)
    # print(feat.shape)
    # print(tar.shape)
    # feat = torch.Tensor(feat).to(device)
    # preds = np.squeeze(network(feat).cpu().detach().numpy())
    # new_data = []
    # for i in range(0, preds.shape[0], 200):
    #     new_data.append(preds[i:i + 200].mean(axis=0))
    # new_data = np.array(new_data)
    # print(new_data.shape)


    # ind = np.arange(tar.shape[0])
    # pos_pred = recon_traj_with_preds_global(train_dataset, new_data,  type='pred', seq_id=0)
    pos_gt = recon_traj_with_preds_global(train_dataset, tar, seq_id=0,type='gt')
    print(pos_pred)
    #
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1],label = 'gt')
    ax1.plot(pos_pred[:, 0], pos_pred[:, 1],label = 'pred')
    ax1.legend()
    plt.show()



if __name__ == '__main__':
    # train(args,use_scheduler = True)
    test(args)
