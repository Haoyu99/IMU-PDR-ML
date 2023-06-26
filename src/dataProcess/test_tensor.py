import time

from torch.utils.tensorboard import SummaryWriter
import torch
import IMUNet
from tensorboardX import SummaryWriter

# SummaryWriter encapsulates everything
writer = SummaryWriter('log/exp-1')  # save in 'runs/exp-1'
writer2 = SummaryWriter()             #  save in 'runs/Aug20-17-20-33'
writer3 = SummaryWriter(comment='3x learning rate')  # save in 'runs/Aug20-17-20-33-3xlearning rate'
writer = SummaryWriter("log/scalar")
x = range(100)
for i in x:
    time.sleep(0.1)
    writer.add_scalar('y=2x', i * 2, i, walltime=time.time())
writer.close()

