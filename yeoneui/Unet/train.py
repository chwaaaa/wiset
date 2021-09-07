import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder

## Hyperparameters
LR = 1e-3
BATCH_SIZE = 4
EPOCH = 30

data_dir = './data/Splitted'
check_dir = 'checkpoint'
log_dir ='log'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##