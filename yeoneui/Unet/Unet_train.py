# resize = 256
# UNet Model
# Block1: ( Conv3x3 - (BN) - ReLU )*2 - Maxpool2x2
# Block2: UpConv2x2 - ( Conv3x3 - (BN) - ReLU )*2
# Block1*4 -> Block2*4 -> ( Conv3x3 - (BN) - ReLU )*2
# -> ( Conv1x1 - (BN) - ReLU )
# act_fc = nn.ReLU(inplace = true) or nn.LeakyReLU(0.2, inplace=True)
# concatenation 과정에서 padding 문제가 발생하여 conv3에서 padding=1을 하여 해결
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

## Hyperparameters
LR = 1e-3
BATCH_SIZE = 4
EPOCH = 30

data_dir = 'Splitted'
check_dir = 'checkpoint'
log_dir ='log'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Conv1x1BR(in_channels, out_channels, act_fc):
    out = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        act_fc
    )
    return out

def DoubleConv3x3BR(in_channels, out_channels, act_fc):
    out = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3,
                  stride=1,bias=True, padding =1),
        nn.BatchNorm2d(out_channels),
        act_fc,
        nn.Conv2d(out_channels, out_channels,kernel_size=3,
                  stride=1, bias = True, padding =1),
        nn.BatchNorm2d(out_channels),
        act_fc
    )
    return out

def Conv3x3BR(in_channels, out_channels, act_fc):
    out = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3,
                  stride=1,bias=True, padding =1),
        nn.BatchNorm2d(out_channels),
        act_fc)

    return out

def UpConv2x2(in_channels, out_channels, mode, act_fc):
    if mode =='up-conv':
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            act_fc
        )

    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            Conv1x1BR(in_channels, out_channels, act_fc),
            nn.BatchNorm2d(out_channels),
            act_fc
        )




class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, act_fc):
        super(UNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_fc = act_fc

        # Contracting Path
        self.conv1 = DoubleConv3x3BR(self.in_channels, 64, act_fc)
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride =2)
        self.conv2 = DoubleConv3x3BR(64, 128, act_fc)
        self.conv3 = DoubleConv3x3BR(128,256, act_fc)
        self.conv4 = DoubleConv3x3BR(256,512, act_fc)

        self.conv5 = Conv3x3BR(512,1024,act_fc)

        # Expanding Path
        self.conv6 = Conv3x3BR(1024, 512, act_fc)

        self.up_1 = UpConv2x2(512, 512, 'up-conv', act_fc)
        self.conv7_1 = Conv3x3BR(1024, 512, act_fc)
        self.conv7_2 = Conv3x3BR(512, 256, act_fc)

        self.up_2 = UpConv2x2(256, 256, 'up-conv', act_fc)
        self.conv8_1 = Conv3x3BR(512, 256, act_fc)
        self.conv8_2 = Conv3x3BR(256, 128, act_fc)

        self.up_3 = UpConv2x2(128, 128, 'up-conv', act_fc)
        self.conv9_1 = Conv3x3BR(256, 128, act_fc)
        self.conv9_2 = Conv3x3BR(128, 64, act_fc)

        self.up_4 = UpConv2x2(64, 64, 'up-conv', act_fc)
        self.conv10_1 = Conv3x3BR(128, 64, act_fc)
        self.conv10_2 = Conv3x3BR(64, 64, act_fc)

        self.final = Conv1x1BR(64,out_channels, act_fc)

    def forward(self, x):
        #Contracting Path
        c1 = self.conv1(x)
        p1 = self.maxpool2x2(c1)

        c2 = self.conv2(p1)
        p2 = self.maxpool2x2(c2)

        c3 = self.conv3(p2)
        p3 = self.maxpool2x2(c3)

        c4 = self.conv4(p3)
        p4 = self.maxpool2x2(c4)

        c5 = self.conv5(p4)

        #Expanding Path
        c6 = self.conv6(c5)
        up1 = self.up_1(c6)
        merge1 = torch.cat([up1, c4], dim = 1)
        c7_1 = self.conv7_1(merge1)
        c7_2 = self.conv7_2(c7_1)

        up2 = self.up_2(c7_2)
        merge2 = torch.cat([up2, c3], dim = 1)
        c8_1 = self.conv8_1(merge2)
        c8_2 = self.conv8_2(c8_1)

        up3 = self.up_3(c8_2)
        merge3 = torch.cat([up3, c2], dim = 1)
        c9_1 = self.conv9_1(merge3)
        c9_2 = self.conv9_2(c9_1)

        up4 = self.up_4(c9_2)
        merge4 = torch.cat([up4, c1], dim = 1)
        c10_1 = self.conv10_1(merge4)
        c10_2 = self.conv10_2(c10_1)

        out = self.final(c10_2)

        return out




## Dataloader
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform =None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_input = [f for f in lst_data if f.startswith('tr_im')]
        lst_label = [f for f in lst_data if f.startswith('tr_mask')]

        lst_input.sort()
        lst_label.sort()

        self.lst_input = lst_input
        self.lst_label = lst_label

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self,index):
        input = np.load(os.path.join(self.data_dir,self.lst_input[index]))
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))

        input = input/255.0
        label = label/255.0

        if label.ndim ==2:
            label = label[:,:,np.newaxis]

        if input.ndim ==2:
            input = input[:,:,np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## Transform
MEAN = 0.5
STD = 0.5

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2,0,1)).astype(np.float32)
        input = input.transpose((2,0,1)).astype(np.float32)

        data = {'label': torch.from_numpy(label),
                'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean = MEAN, std = STD):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data


Unet_trans = {
    'train': transforms.Compose([
        RandomFlip(),
        ToTensor(),
        Normalization(MEAN, STD)]),
    'val': transforms.Compose([
        ToTensor(),
        Normalization(MEAN, STD)])
}

dataset_train = Dataset(data_dir= os.path.join(data_dir, 'train'), transform=Unet_trans['train'])
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle= True, num_workers= 0)

dataset_val = Dataset(data_dir= os.path.join(data_dir, 'train'), transform=Unet_trans['val'])
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle= False, num_workers= 0)





## Create Network
in_channel = 1
out_channel = 1 # segmentation mask (white: foreground, black: background)
act_fc = nn.ReLU(inplace= True)

unet = UNet(in_channel, out_channel, act_fc)
unet = unet.to(DEVICE)


from torchsummary import summary as summary_
summary_(unet,(1,512,512),batch_size=BATCH_SIZE)


## Define Loss, Optimizer
criterion = nn.BCEWithLogitsLoss().to(DEVICE)
optim = optim.Adam(unet.parameters(), lr= LR)

# 7 epoch마다 0.1씩 곱해서 lr을 감소시킨다.
#exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

## Other necessary variables
num_data_train =len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train/BATCH_SIZE)
num_batch_val = np.ceil(num_data_val/BATCH_SIZE)

## Other necessary functions
fn_tonumpy = lambda  x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda  x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## SummaryWriter settings
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir,'val'))


# Save network
def save(check_dir, net, optim, epoch):
    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                './%s/model)epoch%d.pth' %(check_dir, epoch))

# Load network
def load(chek_dir, net, optim):
    if not os.path.exists(check_dir):
        epoch = 0
        return  net, optim, epoch
    check_lst = os.listdir(check_dir)
    check_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' %(check_dir,check_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(check_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


## Train network
st_epoch = 0
unet, optim, st_epoch = load(chek_dir=check_dir, net= unet, optim = optim)

for epoch in range(st_epoch+1, EPOCH+1):
    unet.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        # forward pass
        label = data['label'].to(DEVICE)
        input = data['input'].to(DEVICE)

        output = unet(input)

        #backward pass
        optim.zero_grad()

        loss = criterion(output, label)
        loss.backward()

        optim.step()

        # compute loss
        loss_arr += [loss.item()]

        print("TRAIN: EPOCH %04d / %04d | BATCH %04d /%04d | LOSS %.4f" %
              (epoch, EPOCH, batch, num_batch_train, np.mean(loss_arr)))

        # Save in tensorboard
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean = MEAN, std= STD))
        output = fn_tonumpy(fn_class(output))

        writer_train.add_image('label', label, num_data_train*(epoch-1)+batch, dataformats='NHWC')
        writer_train.add_image('input', input, num_data_train*(epoch-1)+batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_data_train*(epoch-1)+batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    # Validation mode
    with torch.no_grad():
        unet.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val,1):
            # forward pass (no backward pass)
            label = data['label'].to(DEVICE)
            input = data['input'].to(DEVICE)

            output = unet(input)

            # Compute loss
            loss = criterion(output, label)

            loss_arr += [loss.item()]
            print("VAL: EPOCH %04d / %04d | BATCH %04d /%04d | LOSS %.4f" %
                  (epoch, EPOCH, batch, num_batch_val, np.mean(loss_arr)))

            #Save in Tensorboard
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean = MEAN, std= STD))
            output = fn_tonumpy(fn_class(output))

            writer_val.add_image('label', label, num_data_val*(epoch-1)+batch, dataformats='NHWC')
            writer_val.add_image('input', input, num_data_val*(epoch-1)+batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_data_val*(epoch-1)+batch, dataformats='NHWC')

    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

    if epoch % 5 ==0: # save every 5 epochs
        save(check_dir=check_dir, net=unet, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()

