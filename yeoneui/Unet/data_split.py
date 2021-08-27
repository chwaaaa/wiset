
## library
import os
import shutil
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


##
data_dir = './data/COVID-19-CT100'
base_dir = './Splitted'

if not os.path.exists(base_dir):
    os.mkdir(base_dir)


train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# data 분할 train:val:test = 0.6: 0.2: 0.2
train_size = 60
val_size = 20
test_size = 20

classes = ['image', 'mask']

for c in classes:
    path = os.path.join(data_dir, c)
    files = os.listdir(path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    train_file = files[:train_size]
    for fname in train_file:
        img = Image.open(os.path.join(path, fname))
        dst = os.path.join(train_dir, fname)
        img_ = np.asarray(img)
        np.save(dst, img_)


    val_file = files[train_size:(val_size + train_size)]
    for fname in val_file:
        img = Image.open(os.path.join(path, fname))
        dst = os.path.join(val_dir, fname)
        img_ = np.asarray(img)
        np.save(dst, img_)

    test_file = files[(train_size + val_size):(val_size + train_size + test_size)]
    for fname in test_file:
        img = Image.open(os.path.join(path, fname))
        dst = os.path.join(test_dir, fname)
        img_ = np.asarray(img)
        np.save(dst, img_)


