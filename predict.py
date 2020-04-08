import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

import numpy as np
import time
import os
import sys

from models.resnet import *
from models.mvcnn import *
import util
from custom_dataset import MultiViewDataSet

import nibabel as nib


transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
])

def transferNii(file_path):
    img = nib.load(file_path)
    img_fdata = img.get_fdata()
    (x, y, z) = img.shape
    cou_num = 0
    data = np.zeros(shape=(20, y, z))
    for i in range(30, x, int((x - 30) / 20)):
        if cou_num < 20 :
            data[cou_num] = img_fdata[i]
            cou_num = cou_num + 1
    return data

# Helper functions
def load_checkpoint(resume = "checkpoint/resnet101_checkpoint.pth.tar"):
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(resume), 'Error: no checkpoint file found!'

    checkpoint = torch.load(resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


if __name__ == "__main__":
    file_path = sys.argv[1]
    raw_data = transferNii(file_path)
    (x, y, z) = raw_data.shape
    views = []
    for i in range(20):
        temp = raw_data[i]
        im=Image.fromarray(temp)
        print(im.shape)
        im = im.convert('L')
        print(im.shape)
        im = transform(im)
        views.append(im)
    data = views.from_numpy(data)
    data = data.unsqueeze(0)
    net = load_checkpoint()
    output = net(data)
    print(output)

