import os
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
# pip install torchvision
from torchvision import transforms, models, datasets
# https://pytorch.org/docs/stable/torchvision/index.html
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
from utils import im_convert




# 数据读取与预处理操作
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),  # 从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
    ]),
    'valid': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 8

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

if __name__ == '__main__':
    print(image_datasets)
    print(dataloaders)
    print(dataset_sizes)

    # 读取标签对应的实际名字
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(cat_to_name)

    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2

    dataiter = iter(dataloaders['valid'])
    inputs, classes = dataiter.next()

    for idx in range(columns*rows):
        ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
        ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
        plt.imshow(im_convert(inputs[idx]))
    plt.show()












































































































