#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 21:15
# @Author  : Joisen
# @File    : eatModel.py

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        # self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        res = x
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = x + res     #残差连接
        x = self.relu2(x)
        return x
class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
    def forward(self, x):
        res = x
        res = self.conv(res)    #对输入进行下采样
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = self.relu2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(418, 128, kernel_size=3, stride=1, padding=1)  # 尺寸不变
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Sequential(
            ResBlockDown(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.conv3 = nn.Sequential(
            ResBlockDown(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        self.conv4 = nn.Sequential(
            ResBlockDown(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        self.conv5 = nn.Sequential( # 将256改成了128 以及取消了conv6
            ResBlockDown(256, 128),
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        # self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128*34, 4)

    def forward(self, x):
        x = self.conv1(x) # batchsize*418*34*1 -> bs*64*34*1
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    model = Net()
    input = torch.randn(1, 418, 34, 1)
    output = model(input)
    print(model)
    print(output)
