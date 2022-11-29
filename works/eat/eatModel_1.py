#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 17:23
# @Author  : Joisen
# @File    : eatModel_1.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet(nn.Module):
    def __init__(self, num_classes=4) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 256
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(418,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])

        self.conv = nn.Conv2d(256,128,3,1,1)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4352, num_classes)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # out = self.avgpool(out)
        out = self.conv(out)
        # print(out.shape)
        out = out.reshape(x.shape[0], -1)
        # print(out.shape)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    res18 = ResNet()
    input = torch.randn(1, 418, 34, 1)
    print(res18(input))

    # 准确率最高63.86%