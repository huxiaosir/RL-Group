#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/20 15:32
# @Author  : Joisen
# @File    : model_1.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''
    实现子module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    '''
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, num_classes=34):
        super(ResNet, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(69, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(128, 256, 3)
        self.layer2 = self._make_layer(256, 256, 4)
        self.layer3 = self._make_layer(256, 256, 6)
        self.layer4 = self._make_layer(256, 256, 3)
        self.layer5 = nn.Conv2d(256, 128, 3, 1, 1)
        self.relu = nn.ReLU()
        # 分类用的全连接
        self.fc = nn.Linear(4608, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer,包含多个residual block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.relu(x)
        # print(x.shape)
        # x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        return self.fc(x)



if __name__ == '__main__':
    res34 = ResNet()
    input = torch.randn(1, 69, 4, 9)
    print(res34(input))
