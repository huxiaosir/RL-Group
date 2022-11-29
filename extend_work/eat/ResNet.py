#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/19 11:00
# @Author  : Joisen
# @File    : ResNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 18
# class BasicBlock(nn.Module):
#     def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
#         super(BasicBlock, self).__init__()
#         # 残差部分
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding),
#             nn.ReLU(inplace=True), # 原地替换 节省内存开销
#             nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding),
#         )
#
#         # shortcut 部分
#         # 由于存在维度不一致的情况 所以分情况
#         self.shortcut = nn.Sequential()
#         if stride[0] != 1 or in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0])
#
#     def forward(self, x):
#         out = self.layer(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
# # 采用bn的网络中，卷积层的输出并不加偏置
# class ResNet(nn.Module):
#     def __init__(self, num_classes=4) -> None:
#         super(ResNet, self).__init__()
#         self.in_channels = 256
#         # 第一层作为单独的 因为没有残差快
#         self.conv1 = nn.Conv2d(69,128,kernel_size=3,stride=1,padding=1)
#         self.conv1_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         # conv2_x
#         self.conv2 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])
#
#         # conv3_x
#         self.conv3 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])
#
#         # conv4_x
#         self.conv4 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])
#
#         # conv5_x
#         self.conv5 = self._make_layer(BasicBlock,256,[[1,1],[1,1]])
#
#         self.conv = nn.Conv2d(256,128,3,1,1)
#
#         self.fc = nn.Linear(128*36, num_classes)
#
#     #这个函数主要是用来，重复同一个残差块
#     def _make_layer(self, block, out_channels, strides):
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels
#         return nn.Sequential(*layers)
#
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#
#         out = self.conv(out)
#         # print(out.shape)
#         out = out.reshape(x.shape[0], -1)
#         # print(out.shape)
#         out = self.fc(out)
#         return out


# 34
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


class ResNet(nn.Module): # 67%
    '''
    实现主module：ResNet34
    ResNet34 包含多个layer，每个layer又包含多个residual block
    用子module来实现residual block，用_make_layer函数来实现layer
    '''
    def __init__(self, num_classes=4):
        super(ResNet, self).__init__()

        # 前几层图像转换
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
