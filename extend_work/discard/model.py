#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/19 9:32
# @Author  : Joisen
# @File    : model.py


import torch
import torch.nn as nn
'''
    
'''
class Basic_Layer(nn.Module):

    def __init__(self):
        super(Basic_Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    def forward(self,x):
        out = self.layer(x) + x
        return out


class Model_Me(nn.Module):
    def __init__(self, Basic_Layer, out_channel, nums):
        super(Model_Me, self).__init__()
        self.conv1 = nn.Conv2d(69, 128, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu1_2 = nn.ReLU()
        self.conv2 = self._make_layer(Basic_Layer, nums)
        self.conv3 = nn.Conv2d(256,128,3,1,1)
        self.fc = nn.Linear(128*36, out_channel)

    def _make_layer(self, basic, nums):
        layers = []
        for i in range(nums):
            layers.append(basic())
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = Model_Me(Basic_Layer,34,20)
    input = torch.randn(1, 69, 4, 9)
    print(model)
    # print(model(input))
    # 训练数据长度： 28308
    # 验证数据长度： 13211