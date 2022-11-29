#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/9 14:02
# @Author  : Joisen
# @File    : testModel.py

import torch
from torch.utils.data import Dataset, DataLoader
from works.discard.dataset import *
from tqdm import tqdm
from works.discard.hzp_DisModel import *

if __name__ == '__main__':
    # 设置启用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据存放路径
    file_dir = 'D:\\ML\\discard_data\\'
    # 设置batch_size
    batch_size = 128
    # 加载数据集
    dataset = DiscardsDataset(file_dir)
    train_size = int(0.96 * len(dataset))
    test_size = int(0.025 * len(dataset))
    val_size = int(len(dataset) - train_size - test_size)
    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=2)  # train需要shuffle
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=2)
    # 定义网络
    model = torch.load('model_me/_model_me.pth')

    model.to(device)

    # 开始验证
    model.eval()
    sum_data = 0  # 记录测试样本数
    acc = 0  # 记录准确率
    with torch.no_grad():
        print('验证数据长度：', len(val_dataset))
        for feature, label in tqdm(val_loader):
            # 将数据放到指定设备上运算
            inputs = feature.to(device)
            targets = label.to(device)
            # 预测
            outputs = model(inputs)
            # 计算测试样本数
            sum_data += outputs.shape[0]
            # 计算正确数
            acc = (outputs.argmax(1) == targets).sum() + acc
        acc = acc / sum_data
        print(f"准确率为:{acc}")
