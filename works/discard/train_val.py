#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/2 16:08
# @Author  : Joisen
# @File    : train_val.py

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from works.discard.dataset import *
from works.discard.model import *
from tqdm import tqdm
from works.discard.hzp_DisModel import *
from works.discard.model_2 import *
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    writer = SummaryWriter('log_me')
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
    # model = ResNet50(Bottleneck, block_num=[3, 4, 6, 3], num_classes=34)
    model = Model_Me(Basic_Layer,34, 20)
    # model = Net()

    # 定义损失函数
    loss = nn.CrossEntropyLoss()

    # 设置学习率
    LR = 0.0005

    # 定义优化器
    optim = Adam(model.parameters(), lr=LR)

    # 训练轮次
    train_round = 20

    # 将loss和神经网络放到指定设备上执行
    loss.to(device)
    model.to(device)

    for i in range(train_round):
        # 开始训练
        loss_sum = 0
        model.train()
        print(f"------训练第{i + 1}开始------")
        print('数据长度：', len(test_dataset))
        for feature, label in tqdm(test_loader):
            # 初始化梯度
            optim.zero_grad()
            # 将数据放到指定设备上运算
            feature = torch.FloatTensor(feature.numpy())
            inputs = feature.to(device)
            targets = label.to(device)
            # 进行训练
            outputs = model(inputs)
            # 计算损失，并计算梯度
            train_loss = loss(outputs, targets)
            train_loss.backward()
            # 更新参数
            optim.step()

            # 计算总损失
            loss_sum += train_loss.item()
        writer.add_scalar('sum loss', loss_sum, i)
        print(f"训练损失为:{loss_sum}")

        # 开始验证
        print(f"----开始第{i + 1}次验证----")
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
            writer.add_scalar('Accuracy', acc, i)
        torch.save(model, f"./model_me/model_{i}.pth")
    writer.close()
    print("")
# tensorboard --logdir=D:\PyCharm\PyCharmWorkPlace\RL-Group\works\discard\logs
# tensorboard --logdir=D:\PyCharm\PyCharmWorkPlace\RL-Group\works\discard\log_me
