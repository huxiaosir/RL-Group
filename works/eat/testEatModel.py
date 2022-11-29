#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/10 9:03
# @Author  : Joisen
# @File    : testEatModel.py


from torch.utils.data import Dataset, DataLoader
from works.eat.dataset_eat import *
from tqdm import tqdm
from works.eat.eatModel_2 import *

if __name__ == '__main__':
    # 设置启用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据存放路径
    test_dir = '../../datasets/extract_data_chi/test/'
    # 设置batch_size
    batch_size = 128
    # 加载数据集
    dataset = EatDataset(test_dir)

    test_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=2)
    # 定义网络
    model = torch.load('model_me/model_one_8.pth')

    model.to(device)

    # 开始验证
    model.eval()
    sum_data = 0  # 记录测试样本数
    acc = 0  # 记录准确率
    with torch.no_grad():
        print('验证数据长度：', len(dataset))
        for feature, label in tqdm(test_loader):
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