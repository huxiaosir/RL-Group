#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 9:09
# @Author  : Joisen
# @File    : dataset.py

import os
import torch
import torch.utils.data as data
import numpy as np
import json
from feature_extract.feature_extract import *
from mah_tool.so_lib.lib_MJ import *
import random, shutil

class PongDataset(data.Dataset):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.file_name_list = os.listdir(self.file_dir)

    def __getitem__(self, idx):
        file_name = self.file_name_list[idx] # 文件名
        file_path = os.path.join(self.file_dir, file_name) # 文件idx的路径
        file = open(file_path, encoding='utf-8')
        # load info in json
        info = json.load(file)
        handcards0 = info['handCards0']
        fulu_ = info['fulu_']
        king_card = info['king_card']
        discards_seq = info['discards_seq']
        remain_card_num = info['remain_card_num']
        self_king_num = info['self_king_num']
        fei_king_nums = info['fei_king_nums']
        round_ = info['round_']
        dealer_flag = info['dealer_flag']
        label = info['isPong']

        features = calculate_king_sys_suphx(handcards0,fulu_,king_card,discards_seq,remain_card_num,self_king_num,
                                            fei_king_nums,round_,dealer_flag,search=False)




        return features, label

    def __len__(self):
        return len(self.file_name_list)


if __name__ == '__main__':
    file_dir = 'D:\\ML\\pong_data\\train\\'
    dataset = PongDataset(file_dir)
    # file_name_list = os.listdir(file_dir)
    # file_path = os.path.join(file_dir, file_name_list[1])
    # feature, label = dataset[1]
    train_size = int(0.8 * len(dataset))
    val_size = int(len(dataset) - train_size)
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_dataset, batch_size=100,shuffle=True,drop_last=True,num_workers=0) # train需要shuffle

    for features, labels in train_loader:
        print(features, labels)

        print()


    print()