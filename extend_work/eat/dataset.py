#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/18 15:49
# @Author  : Joisen
# @File    : dataset.py

import os
import torch
import torch.utils.data as data
import numpy as np
import json
from feature_extract.feature_extract_me import *
from mah_tool.so_lib.lib_MJ import *
import random, shutil

class EatDataset(data.Dataset):
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
        self_king_num = info['self_king_num']
        fei_king_nums = info['fei_king_nums']
        discards = info['discards']
        label = info['chi_position']

        features = card_preprocess(handcards0,king_card,discards_seq,discards,self_king_num,fei_king_nums,fulu_)




        return features, label

    def __len__(self):
        return len(self.file_name_list)


if __name__ == '__main__':
    file_dir = 'D:\\ML\\extend_work\\eat\\train\\'
    dataset = EatDataset(file_dir)
    # file_name_list = os.listdir(file_dir)
    # file_path = os.path.join(file_dir, file_name_list[1])
    # feature, label = dataset[1]
    train_size = int(0.8 * len(dataset))
    val_size = int(len(dataset) - train_size)
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    train_loader = data.DataLoader(train_dataset, batch_size=100,shuffle=True,drop_last=True,num_workers=0) # train需要shuffle

    for features, labels in train_loader:
        print(features, labels)
        exit()
        print()


    print()
