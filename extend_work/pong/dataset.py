#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/22 10:26
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
        discards = info['discards']
        round_ = info['round_']
        dealer_flag = info['dealer_flag']
        label = info['isPong']

        features = card_preprocess(handcards0,king_card,discards_seq,discards,self_king_num,fei_king_nums,fulu_)




        return features, label

    def __len__(self):
        return len(self.file_name_list)
