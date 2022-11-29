#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/2 17:17
# @Author  : Joisen
# @File    : test_.py
import os
import random, shutil

if __name__ == '__main__':
    file_dir = 'D:\\ML\\extend_work\\pong\\np\\'
    target_dir = 'D:\\ML\\extend_work\\pong\\val\\'
    # target_dir = 'D:\\ML\\discard_data'
    file_list = os.listdir(file_dir)
    files = random.sample(file_list, 3000)
    for file in files:
        shutil.move(os.path.join(file_dir, file), os.path.join(target_dir,file))



