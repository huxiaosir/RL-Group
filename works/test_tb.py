#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 19:10
# @Author  : Joisen
# @File    : test_tb.py
#
# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# writer = SummaryWriter('logs')
#
# for i in range(100):
#     writer.add_scalar('y=x', i, i)
# writer.close()
#
# def hadn(a,b,n):
#     lid = n
#     bottle = n
#     wine = n  # 初始的瓶盖、瓶子、和酒的数量均为n
#     while lid >= a or bottle >= b:
#         if lid >= a:  # 4个瓶盖换1瓶酒
#             lid -= a
#         else:  # 2个瓶子换1瓶酒
#             bottle -= b
#         lid += 1
#         bottle += 1
#         wine += 1  # 得到一瓶酒后瓶子和瓶盖和酒的数量均加一
#     print(wine)
#
# if __name__ == '__main__':
#     while(True):
#         a,b,n = map(int,input().split())
#         if a==0 and b==0 and n == 0:
#             break
#         hadn(a,b,n)
count = 0
surplus_bottleTop = 0
surplus_bottle = 0
def drinkAndSurplus(bottleTop, bottle):
    global count,surplus_bottleTop,surplus_bottle
    beers = 0
    beers += bottleTop//4
    beers += bottle//2
    count += beers
    bottleTop = bottleTop%4 + beers
    bottle = bottle%2 + beers
    print('本次喝了%d瓶，剩余瓶盖%d个，剩余瓶子%d个'%(beers, bottleTop, bottle))
    if(bottleTop//4 > 0 or bottle//2 > 0):
        drinkAndSurplus(bottleTop, bottle)
    else:
        surplus_bottleTop = bottleTop
        surplus_bottle = bottle
if __name__ == '__main__':
    money = 20
    count, surplus_bottleTop, surplus_bottle = money//2, money//2, money//2
    print('总共%d元钱,本次喝酒%d瓶，剩余瓶盖%d个，剩余瓶子%d个'%(money,count,count,count))
    drinkAndSurplus(surplus_bottleTop, surplus_bottle)
    print('总共喝了%d瓶，剩余瓶盖%d个，剩余瓶子%d个'%(count,surplus_bottleTop,surplus_bottle))




