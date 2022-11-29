#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/11/2 10:16
# @Author  : Joisen
# @File    : test1.py
# ! /usr/bin/python
def rm(a, b,ts):
   if (a, b) in ts:
       ts.remove((a, b))
   else:
       ts.remove((b, a))
   return ts

def get(num):
    res = []
    for i in range(5):
        for j in range(i+1,5):
            res.append((num[i],num[j]))
    return res
def swap(p,a,b):
    num = p[a]
    p[a] = p[b]
    p[b] = num
    return p

def sort(p):
    a=0
    b=1
    c=2
    d=3
    e=4
    if(p[a] > p[b]):
        p = swap(p,a,b)
    if (p[c] > p[d]):
        p = swap(p, c, d)
    if (p[b] > p[d]):
        p = swap(p, b, d)
    if (p[e] > p[b]):
        if(p[e]<p[a]):
            p = swap(p,a,e)
            p = swap(p,d,e)
        else:
            p = swap(p,b,e)
            p = swap(p,d,e)
    else:
        if(p[e]<p[d]):
            p = swap(p,d,e)
    if(p[c] < p[b]):
        if(p[c] < p[a]):
            p = swap(p,a,c)
            p = swap(p,b,c)
        else:
            p = swap(p,c,b)
    else:
        if(p[c]>p[d]):
            p = swap(p,d,c)
    print(p)

nums = [int(i) for i in input().split()]

res = get(nums)
sort(nums)












# step 1
# if nums[0] > nums[1]:
#     tmp = nums[1]
#     nums.remove(tmp)
#     nums.insert(0, tmp)
#
# # step 2
# if nums[2] > nums[3]:
#     tmp = nums[3]
#     nums.remove(tmp)
#     nums.insert(2, tmp)
#
# # step 3
# save_num = 0
# if nums[0] < nums[2]:
#     save_num = nums[1]
# else:
#     save_num = nums[3]
#     tmp = nums[2]
#     nums.remove(tmp)
#     nums.insert(0, tmp)
#
# nums.remove(save_num)
#
# # step 4
# tmp = nums[3]
# nums.remove(tmp)
# if tmp < nums[1]:
#     if tmp < nums[0]:
#         nums.insert(0, tmp)
#     else:
#         nums.insert(1, tmp)
# else:
#     if tmp < nums[2]:
#         nums.insert(2, tmp)
#     else:
#         nums.insert(3, tmp)
#
# # step 5
# if save_num < nums[2]:
#     if save_num < nums[1]:
#         nums.insert(1, save_num)
#     else:
#         nums.insert(2, save_num)
# else:
#     if save_num < nums[3]:
#         nums.insert(3, save_num)
#     else:
#         nums.insert(4, save_num)
#
# # Last list
# print(nums)
