# -*- coding:utf-8 -*-


'''
本文件为麻将函数相关库
'''
import copy
# import numpy as np
import time
import random

# T_SELFMO = [0]*34
# RT1 = [[0]*34]
# import matplotlib.pyplot as plt

cards_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39, 40, 41, 49, 50, 51, 52, 53, 54, 55]

t2s = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [33, 33], [34, 34], [35, 35], [36, 36], [37, 37], [38, 38], [39, 39], [40, 40], [41, 41], [49, 49], [50, 50], [51, 51], [52, 52], [53, 53], [54, 54], [55, 55], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23], [22, 24], [23, 25], [33, 35], [34, 36], [35, 37], [36, 38], [37, 39], [38, 40], [39, 41], [1, 1, 2], [1, 1, 3], [1, 2, 2], [2, 2, 3], [2, 2, 4], [1, 3, 3], [2, 3, 3], [3, 3, 4], [3, 3, 5], [2, 4, 4], [3, 4, 4], [4, 4, 5], [4, 4, 6], [3, 5, 5], [4, 5, 5], [5, 5, 6], [5, 5, 7], [4, 6, 6], [5, 6, 6], [6, 6, 7], [6, 6, 8], [5, 7, 7], [6, 7, 7], [7, 7, 8], [7, 7, 9], [6, 8, 8], [7, 8, 8], [8, 8, 9], [7, 9, 9], [8, 9, 9], [17, 17, 18], [17, 17, 19], [17, 18, 18], [18, 18, 19], [18, 18, 20], [17, 19, 19], [18, 19, 19], [19, 19, 20], [19, 19, 21], [18, 20, 20], [19, 20, 20], [20, 20, 21], [20, 20, 22], [19, 21, 21], [20, 21, 21], [21, 21, 22], [21, 21, 23], [20, 22, 22], [21, 22, 22], [22, 22, 23], [22, 22, 24], [21, 23, 23], [22, 23, 23], [23, 23, 24], [23, 23, 25], [22, 24, 24], [23, 24, 24], [24, 24, 25], [23, 25, 25], [24, 25, 25], [33, 33, 34], [33, 33, 35], [33, 34, 34], [34, 34, 35], [34, 34, 36], [33, 35, 35], [34, 35, 35], [35, 35, 36], [35, 35, 37], [34, 36, 36], [35, 36, 36], [36, 36, 37], [36, 36, 38], [35, 37, 37], [36, 37, 37], [37, 37, 38], [37, 37, 39], [36, 38, 38], [37, 38, 38], [38, 38, 39], [38, 38, 40], [37, 39, 39], [38, 39, 39], [39, 39, 40], [39, 39, 41], [38, 40, 40], [39, 40, 40], [40, 40, 41], [39, 41, 41], [40, 41, 41], [1, 2, 4], [2, 3, 5], [1, 3, 4], [3, 4, 6], [2, 4, 5], [4, 5, 7], [3, 5, 6], [5, 6, 8], [4, 6, 7], [6, 7, 9], [5, 7, 8], [6, 8, 9], [17, 18, 20], [18, 19, 21], [17, 19, 20], [19, 20, 22], [18, 20, 21], [20, 21, 23], [19, 21, 22], [21, 22, 24], [20, 22, 23], [22, 23, 25], [21, 23, 24], [22, 24, 25], [33, 34, 36], [34, 35, 37], [33, 35, 36], [35, 36, 38], [34, 36, 37], [36, 37, 39], [35, 37, 38], [37, 38, 40], [36, 38, 39], [38, 39, 41], [37, 39, 40], [38, 40, 41], [1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [17, 19, 21], [18, 20, 22], [19, 21, 23], [20, 22, 24], [21, 23, 25], [33, 35, 37], [34, 36, 38], [35, 37, 39], [36, 38, 40], [37, 39, 41]]

T2_HALF=[[1,3,4],[2,4,5],[3,5,6],[4,6,7],[5,7,8],
     [0x11, 0x13, 0x14], [0x12, 0x14, 0x15], [0x13, 0x15, 0x16], [0x14, 0x16, 0x17], [0x15, 0x17, 0x18],
     [0x21, 0x23, 0x24], [0x22, 0x24, 0x25], [0x23, 0x25, 0x26], [0x24, 0x26, 0x27], [0x25, 0x27, 0x28],
     [2,3,5],[3,4,6],[4,5,7],[5,6,8],[6,7,9],
     [0x12, 0x13, 0x15], [0x13, 0x14, 0x16], [0x14, 0x15, 0x17], [0x15, 0x16, 0x18], [0x16, 0x17, 0x19],
     [0x22, 0x23, 0x25], [0x23, 0x24, 0x26], [0x24, 0x25, 0x27], [0x25, 0x26, 0x28], [0x26, 0x27, 0x29],
     [6,8,9],[0x16,0x18,0x19],[0x26,0x28,0x29],[1,2,4],[0x11,0x12,0x14],[0x21,0x22,0x24]]

T2_HALF_T1 = [1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37, 5, 6, 7, 8, 9, 21, 22, 23, 24, 25, 37, 38, 39, 40, 41, 9, 0x19, 0x29, 1, 0x11, 0x21]

T2_HALF_T2 = [[3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [6,8],[0x16,0x18],[0x26,0x28],[2,4],[0x12,0x14],[0x22,0x24]]
KING = None
#global variable

w_aa = 6
w_ab = 2

def splitColor(cards):
    color = [[], [], []]
    for card in cards:
        if card & 0xf0 == 0:
            color[0].append(card)
        elif card & 0xf0 == 0x10:
            color[1].append(card)
        elif card & 0xf0 == 0x20:
            color[2].append(card)
    return color

def get_index(list=[], n=[]):
    list = copy.copy(list)
    n = copy.copy(n)
    index = []
    j_used = []
    for i in n:
        for j in range(len(list)):
            if i == list[j] and j not in j_used:
                index.append(j)
                j_used.append(j)
    return index


def split_type_s(cards=[]):
    """
    功能：手牌花色分离，将手牌分离成万条筒字各色后输出
    :param cards: 手牌　[]
    :return: 万,条,筒,字　[],[],[],[]
    """
    cards_wan = []
    cards_tiao = []
    cards_tong = []
    cards_zi = []
    for card in cards:
        if card & 0xF0 == 0x00:
            cards_wan.append(card)
        elif card & 0xF0 == 0x10:
            cards_tiao.append(card)
        elif card & 0xF0 == 0x20:
            cards_tong.append(card)
        elif card & 0xF0 == 0x30:
            cards_zi.append(card)
    return cards_wan, cards_tiao, cards_tong, cards_zi


def get_effective_cards(dz_set=[]):
    """
    获取有效牌
    :param dz_set: 搭子集合 list [[]]
    :return: 有效牌 list []
    """
    effective_cards = []
    for dz in dz_set:
        if len(dz) == 1:
            effective_cards.append(dz[0])
        elif dz[1] == dz[0]:
            effective_cards.append(dz[0])
        elif dz[1] == dz[0] + 1:
            if int(dz[0]) & 0x0F == 1:
                effective_cards.append(dz[0] + 2)
            elif int(dz[0]) & 0x0F == 8:
                effective_cards.append((dz[0] - 1))
            else:
                effective_cards.append(dz[0] - 1)
                effective_cards.append(dz[0] + 2)
        elif dz[1] == dz[0] + 2:
            effective_cards.append(dz[0] + 1)
    effective_cards = set(effective_cards)  # set 和list的区别？
    return list(effective_cards)

def get_32N(cards=[]):
    """
    功能：计算所有存在的手牌的３Ｎ与２Ｎ的集合，例如[3,4,5]　，将得到[[3,4],[3,5],[4,5],[3,4,5]]
    思路：为减少计算量，对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
    :param cards: 手牌　[]
    :return: 3N与2N的集合　[[]]
    """
    cards.sort()
    kz = []
    sz = []
    aa = []
    ab = []
    ac = []
    lastCard = 0
    # 对长度在12张以上的单花色的手牌，当存在顺子时，不再计算搭子
    if len(cards) >= 12:
        for card in cards:
            if card == lastCard:
                continue
            else:
                lastCard = card
            if cards.count(card) >= 3:
                kz.append([card, card, card])
            elif cards.count(card) >= 2:
                aa.append([card, card])
            if card + 1 in cards and card + 2 in cards:
                sz.append([card, card + 1, card + 2])
            else:
                if card + 1 in cards:
                    ab.append([card, card + 1])
                if card + 2 in cards:
                    ac.append([card, card + 2])
    else:
        for card in cards:
            if card == lastCard:
                continue
            else:
                lastCard = card
            if cards.count(card) >= 3:
                kz.append([card, card, card])
            if cards.count(card) >= 2:
                aa.append([card, card])
            if card + 1 in cards and card + 2 in cards:
                sz.append([card, card + 1, card + 2])
            if card + 1 in cards:
                ab.append([card, card + 1])
            if card + 2 in cards:
                ac.append([card, card + 2])
    return kz + sz + aa + ab + ac

    # 判断３２Ｎ是否存在于ｃａｒｄｓ中


def in_cards(t32=[], cards=[]):
    """
    判断３２Ｎ是否存在于ｃａｒｄｓ中
    :param t32: ３Ｎ或2N组合牌
    :param cards: 本次判断的手牌
    :return: bool
    """
    for card in t32:
        if card not in cards:
            return False
    return True


def extract_32N(cards=[], t32_branch=[], t32_set=[]):
    """
    功能：递归计算手牌的所有组合信息，并存储在t32_set，
    思路: 每次递归前检测是否仍然存在３２N的集合,如果没有则返回出本此计算的结果，否则在手牌中抽取该３２N，再次进行递归
    :param cards: 手牌
    :param t32_branch: 本次递归的暂存结果
    :param t32_set: 所有组合信息
    :return: 结果存在t32_set中
    """
    t32N = get_32N(cards=cards)

    if len(t32N) == 0:
        t32_set.extend(t32_branch)
        # t32_set.extend([cards])
        t32_set.append(0)
        t32_set.extend([cards])
    else:
        for t32 in t32N:
            if in_cards(t32=t32, cards=cards):
                cards_r = copy.copy(cards)
                for card in t32:
                    cards_r.remove(card)
                t32_branch.append(t32)
                extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                if len(t32_branch) >= 1:
                    t32_branch.pop(-1)


def tree_expand(cards):
    """
    功能：对extract_32N计算的结果进行处理同一格式，计算万条筒花色的组合信息
    思路：对t32_set的组合信息进行格式统一，分为[kz,sz,aa,ab,xts,leftCards]保存，并对划分不合理的地方进行过滤，例如将３４５划分为35,4为废牌的情况

    :param cards: cards [] 万条筒其中一种花色手牌
    :return: allDeWeight　[kz,sz,aa,ab,xts,leftCards] 去除不合理划分情况的组合后的组合信息
    """
    all = []
    t32_set = []
    extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)
    kz = []
    sz = []
    t2N = []
    aa = []
    length_t32_set = len(t32_set)
    i = 0
    # for i in range(len(t32_set)):
    while i < length_t32_set:
        t = t32_set[i]
        flag = True  # 本次划分是否合理
        if t != 0:
            if len(t) == 3:

                if t[0] == t[1]:
                    kz.append(t)
                else:
                    sz.append(t)  # print (sub)
            elif len(t) == 2:
                if t[1] == t[0]:
                    aa.append(t)
                else:
                    t2N.append(t)

        else:
            '修改，使计算时间缩短'
            leftCards = t32_set[i + 1]
            efc_cards = get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ
            # 去除划分不合理的情况，例如345　划分为34　或35等，对于333 划分为33　和3的情况，考虑有将牌的情况暂时不做处理
            for card in leftCards:
                if card in efc_cards:
                    flag = False
                    break

            if flag:
                all.append([kz, sz, aa, t2N, 0, leftCards])
            kz = []
            sz = []
            aa = []
            t2N = []
            i += 1
        i += 1

    allSort = []  # 给每一个元素排序
    allDeWeight = []  # 排序去重后

    for e in all:
        for f in e:
            if f == 0:  # 0是xts位，int不能排序
                continue
            else:
                f.sort()
        allSort.append(e)

    for a in allSort:
        if a not in allDeWeight:
            allDeWeight.append(a)

    allDeWeight = sorted(allDeWeight, key=lambda k: (len(k[0]), len(k[1]), len(k[2])), reverse=True)  # 居然可以这样排序！！
    return allDeWeight


def assess_card(card):
    value = 0
    if card & 0x0f == 1 or card & 0x0f == 9:
        value = 3
    elif card & 0x0f == 2 or card & 0x0f == 8:
        value = 4
    elif card & 0x0f == 3 or card & 0x0f == 7:
        value = 8
    elif card & 0x0f == 4 or card & 0x0f == 6:
        value = 5
    elif card & 0x0f == 5:
        value = 6
    return value


def translate16_33(i):
    i = int(i)
    if i >= 0x01 and i <= 0x09:
        i = i - 1
    elif i >= 0x11 and i <= 0x19:
        i = i - 8
    elif i >= 0x21 and i <= 0x29:
        i = i - 15
    elif i >= 0x31 and i <= 0x37:
        i = i - 22
    else:
        print ("translate16_33 is error,i=%d" % i)
        i = -1
    return i

def translate33_16(i): # 将下标转换成16进制的牌
    if 0 <= i < 9:
        return i + 1
    elif 9 <= i < 18:
        return i + 8
    elif 18<= i < 27:
        return i + 15
    elif 27 <= i < 34:
        return i + 22
    else:
        print("[INFO_ZW]:INPUT ERROR")

def translate33_10(i):  #转换成10进制
    if 0 <= i < 9:
        return i + 1
    elif 9 <= i < 18:
        return i + 2
    elif 18 <= i < 27:
        return i + 3
    elif 27 <= i < 34:
        return i + 4
    else:
        print("[INFO_ZW]:INPUT ERROR")

def translate16_10(i):
    # 16进制转进制
    return i//16 * 10 + i%16

# 把对应的十六进制牌转换成[]*34的索引值
# 如0x01代表的是第0个数
def convert_hex2index(a):
    if a > 0 and a < 0x10:
        return a - 1
    if a > 0x10 and a < 0x20:
        return a - 8
    if a > 0x20 and a < 0x30:
        return a - 15
    if a > 0x30 and a < 0x40:
        return a - 22


def trandfer_discards(discards, discards_op, handcards, type=34):
    discards_map = {
        0x01: 0,
        0x02: 1,
        0x03: 2,
        0x04: 3,
        0x05: 4,
        0x06: 5,
        0x07: 6,
        0x08: 7,
        0x09: 8,
        0x11: 9,
        0x12: 10,
        0x13: 11,
        0x14: 12,
        0x15: 13,
        0x16: 14,
        0x17: 15,
        0x18: 16,
        0x19: 17,
        0x21: 18,
        0x22: 19,
        0x23: 20,
        0x24: 21,
        0x25: 22,
        0x26: 23,
        0x27: 24,
        0x28: 25,
        0x29: 26,
        0x31: 27,
        0x32: 28,
        0x33: 29,
        0x34: 30,
        0x35: 31,
        0x36: 32,
        0x37: 33,
    }
    # print ("discards=",discards)
    # print ("discards_op=",discards_op)
    left_num = [4] * type
    discards_list = [0] * type
    # print "discards",discards
    for per in discards:
        for item in per:
            # print "per",per
            discards_list[discards_map[item]] += 1
            left_num[discards_map[item]] -= 1
    for seat_op in discards_op:
        for op in seat_op:
            for item in op:
                discards_list[discards_map[item]] += 1
                left_num[discards_map[item]] -= 1
    for item in handcards:
        left_num[discards_map[item]] -= 1

    # print ("trandfer_discards,left_num=",left_num)
    return left_num, discards_list


# 获取ｌｉｓｔ中的最小值和下标
def get_min(list=[]):
    min = 14
    index = 0

    for i in range(len(list)):
        if list[i] < min:
            min = list[i]
            index = i
    return min, index

    # [aaa,abc,aa,ab,aab,xts,t1]
    # 递归检测aab


def get_a_fastWinning(flag=True, a_CB=[], all_fastWinning=[]):
    # flag=False
    if flag == False:
        if a_CB not in all_fastWinning:
            all_fastWinning.append(a_CB)
        return

    flag = False
    for T1 in a_CB[-1]:
        if T1<0x31:
            for aa in a_CB[2]:
                if T1 + 1 in aa or T1 + 2 in aa or T1 - 1 in aa or T1 - 2 in aa:
                    a_CP = copy.deepcopy(a_CB)
                    a_CP[-1].remove(T1)
                    a_CP[2].remove(aa)
                    a_CP[3].append(sorted([aa[0], aa[1], T1]))
                    a_CP[3].sort()
                    flag = True
                    get_a_fastWinning(flag, a_CP, all_fastWinning)

            for ab in a_CB[4]:

                if T1 + 1 in ab or T1 + 2 in ab or T1 - 1 in ab or T1 - 2 in ab or T1 in ab:
                    t2_5 = sorted([ab[0], ab[1], T1])

                    a_CP = copy.deepcopy(a_CB)
                    a_CP[4].remove(ab)
                    a_CP[-1].remove(T1)

                    if T1 in ab:
                        a_CP[3].append(t2_5)
                        a_CP[3].sort()
                        # a_CP[3].sort(lambda k:k=)
                    else:
                        a_CP[5].append(t2_5)
                        a_CP[5].sort()

                    flag = True

                    get_a_fastWinning(flag, a_CP, all_fastWinning)
    if flag == False:
        get_a_fastWinning(flag, a_CB, all_fastWinning)


def fast_win_CS(all):
    """
    :return:
    """
    all_fastWinning = []
    for a in all:
        # [kz,sz,aa,aab,ab,abc,wn,left_cards]
        a_fastWinning = [a[0], a[1], a[2], [], a[3], [], a[4], a[5]]  #
        get_a_fastWinning(True, a_fastWinning, all_fastWinning)
    print('all_fastWinning', all_fastWinning)

    TMP = []
    for e in all_fastWinning:
        # for i in e:
        #     while(type(i)==list):

        # if e not in TMP:
        temp = deepcopy(e)
        for t2_5 in e[5]:
            # print "test...", t2_5, t2_5 in T2_HALF
            if t2_5 in T2_HALF:
                index = T2_HALF.index(t2_5)
                # print index
                temp[5].remove(t2_5)
                t2 = T2_HALF_T2[index]
                t1 = T2_HALF_T1[index]

                t2s = e[2] + e[3] + e[4] + e[5]
                t2s.remove(t2_5)
                # for t2p in t2s:
                #     if t1 + 1 in t2p or t1 + 2 in t2p or t1 - 1 in t2p or t1 - 2 in t2p or t1 in t2p:
                #         temp=[]
                #         break
                # if temp==[]:
                #     break
                temp[4].append(t2)
                temp[4].sort()
                temp[-1].append(t1)
                temp[-1].sort()

        if temp!=[] and temp not in TMP:
            TMP.append(temp)
    all_fastWinning = TMP
    all_fastWinning.sort(key=lambda k: len(k[-1]))
    return all_fastWinning

# 全局变量
# 向听数随轮数的分布表
Txts = [[4, 14, 81, 286, 680, 1294, 1999, 2906, 3709, 4525, 5314, 5964, 6518, 6990],
        [101, 361, 952, 1910, 2865, 3763, 4421, 4625, 4545, 4269, 3893, 3518, 3121, 2787],
        [994, 2125, 3268, 3961, 4109, 3601, 2835, 2100, 1536, 1094, 729, 480, 342, 209],
        [3284, 3951, 3776, 2948, 1951, 1145, 657, 319, 178, 93, 50, 27, 12, 9],
        [3708, 2752, 1607, 781, 345, 165, 61, 27, 13, 3, 3, 2, 0, 0],
        [1569, 670, 270, 87, 33, 13, 10, 7, 3, 4, 3, 2, 1, 2],
        [308, 114, 41, 23, 12, 13, 11, 12, 10, 8, 5, 2, 3, 0],
        [29, 12, 4, 3, 4, 5, 5, 2, 4, 2, 1, 3, 3, 2],
        [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        ]
Txts_transpose = list(zip(*Txts))
[(4, 101, 994, 3284, 3708, 1569, 308, 29, 2), (14, 361, 2125, 3951, 2752, 670, 114, 12, 0),
 (81, 952, 3268, 3776, 1607, 270, 41, 4, 0), (286, 1910, 3961, 2948, 781, 87, 23, 3, 0),
 (680, 2865, 4109, 1951, 345, 33, 12, 4, 0), (1294, 3763, 3601, 1145, 165, 13, 13, 5, 0),
 (1999, 4421, 2835, 657, 61, 10, 11, 5, 0), (2906, 4625, 2100, 319, 27, 7, 12, 2, 1),
 (3709, 4545, 1536, 178, 13, 3, 10, 4, 1), (4525, 4269, 1094, 93, 3, 4, 8, 2, 1),
 (5314, 3893, 729, 50, 3, 3, 5, 1, 1), (5964, 3518, 480, 27, 2, 2, 2, 3, 1),
 (6518, 3121, 342, 12, 0, 1, 3, 3, 1), (6990, 2787, 209, 9, 0, 2, 0, 2, 0)]

for i in range(len(Txts_transpose)):
    Txts_transpose[i] = list(Txts_transpose[i])

    for j in range(len(Txts_transpose[i])):
        Txts_transpose[i][j] = float(Txts_transpose[i][j]) / 10000
# print Txts_transpose
        # 求每轮的平均向听数
avg_xts = [0] * 14
for round in range(len(Txts_transpose)):
    round_xts = Txts_transpose[round]
    for x in range(len(round_xts)):
        xts = round_xts[x]
        avg_xts[round] += (xts * (x + 1))
# print avg_xts


def get_t2info():
    dzSet = [0] * (34 + 15 * 3)  # 34+15*3
    # 生成搭子有效牌表
    dzEfc = [0] * (34 + 15 * 3)
    for i in range(len(dzSet)):
        if i <= 33:  # aa
            card = int(i / 9) * 16 + i % 9 + 1
            dzSet[i] = [card, card]
            dzEfc[i] = [card]
        elif i <= 33 + 8 * 3:  # ab
            card = int((i - 34) / 8) * 16 + (i - 34) % 8 + 1
            dzSet[i] = [card, card + 1]
            if card & 0x0f == 1:
                dzEfc[i] = [card + 2]
            elif card & 0x0f == 8:
                dzEfc[i] = [card - 1]
            else:
                dzEfc[i] = [card - 1, card + 2]
        else:
            card = int((i - 34 - 8 * 3) / 7) * 16 + (i - 34 - 8 * 3) % 7 + 1

            dzSet[i] = [card, card + 2]
            dzEfc[i] = [card + 1]

    efc_dzindex = {}  # card->34+8+8+8+7+7+7

    cardSet = []
    for i in range(34):
        cardSet.append(int(i // 9 * 16 + i % 9 + 1))
    for card in cardSet:
        efc_dzindex[card] = []
        efc_dzindex[card].append(translate16_33(card)) #加aa
        color = int(card / 16)
        if color != 3:
            if card & 0x0f == 1:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)

            elif card & 0x0f == 2:  # 13 34
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)
            elif card & 0x0f == 8:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
            elif card & 0x0f == 9:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
            else:
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) - 2)
                efc_dzindex[card].append(33 + 24+color * 7 + (card & 0x0f) - 1)
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)

    return dzSet, dzEfc, efc_dzindex


def get_t3info():
    t3Set=[]
    for i in range(34):
        card=int(i/9)*16+i%9+1
        t3Set.append([card,card,card])
    for i in range(34,34+7*3):
        card = int((i-34)/7)*16+(i-34)%7+1
        t3Set.append([card,card+1,card+2])
    return t3Set

dzSet,dzEfc,efc_dzindex=get_t2info()
t3Set=get_t3info()
# print "t3set", t3Set

[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [33, 33], [34, 34], [35, 35], [36, 36], [37, 37], [38, 38], [39, 39], [40, 40], [41, 41], [49, 49], [50, 50], [51, 51], [52, 52], [53, 53], [54, 54], [55, 55], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23], [22, 24], [23, 25], [33, 35], [34, 36], [35, 37], [36, 38], [37, 39], [38, 40], [39, 41]]
[[1], [2], [3], [4], [5], [6], [7], [8], [9], [17], [18], [19], [20], [21], [22], [23], [24], [25], [33], [34], [35], [36], [37], [38], [39], [40], [41], [49], [50], [51], [52], [53], [54], [55], [3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7], [19], [17, 20], [18, 21], [19, 22], [20, 23], [21, 24], [22, 25], [23], [35], [33, 36], [34, 37], [35, 38], [36, 39], [37, 40], [38, 41], [39], [2], [3], [4], [5], [6], [7], [8], [18], [19], [20], [21], [22], [23], [24], [34], [35], [36], [37], [38], [39], [40]]

{1: [0, 35], 2: [1, 58, 36], 3: [2, 34, 59, 37], 4: [3, 35, 60, 38], 5: [4, 36, 61, 39], 6: [5, 37, 62, 40], 7: [6, 38, 63, 41], 8: [7, 39, 64], 9: [8, 40], 17: [9, 43], 18: [10, 65, 44], 19: [11, 42, 66, 45], 20: [12, 43, 67, 46], 21: [13, 44, 68, 47], 22: [14, 45, 69, 48], 23: [15, 46, 70, 49], 24: [16, 47, 71], 25: [17, 48], 33: [18, 51], 34: [19, 72, 52], 35: [20, 50, 73, 53], 36: [21, 51, 74, 54], 37: [22, 52, 75, 55], 38: [23, 53, 76, 56], 39: [24, 54, 77, 57], 40: [25, 55, 78], 41: [26, 56], 49: [27], 50: [28], 51: [29], 52: [30], 53: [31], 54: [32], 55: [33]}


def generate_t2():
    """
    生成t2表
    :return: t2表
    """
    cards  = []
    t2s = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [17, 17], [18, 18], [19, 19], [20, 20], [21, 21], [22, 22], [23, 23], [24, 24], [25, 25], [33, 33], [34, 34], [35, 35], [36, 36], [37, 37], [38, 38], [39, 39], [40, 40], [41, 41], [49, 49], [50, 50], [51, 51], [52, 52], [53, 53], [54, 54], [55, 55], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23], [22, 24], [23, 25], [33, 35], [34, 36], [35, 37], [36, 38], [37, 39], [38, 40], [39, 41]]
    t3s = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [17, 17, 17],
     [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25],
     [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37], [38, 38, 38], [39, 39, 39], [40, 40, 40],
     [41, 41, 41], [49, 49, 49], [50, 50, 50], [51, 51, 51], [52, 52, 52], [53, 53, 53], [54, 54, 54], [55, 55, 55],
     [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [17, 18, 19], [18, 19, 20],
     [19, 20, 21], [20, 21, 22], [21, 22, 23], [22, 23, 24], [23, 24, 25], [33, 34, 35], [34, 35, 36], [35, 36, 37],
     [36, 37, 38], [37, 38, 39], [38, 39, 40], [39, 40, 41]]

    #生成所有牌值表
    for i in range(34):
        cards.append(t2s[i][0])
    # print cards
    #花色对子 0-27
    for i in range(0, 9*3)+range(34,len(t2s)):
        for card in t2s[i]:
            for j in [-2, -1, 0, 1, 2]:
                if card + j in cards:
                    tmp = copy.copy(t2s[i])
                    tmp.append(card + j)
                    tmp.sort()
                    if tmp not in t2s and tmp not in t3s:
                        t2s.append(tmp)
    return t2s

def pre_king(king_card=None):
    """
    计算宝牌的前一张
    :param king_card: 宝牌
    :return:宝牌的前一张牌
    """
    if king_card == None:
        return None
    if king_card == 0x01:
        return 0x09
    elif king_card == 0x11:
        return 0x19
    elif king_card == 0x21:
        return 0x29
    elif king_card == 0x31:
        return 0x37
    else:
        return king_card - 1

def t2tot3_info(T_selfmo=[], RT1=[], RT2=[], RT3=[]):
    """
    生成t2转化为t3的状态
    :return: "t2": [[t2_,t3,t1_left,valid,p]]
    """
    t2tot3_dict={}
    for t2 in t2s:
        t2tot3_dict[str(t2)]=[]
        if len(t2) == 2:
            t2_decompose_valid = [t2]
            t1_left = [[]]

        elif len(t2) == 3:
            t2_decompose = [[t2[0],t2[1]],[t2[0],t2[2]],[t2[1],t2[2]]] #存储拆分后的t2
            t1 = [[t2[2]], [t2[1]], [t2[0]]]
            t2_decompose_valid = []
            t1_left = []
            for i in range(len(t2_decompose)):
                t2_ = t2_decompose[i]
                if abs(t2_[1]-t2_[0]) <= 2:
                    if t2_ not in t2_decompose_valid:
                        t2_decompose_valid.append(t2_)
                        t1_left.append(t1[i])

        for j in range(len(t2_decompose_valid)):
            t2_ = t2_decompose_valid[j]
            valids = dzEfc[dzSet.index(t2_)]
            for valid_card in valids:
                info = []
                t3 = copy.copy(t2_)
                t3.append(valid_card)
                t3.sort()
                info.append(t2_)
                info.append(t3)
                info.append(t1_left[j])
                info.append(valid_card)

                index = convert_hex2index(valid_card)
                r = T_selfmo[index]

                if RT1 == []:
                    if t2_[0]==t2_[1]:
                        # info.append(r*w_aa)
                        # if T_selfmo[convert_hex2index(t2_[0])]==2: #奖励机制，可以杠牌
                        #     info.append(w_aa+1)
                        # else:
                        if t2_[0] == pre_king(KING) and T_selfmo[convert_hex2index(t2_[0])]!=0:
                            info.append(w_aa*2)
                        else:
                            info.append(w_aa)
                    else:
                        info.append(w_ab)
                        # if len(valids)>1:
                        #     info.append(w_ab)
                        # else:
                        #     info.append(w_ab-0.5)
                else:
                    if t2_[0] == t2_[1]:
                        rt1 = RT1[2][index]
                        rt2 = RT2[2][index]
                        rt3 = RT3[2][index]
                        p = r * (4 - rt1 - rt2 - rt3) * 1.2
                    else:
                        p = r * (2 - rt3) * 1.2

                    info.append(p)

                t2tot3_dict[str(t2)].append(info)
    return t2tot3_dict

def t1tot2_info(T_selfmo):
    """
    t1转换为t2的状态
    :return:
    """
    t1tot2_dict = {}
    for card in cards_value:
        t1tot2_dict[str(card)]=[]
        if card < 0x31:
            valid_tile = [card-2,card-1,card,card+1,card+2]
        else:
            valid_tile = [card]
        # t2_transform=[]
        for tile in valid_tile:
            if tile in cards_value:
                t2 = sorted([card,tile])
                # if t2_ not in t2_transform
                # t2_transform.append(t2)
                info = []
                info.append(t2)
                info.append(tile)
                info.append(T_selfmo[convert_hex2index(tile)])
                t1tot2_dict[str(card)].append(info)
    return t1tot2_dict

def t1tot3_info(T_selfmo=[], RT1=[], RT2=[], RT3=[]):
    """

    :param T_selfmo:
    :param RT1:
    :param RT2:
    :param RT3:
    :return: {"t1":[[t3,t2(valid card),p]]}

    """
    t1tot3_dict={}
    # t1tot2_dict = t1tot2()
    for card in cards_value:
        t1tot3_dict[str(card)]=[]
        if card < 0x31:
            valid_tiles = [[card-2,card-1],[card-1,card+1],[card,card],[card+1,card+2]]
        else:
            valid_tiles = [[card,card]]
        for t2 in valid_tiles:
            if t2[0] in cards_value and t2[1] in cards_value:
                info = []
                t3=copy.copy(t2)
                t3.append(card)
                t3.sort()
                info.append(t3)
                info.append(t2)
                # info.append()
                # for valid_tile in t2:
                #     index = convert_hex2index(valid_tile)
                #     r = T_selfmo[index]
                index1 = convert_hex2index(t2[0])
                index2 = convert_hex2index(t2[1])
                if RT1 == []:
                    if t2[0] == t2[1]:
                        # info.append(T_selfmo[index1] * T_selfmo[index2] * w_aa)
                        info.append([1,w_aa])
                    else:
                        # info.append(T_selfmo[index1] * T_selfmo[index2] * w_ab)
                        info.append([1,w_ab])
                else:
                    if t2[0] == t2[1]:
                        # rt1 = RT1[2][index1]
                        # rt2 = RT2[2][index1]
                        # rt3 = RT3[2][index1]
                        p1 = T_selfmo[index1] * T_selfmo[index2] *(4-RT1[2][index2]-RT2[2][index2]-RT3[2][index2]) * 1.2
                        p2 = T_selfmo[index2] * T_selfmo[index1] *(4-RT1[2][index1]-RT2[2][index1]-RT3[2][index1]) * 1.2
                    else:
                        p1 = T_selfmo[index1] * T_selfmo[index2] *(2-RT3[2][index2])*1.2
                        p2 = T_selfmo[index2] * T_selfmo[index1] *(2-RT3[2][index1])*1.2
                    info.append((p1+p2)/2)
                t1tot3_dict[str(card)].append(info)
    return t1tot3_dict


def is_1l_list(l):
    for i in l:
        if type(i)==list:
            return False
    return True

def deepcopy(src):
    dst = []
    for i in src:
        if type(i)==list and not is_1l_list(i):
            i = deepcopy(i)
        dst.append(copy.copy(i))
    return dst

# if __name__ == '__main__':
#     print fast_win_CS([[[], [], [[3, 3], [52, 52]], [[5, 6], [8, 9], [34, 35], [39, 41]], 4, [2, 37]]])
#
#     # print t1tot3()
#
#
# [[[], [], [[52, 52]], [[2, 3, 3]], [[5, 6], [8, 9], [34, 35], [39, 41]], [], 4, [37]],
#  [[], [], [[52, 52]], [[2, 3, 3]], [[5, 6], [8, 9], [34, 35]], [[37, 39, 41]], 4, []]]
