# -*- coding: utf-8 -*-
# !/usr/bin/python

import random
import copy
import logging


# 十进制转十六进制
def f10_to_16(num):  # 用16进制表示
    a = int(num / 10)
    b = num - a * 10
    return a * 16 + b


# 十六进制转十进制
def f16_to_10(num):  # 用十进制表示
    a = int(num / 16)
    b = num - 16 * a
    return a * 10 + b


def list10_to_16(cardlist):
    cardlist2 = []
    for i in cardlist:
        cardlist2.append(f10_to_16(i))
        cardlist2.sort()
    return cardlist2


def list16_to_10(cardlist):
    temp_cardlist = []
    for card in cardlist:
        temp_cardlist.append(f16_to_10(card))
    temp_cardlist.sort()
    return temp_cardlist


def list10_to_16_2(cardlist):
    cardlist2 = []
    for i in cardlist:
        cardlist2.append(f10_to_16(i))
        # cardlist2.sort()
    return cardlist2


def fulu_translate(fulu):
    actions2 = []
    for i in fulu:
        actions2.append(list10_to_16(i))
    return actions2

'''
这里是十进制还是16进制不得而知啊
需要知道获得的手牌是如何编码的
其次需要知道我们要输入模型的x是什么
'''

def translate3(op_card):  # 16进制op_card转换到 0-33 34转换---好像是十进制
    if 1 <= op_card <= 9:
        op_card = op_card - 1
    elif 17 <= op_card <= 25:
        op_card = op_card - 8
    elif 33 <= op_card <= 41:
        op_card = op_card - 15
    elif 49 <= op_card <= 55:
        op_card = op_card - 22
    elif op_card == 255:
        op_card = 34
    return op_card


def card_to_index(card):  # 十进制card转换成0-33下标
    index = 0
    if 0 < card < 10:
        index = card - 1
    elif 10 < card < 20:
        index = card - 2
    elif 20 < card < 30:
        index = card - 3
    elif 30 < card < 38:
        index = card - 4
    else:
        logging.error("card:", card, "输入错误，请检查！！")

    return index


def discard_translate(cardlist):
    cardlist2 = [0] * 34
    for i in cardlist:
        temp = f10_to_16(i)
        temp2 = translate3(temp)
        cardlist2[temp2] += 1
    return cardlist2


# 检查生成的手牌是否合理
def check(cardlist):
    for i in cardlist:
        if cardlist.count(i) > 4:
            return False

    return True


# 随机发放副露（未检查）
def random_deal_fulu(random_deal_handcards):
    shunzi = [  # 所有顺子
        # 万
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        # 条
        [11, 12, 13],
        [12, 13, 14],
        [13, 14, 15],
        [14, 15, 16],
        [15, 16, 17],
        [16, 17, 18],
        [17, 18, 19],
        # 筒
        [21, 22, 23],
        [22, 23, 24],
        [23, 24, 25],
        [24, 25, 26],
        [25, 26, 27],
        [26, 27, 28],
        [27, 28, 29], ]

    kezi = [  # 所有刻子
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
        [9, 9, 9],

        [11, 11, 11],
        [12, 12, 12],
        [13, 13, 13],
        [14, 14, 14],
        [15, 15, 15],
        [16, 16, 16],
        [17, 17, 17],
        [18, 18, 18],
        [19, 19, 19],

        [21, 21, 21],
        [22, 22, 22],
        [23, 23, 23],
        [24, 24, 24],
        [25, 25, 25],
        [26, 26, 26],
        [27, 27, 27],
        [28, 28, 28],
        [29, 29, 29],

        [31, 31, 31],
        [32, 32, 32],
        [33, 33, 33],
        [34, 34, 34],
        [35, 35, 35],
        [36, 36, 36],
        [37, 37, 37],
    ]
    list1 = shunzi + kezi
    length = len(random_deal_handcards)  # 手牌长度
    s = 4 - (length - 2)  # 应该构建的幅露长度
    fulu = random.sample(list1, s)
    return fulu


# 列表的减法
def list_sub(list1, list2):
    temp = copy.deepcopy(list1)
    for i in list2:
        temp.remove(i)  # 从list1中去除list2
    return temp


# 给某为玩家发指定数量的牌
def distribution_cards(card_library, num):
    cards_list = []
    for i in range(num):
        cards_list.append(card_library.pop())  # 发牌
    cards_list.sort()

    return cards_list


def translate34_to_136(cardlist):
    output = [0] * 136
    for i in range(len(cardlist)):
        if cardlist[i] != 0:
            output[i * 4 + cardlist[i] - 1] = 1
    return output


def translate34_to_136_sr(cardlist):
    output = [0] * 136
    for index, card_count in zip(range(34), cardlist):  # card_count = 对应牌的数量
        if card_count > 0:
            output[index * 4 + card_count - 1] = 1
    return output


def index_to_card(index):  # 下标转换成十进制的card
    card = 0
    if 0 <= index <= 8:
        card = index + 1
    elif 8 < index <= 17:
        card = index + 2
    elif 17 < index <= 26:
        card = index + 3
    elif 26 < index <= 33:
        card = index + 4
    else:
        logging.error("index:", index, "输入错误，请检查")
    return card


def decode_num_cards(nums_cards):
    handcards = []
    for index, card_num in zip(range(34), nums_cards):
        for j in range(card_num):
            handcards.append(index_to_card(index))
    return handcards


def decode_136_to_34_sr(fea_list_136):  # 返回手牌数据， 用十进制表示
    cards_nums = [0] * 34
    for i in range(34):
        for j in range(0, 4):
            if fea_list_136[i * 4 + j] == 1:
                cards_nums[i] = j + 1
                continue
    handcards = decode_num_cards(cards_nums)
    return handcards


def decode_batch_one_hot(max_, one_hot_encode):
    code_len = len(one_hot_encode)
    code_num = int(code_len / max_)
    decode = [0] * code_num
    for i in range(code_num):
        for j in range(0, max_ - 1):
            if one_hot_encode[i * max_ + j] == 1:
                decode[i] = j + 1
                continue
    return decode


def batch_one_hot_to_cards(max_, one_hot_encode):
    num_of_cards = decode_batch_one_hot(max_, one_hot_encode)
    cards = decode_num_cards(num_of_cards)
    return cards


def translate(i):
    if i >= 1 and i <= 9:
        return i
    elif i >= 10 and i <= 18:
        return i + 1
    elif i >= 19 and i <= 27:
        return i + 2
    elif i >= 28 and i <= 34:
        return i + 3
    else:
        logging.info('Error !')


# 对某一数据进行onehot编码,max为数据中的最大值
def one_hot(max, x):
    newlist = [0] * max
    newlist[x - 1] = 1
    return newlist


# 为某一连续特征进行onehot编码
def batch_one_hot(max, list1):
    feature = []
    for i in list1:
        fea = one_hot(max, i)
        feature.extend(fea)
    return feature


def get_comm_single_card(handcards):  # 获取孤张，一张牌跟他相邻2以内都没有牌，则作为孤张
    '''
    获取平胡孤张 一张牌跟他相邻2以内都没有牌，则作为孤张
    :param handcards: 手牌用十进制表示
    :return: 孤张列表
    '''

    def is_single_card(card, handcards):
        if card - 1 in handcards or card - 2 in handcards or card + 1 in handcards or card + 2 in handcards:
            return False
        else:
            return True

    single_cards = []
    temp_handcards = copy.deepcopy(handcards)
    L = set(temp_handcards)
    for card in L:
        if handcards.count(card) == 1:  # 需要判断前后两张内是否有牌
            if card & 0xF0 != 0x30:
                if is_single_card(card, handcards):
                    single_cards.append(card)
            else:
                single_cards.append(card)
    single_cards.sort()
    return single_cards


def get_qidui_single_card(handcards):  # 获取七对的孤张
    single_cards = []
    L = set(handcards)

    for card in L:
        if handcards.count(card) == 1:
            single_cards.append(card)
    single_cards.sort()
    return single_cards


def get_91_single_card(handcards):  # 91
    single_cards = []
    L = set(handcards)
    for card in L:
        # print(card & 0xF0)
        if (card & 0xF0) / 16 < 3:
            if not (card & 0x0F in [1, 9]):  # 非19边张
                for _ in range(handcards.count(card)): single_cards.append(card)
    single_cards.sort()
    return single_cards


def get_13_single_card(handcards):  # 13

    single_cards = copy.deepcopy(handcards)
    useful_cards = []
    L = set(single_cards)  # 去除重复手牌
    L_num0 = []  # 万数牌
    L_num1 = []  # 条数牌
    L_num2 = []  # 筒数牌
    for i in L:
        if i & 0xf0 == 0x30:
            # 计算字牌的向听数
            useful_cards.append(i)
        if i & 0xf0 == 0x00:
            L_num0.append(i & 0x0f)
        if i & 0xf0 == 0x10:
            L_num1.append(i & 0x0f)
        if i & 0xf0 == 0x20:
            L_num2.append(i & 0x0f)

    def get_useful_cards(tiles, type):  # 获取有用的组合牌

        numerous_cards = [[1, 5, 9], [1, 4, 7], [1, 4, 8], [1, 4, 9], [1, 5, 8], [1, 6, 9], [2, 5, 8], [2, 5, 9],
                          [2, 6, 9], [3, 6, 9]]

        en_numerous = [(tiles.count(1) + tiles.count(5) + tiles.count(9)),
                       (tiles.count(1) + tiles.count(4) + tiles.count(7)),
                       (tiles.count(1) + tiles.count(4) + tiles.count(8)),
                       (tiles.count(1) + tiles.count(4) + tiles.count(9)),
                       (tiles.count(1) + tiles.count(5) + tiles.count(8)),
                       (tiles.count(1) + tiles.count(6) + tiles.count(9)),
                       (tiles.count(2) + tiles.count(5) + tiles.count(8)),
                       (tiles.count(2) + tiles.count(5) + tiles.count(9)),
                       (tiles.count(2) + tiles.count(6) + tiles.count(9)),
                       (tiles.count(3) + tiles.count(6) + tiles.count(9))]
        # 如果有效牌为两张
        max_useful_card = max(en_numerous)
        index = en_numerous.index(max_useful_card)

        cards = numerous_cards[index]

        if max_useful_card == 2:  # 简单的规则过滤
            if len(tiles) > 2:  # 该类牌大于2张
                if 1 in tiles:
                    if 9 in tiles:
                        cards = [1, 9]
                    elif 4 in tiles:
                        cards = [1, 4]
                    elif 5 in tiles:
                        cards = [1, 5]
                elif 9 in tiles:
                    if 6 in tiles:
                        cards = [6, 9]
                    elif 5 in tiles:
                        cards = [5, 9]
        real_cards = [(card + type * 16) for card in cards]  # transfer real_cards
        return real_cards

    useful_cards.extend(get_useful_cards(L_num0, 0))
    useful_cards.extend(get_useful_cards(L_num1, 1))
    useful_cards.extend(get_useful_cards(L_num2, 2))

    for card in useful_cards:
        if card in single_cards: single_cards.remove(card)
    single_cards.sort()
    return single_cards

# if __name__ == '__main__':
#     pass
# handcards = list10_to_16([1,2,4,6,9,17,18,19, 21,31,31,32,34,35])
# print(get_comm_single_card(handcards), get_91_single_card(handcards), get_qidui_single_card(handcards),list16_to_10(get_13_single_card(handcards)))
# print(6 in [1,2,6,3,9] and 9 in [1,2,6,3,9])
# print(21&0x0F)
