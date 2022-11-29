#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 16:17
# @Author  : Joisen
# @File    : feature_extract_me.py

import copy
import numpy as np
# from feature.extract import get_param
from mah_tool.suphx_extract_features import tool
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo
import torch
'''
    card_preprocess_sr_suphx(handCards0, fulu_, king_card, discards_seq, remain_card_num,
                             self_king_num, fei_king_nums, round_, dealer_flag=[1, 0, 0, 0],
                             search=True):
    each feature:4 * 9
'''

'''
    对于高手玩家的手牌：直接调用card_feature_encode(handCard0, 4) ： 4 * 4 * 9
    对于单个玩家吃的牌：使用的特征为4 * 4 * 9
    对于玩家碰的牌：使用的特征为1 * 4 * 9，先将副露中每个玩家的碰牌进行提取，再调用card_feature_encode(peng_fulu,1),过程重复4次
    对于玩家杠的牌：使用的特征为1 * 4 * 9，和玩家碰的牌的处理过程一样
    对于玩家丢的牌：4 * 9
    桌面上未出现的牌：4 * 9
    宝牌：4 * 9
    飞宝：
'''
def get_remain_card(fulu, handCards0, discards):
    '''
    :param fulu: 所有玩家副露中的牌
    :param handCards0: 高手玩家的手牌
    :param discards: 所有玩家丢弃的牌（不含副露中的牌）
    :return: 返回 除当前玩家的手牌 和 桌面上已经出现的牌 外 剩余的牌
    '''
    all_card = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,
                17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,
                33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,
                49,49,49,49,50,50,50,50,51,51,51,51,52,52,52,52,53,53,53,53,54,54,54,54,55,55,55,55]
    for fulu_ in fulu:
        for fl in fulu_:
            for card in fl:
                all_card.remove(card)
    for card in handCards0:
        all_card.remove(card)
    for dis in discards:
        for card in dis:
            all_card.remove(card)
    return all_card

def split_eat_pong_gang(fulu):
    '''
    :param fulu: 副露
    :return: 返回 对副露进行分离分离后各玩家的吃、碰、杠; 其中碰和杠进行了处理,返回的是碰和杠了哪些牌
    例如：所有玩家碰的牌为：[[25, 19], [8, 1], [39], [3]]
    '''
    eat = [[], [], [], []]
    pong = [[], [], [], []]
    gang = [[], [], [], []]
    for i in range(len(fulu)):
        for fulu_ in fulu[i]:
            if len(fulu_) == 3:
                if fulu_[0] == fulu_[1]:
                    pong[i].append(fulu_)
                else:
                    eat[i].append(fulu_)
            elif len(fulu_) == 4:
                gang[i].append(fulu_)
    pong_ = []
    for p in pong:
        pong_.append(list(set([x for a in p for x in a])))
    gang_ = []
    for p in gang:
        gang_.append(list(set([x for a in p for x in a])))

    return eat, pong_, gang_

def get_self_King(self_king_num, king_card):
    '''
    :param self_king_num: 具体数字
    :param king_card:
    :return:
    '''
    ret_list = []
    for i in range(self_king_num):
        ret_list.append(king_card)
    return ret_list

def get_players_fei_king(fei_king_nums, king_card):
    '''
    :param fei_king_nums: 格式：[0,0,0,0]
    :param king_card:
    :return: [[], [], [], []]
    '''
    ret_list = [[],[],[],[]]
    for num in range(len(fei_king_nums)):
        for i in range(fei_king_nums[num]):
            ret_list[num].append(king_card)
    return ret_list


# 4*9 编码
def card_feature_encode(cards_, channels):
    '''
        对牌集进行特征编码 可编码 手牌、碰牌、杠牌、暗杠、丢牌、剩余牌
        :param cards_:  牌或者牌集
        :param channels: 通道数
        :return:
    '''
    cards = copy.deepcopy(cards_)  # 深拷贝，防止修改原本的数据

    if not isinstance(cards, list):  # 如果只有一张牌 则不是list  需要将这张牌也转换成list
        cards = [cards]
    cards.sort() # 对牌集进行排序
    features = []
    for channel in range(channels):
        # #遍历所有的通道数，对于每个通道，将手牌中的每一种牌将feature中的该牌的相应位置置为1，然后在cards中去掉该牌
        # #最终返回channnels * 4 * 9
        # 去重，编写为channels * 4 * 9的样式
        S = set(cards)  # 将手牌先去重，每次从这个集合中拿一张牌去编码，之后在原来的手牌中去掉这张牌
        feature = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        for card in S:
            card_index = tool.translate3(card)  # 将牌转换成 0~33
            cards.remove(card) # 移除当前牌
            if card_index < 9: # 当前牌的索引为 0~9
                feature[0][card_index] = 1 # 将feature中card对应的位置(0~33) 置为1
            elif card_index < 18:
                feature[1][card_index - 9] = 1
            elif card_index < 27:
                feature[2][card_index - 18] = 1
            elif card_index < 36:
                feature[3][card_index - 27] = 1
        features.append(feature)
    return features

def calculate_feature(handCards0, players_eat, players_pong, players_gang, king_card,
                             discards_seq, remain_cards, self_kings, players_fei_kings, ):
    '''
    :param handCards0: 高手玩家的手牌 4 *　feature
    :param players_eat: 所有玩家吃的牌 4 *　feature * 4
    :param players_pong: 所有玩家碰的牌 4 *　feature
    :param players_gang: 所有玩家杠的牌 4 *　feature
    :param king_card: 宝牌 1 * feature
    :param discards_seq: 所有玩家丢的牌 4 *　feature * 4
    :param remain_cards: 剩余的牌 4 *　feature
    :param self_kings: 当前玩家的宝牌数 4 * feature
    :param players_fei_kings: 所有玩家的飞宝 4 * 4 *　feature
    :return:
    '''
    # 特征形式
    feature = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # 所有特征
    features = []
    # 手牌特征 4 feature
    handcards_features = card_feature_encode(handCards0, 4) # 高手玩家手牌的特征 是一个 4 * 9 * 4
    features.extend(handcards_features)

    # has do 所有玩家吃牌的特征  16 feature
    eat_features = []
    for player_eat in players_eat:
        eat_feature = [] # 单个玩家吃牌的特征
        for eat in player_eat:
            eat_feature.extend(card_feature_encode(eat, 1))
        pad_len = 4 - len(player_eat)
        for i in range(pad_len):
            eat_feature.append(feature)
        eat_features.extend(eat_feature)
    features.extend(eat_features)

    # has do 所有玩家碰牌的特征 4 feature
    pong_features = []
    for pong in players_pong:
        pong_feature = card_feature_encode(pong,1)
        pong_features.extend(pong_feature)
    features.extend(pong_features)

    # has do 所有玩家杠牌的特征 4 feature
    gang_features = []
    for gang in players_gang:
        gang_feature = card_feature_encode(gang,1)
        gang_features.extend(gang_feature)
    features.extend(gang_features)

    # has do 宝牌的特征 1 feature
    king_feature = card_feature_encode(king_card, 1)
    features.extend(king_feature)

    # has do 所有玩家丢的牌的特征 16 feature
    dis_feature = []
    for dis in discards_seq:
        dis_feature.extend(card_feature_encode(dis, 4))
    features.extend(dis_feature)

    # has do 未出现牌（剩余牌）的特征 4 feature
    remain_feature = card_feature_encode(remain_cards, 4)
    features.extend(remain_feature)

    # has do 当前玩家的宝牌的特征 4 feature
    self_king_feature = card_feature_encode(self_kings, 4)
    features.extend(self_king_feature)

    # has do 所有玩家飞宝的特征 4 * 4 feature
    fei_king_feature = []
    for fei_king in players_fei_kings:
        fei_king_feature.extend(card_feature_encode(fei_king, 4))
    features.extend(fei_king_feature)


    # return torch.tensor(features, dtype=torch.float).reshape(418, 34, 1)
    return features


def card_preprocess(handCards0, king_card, discards_seq, discards, self_king_num, fei_king_nums, fulu):
    players_eat, players_pong, players_gang = split_eat_pong_gang(fulu)
    players_fei_king = get_players_fei_king(fei_king_nums, king_card)
    self_kings = get_self_King(self_king_num, king_card)
    remain_cards = get_remain_card(fulu, handCards0, discards)
    features = calculate_feature(handCards0, players_eat, players_pong, players_gang, king_card,
                                 discards_seq, remain_cards, self_kings, players_fei_king)
    return torch.tensor(features, dtype=torch.float).reshape(69, 4, 9)

if __name__ == '__main__':
    # 特征形式
    # feature = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #            [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    handCards0 = [23,33,34,35]
    # players_eat = []
    # players_pong = []
    # players_gang = []
    king_card = 36
    discards_seq = [
        [
          52,
          50,
          51,
          5,
          37,
          2,
          1,
          9,
          17,
          5,
          55
        ],
        [
          55,
          50,
          54,
          49,
          49,
          9,
          21,
          49,
          17,
          41,
          39
        ],
        [
          55,
          17,
          34,
          49,
          3,
          20,
          40,
          19,
          21,
          2,
          53
        ],
        [
          53,
          34,
          6,
          25,
          4,
          40,
          54,
          8,
          54,
          54
        ]
      ]
    discards = [
        [
          52,
          50,
          51,
          37,
          2,
          9,
          17,
          5,
          55
        ],
        [
          55,
          50,
          54,
          49,
          49,
          9,
          21,
          49,
          17,
          41
        ],
        [
          55,
          17,
          34,
          49,
          40,
          21,
          2,
          53
        ],
        [
          53,
          34,
          4,
          40,
          54,
          54,
          54
        ]
      ] # 34
    self_king_num = 2
    fei_king_nums = [0, 0, 1, 0]
    fulu = [
        [
          [
            4,
            5,
            6
          ],
          [
            25,
            25,
            25
          ],
          [
            19,
            19,
            19
          ]
        ],
        [
          [
            1,
            1,
            1
          ],
          [
            8,
            8,
            8
          ],
          [
            4,
            5,
            6
          ]
        ],
        [
          [
            39,
            39,
            39
          ]
        ],
        [
          [
            3,
            3,
            3
          ],
          [
            18,
            19,
            20
          ]
        ]
      ] # 27

    features = card_preprocess(handCards0,king_card,discards_seq,discards,self_king_num,fei_king_nums,fulu)

    print(features)

