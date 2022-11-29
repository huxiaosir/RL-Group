# -*- coding:utf-8 -*-
from __future__ import print_function
from mah_tool.so_lib import lib_MJ as MJ
import random
import copy
import time
import numpy as np

# 全局变量
# 向听数随轮数的分布表

# 宝牌当宝还原的统计数据
# xts_round_ph = [[   0,    3,  159, 1417, 3967, 3333, 1035,   84,    2],
#                 [   0,   35,  587, 2848, 4131, 1953,  419,   27,    0],
#                 [   0,  168, 1430, 3922, 3324, 1009,  137,   10,    0],
#                 [   0,  456, 2541, 4243, 2237,  471,   47,    5,    0],
#                 [   0,  966, 3601, 3842, 1361,  207,   18,    5,    0],
#                 [   0, 1670, 4191, 3255,  787,   88,    8,    1,    0],
#                 [   0, 2440, 4538, 2529,  451,   40,    2,    0,    0],
#                 [   0, 3123, 4638, 1962,  262,   13,    2,    0,    0],
#                 [   0, 3691, 4655, 1491,  153,    9,    1,    0,    0],
#                 [   0, 4270, 4493, 1143,   88,    6,    0,    0,    0],
#                 [   0, 4710, 4375,  849,   63,    3,    0,    0,    0],
#                 [   0, 5090, 4201,  661,   46,    2,    0,    0,    0],
#                 [   0, 5351, 4071,  552,   24,    2,    0,    0,    0],
#                 [   0, 5591, 3771,  434,   18,    0,    0,    0,    0]
#                 ]
xts_round_ph = [[   0,    3,   90, 1053, 3384, 3531, 1595,  308,   36,    0],
                [   0,   17,  368, 2141, 3793, 2193, 1014,  408,   61,    5],
                [   0,   89,  890, 3151, 3330, 1240,  669,  506,  117,    8],
                [   0,  233, 1690, 3668, 2493,  694,  492,  526,  182,   22],
                [   0,  524, 2487, 3682, 1684,  432,  394,  498,  260,   39],
                [   0,  959, 3181, 3268, 1133,  310,  310,  483,  290,   66],
                [   0, 1471, 3594, 2786,  773,  243,  271,  454,  309,   99],
                [   0, 2052, 3759, 2309,  575,  224,  225,  395,  329,  132],
                [   0, 2495, 3934, 1889,  409,  213,  227,  329,  328,  176],
                [   0, 3042, 3907, 1488,  336,  210,  214,  261,  316,  226],
                [   0, 3419, 3916, 1216,  264,  211,  193,  212,  303,  266],
                [   0, 3855, 3759, 1006,  241,  210,  203,  183,  266,  277],
                [   0, 4225, 3508,  855,  222,  222,  176,  160,  212,  302],
                [   0, 3345, 2552,  540,  168,  154,  119,  108,  134,  227]
                ]

xts_round_qd = [[   0,   27,  553, 2008, 3434, 2761, 1075,  142],
                [   0,  171, 1128, 2770, 3212, 2073,  591,   55],
                [   0,  429, 1889, 3124, 2836, 1346,  354,   22],
                [   0,  762, 2637, 3338, 2205,  865,  185,    8],
                [   0, 1155, 3415, 3115, 1687,  533,   90,    5],
                [   0, 1592, 4001, 2830, 1207,  326,   42,    2],
                [   0, 2008, 4518, 2389,  864,  202,   18,    1],
                [   0, 2518, 4761, 1991,  623,   93,   14,    0],
                [   0, 2827, 5089, 1616,  406,   55,    7,    0],
                [   0, 3191, 5198, 1313,  262,   32,    4,    0],
                [   0, 3558, 5214, 1037,  161,   27,    3,    0],
                [   0, 3762, 5306,  799,  115,   17,    1,    0],
                [   0, 3729, 4591,  570,   71,   12,    0,    0]
                ]
xts_round_ssl = [
                [   0,   26,  439, 2271, 4135, 2558,  541,   29],
                [   0,  141, 1196, 3509, 3754, 1265,  128,    7],
                [   0,  458, 2202, 3998, 2722,  583,   36,    1],
                [   0, 1077, 3076, 3806, 1768,  260,   12,    1],
                [   0, 1858, 3709, 3263, 1053,  107,   10,    0],
                [   0, 2819, 3964, 2543,  620,   48,    6,    0],
                [   0, 3796, 3933, 1858,  392,   18,    3,    0],
                [   0, 4669, 3752, 1347,  219,   12,    1,    0],
                [   0, 5542, 3359,  964,  131,    4,    0,    0],
                [   0, 6180, 3018,  713,   89,    0,    0,    0],
                [   0, 6751, 2732,  471,   46,    0,    0,    0],
                [   0, 7200, 2426,  346,   28,    0,    0,    0],
                [   0, 7533, 2177,  267,   23,    0,    0,    0]
                ]
xts_round_jy = [[0, 3, 13, 150, 751, 2098, 3206, 2512, 994, 248, 23, 2],
                [0, 9, 104, 515, 1597, 2833, 2862, 1549, 457, 67, 6, 1],
                [0, 63, 368, 1103, 2336, 2959, 2084, 847, 213, 25, 2, 0],
                [0, 243, 778, 1802, 2751, 2498, 1347, 465, 107, 8, 1, 0],
                [0, 532, 1376, 2394, 2665, 1910,  821,  250, 48, 3, 1, 0],
                [0, 1053, 1959, 2652, 2306, 1365, 501, 142, 21, 0, 1, 0],
                [0, 1708, 2423, 2658, 1881, 923, 316, 84, 6, 1, 0, 0],
                [0, 2425, 2783, 2427, 1517, 596, 205, 43, 3, 1, 0, 0],
                [0, 3163, 2947, 2200, 1092, 443, 131, 21, 3, 0, 0, 0],
                [0, 3842, 2978, 1915, 845, 320, 81, 16, 3, 0, 0, 0],
                [0, 4385, 2990, 1626, 705, 228, 53, 11, 2, 0, 0, 0],
                [0, 4995, 2841, 1412, 553, 159, 34, 6, 0, 0, 0, 0],
                [0, 5424, 2739, 1259, 449, 104, 23, 2, 0, 0, 0, 0]
                ]
xts_round = [xts_round_ph,xts_round_qd,xts_round_ssl,xts_round_jy]


def get_t2info():
    """
    T2分配度算法

    :return:
    """
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
        cardSet.append(int(i // 9) * 16 + i % 9 + 1)
    for card in cardSet:
        efc_dzindex[card] = []
        efc_dzindex[card].append(MJ.translate16_33(card))
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
                efc_dzindex[card].append(33 + 24 + color * 7 + (card & 0x0f) - 1)
                efc_dzindex[card].append(33 + color * 8 + (card & 0x0f) + 1)

    return dzSet, dzEfc, efc_dzindex

def get_t3info():
    """
    T3的信息，
    :return: T3的集合，包括了刻子和顺子
    """
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

# 烂牌牌组L2 
def get_L2():
    L2 = []
    L2_need = []
    # L2
    for i in range(1,10):
        for j in range(4, 10):
            if j - i >= 3:
                L2.append([i,j])
    cards = [1,2,3,4,5,6,7,8,9]
    _L3 = [[1,4,7],[1,4,8],[1,4,9],[1,5,8],[1,5,9],[1,6,9],[2,5,8],[2,5,9],[2,6,9],[3,6,9]]
    # L2_need
    for L2_cards in L2:
        _need = []
        _L2_cards = copy.copy(L2_cards)
        for card in cards:
            _L2_cards.append(card)
            _L2_cards.sort()
            # print('_L2_cards:',_L2_cards)
            if _L2_cards in _L3:
                _need.append(card)
            _L2_cards.remove(card)
        L2_need.append(_need)
    return L2, L2_need
L2, L2_need = get_L2()

# 防守模型
class DefendModel:
    def __init__(self, cards, suits, king_card, fei_king, discards, discardsOp, discardsReal, round, seat_id, xts_round, M):
        """
        对手建模类变量初始化
        :param cards: 手牌
        :param suits: 副露
        :param discards: 所有弃牌
        :param discardsOp: 所有副露
        :param discardsReal:实际弃牌
        :param round: 轮数
        :param seat_id: 玩家id
        :param Txts_transpose: 向听数转换表
        :param M: 模拟次数
        """
        self.cards = cards
        self.suits = suits
        self.round = round-1 if round <= 13 else 12
        self.seat_id = seat_id
        self.discards0 = discardsReal[seat_id]
        self.king_card = king_card
        otherID = self.getOtherID()
        self.discards1 = discardsReal[otherID[0]]
        self.discards2 = discardsReal[otherID[1]]
        self.discards3 = discardsReal[otherID[2]]
        self.discardsOp0 = discardsOp[seat_id]
        self.discardsOp1 = discardsOp[otherID[0]]
        self.discardsOp2 = discardsOp[otherID[1]]
        self.discardsOp3 = discardsOp[otherID[2]]
        self.leftNum, _ = MJ.trandfer_discards(discards, discardsOp, cards)
        self.xts_round = xts_round
        self.M = M
        self.remain_king_num, self.fei_king0, self.fei_king1, self.fei_king2, self.fei_king3 = self.get_remain_king_num()

    def getOtherID(self):
        """
        获取其他玩家的id
        :return:
        """
        if self.seat_id == 0:
            return [1, 2, 3]
        elif self.seat_id == 1:
            return [2, 3, 0]
        elif self.seat_id == 2:
            return [3, 0, 1]
        elif self.seat_id == 3:
            return [0, 1, 2]
        else:
            print('seat_id Error!', self.seat_id)
            return []

    def get_remain_king_num(self):
        fei_king0 = 0
        remain_king_num = 4
        fei_king1 = 0
        fei_king2 = 0
        fei_king3 = 0

        # 自己手牌中的king_card个数
        king_card_in_hand = 0
        for card in self.cards:
            if card == self.king_card:
                king_card_in_hand += 1

        # 检查弃牌中的宝牌
        for card in self.discards0:
            if card == self.king_card:
                fei_king0 += 1
        for card in self.discards1:
            if card == self.king_card:
                fei_king1 += 1
        for card in self.discards2:
            if card == self.king_card:
                fei_king2 += 1
        for card in self.discards3:
            if card == self.king_card:
                fei_king3 += 1
        # 检查副露的宝牌
        for fulu in self.discardsOp0:
            if fulu[0] == fulu[1] and fulu[0] == self.king_card: # 刻子数
                remain_king_num -= 2
        for fulu in self.discardsOp1:
            if fulu[0] == fulu[1] and fulu[0] == self.king_card: # 刻子数
                remain_king_num -= 2
        for fulu in self.discardsOp2:
            if fulu[0] == fulu[1] and fulu[0] == self.king_card: # 刻子数
                remain_king_num -= 2
        for fulu in self.discardsOp3:
            if fulu[0] == fulu[1] and fulu[0] == self.king_card: # 刻子数
                remain_king_num -= 2


        remain_king_num = remain_king_num - fei_king0 - fei_king1 - fei_king2 - fei_king3 - king_card_in_hand
        return remain_king_num, fei_king0, fei_king1, fei_king2, fei_king3

    def simulate_ph(self, suits, discards_real, wall, sum_xts, fei_king_num):  # 模拟平胡

        # 统计频率
        ''' xts_zd =[[0],
                        [25464, 11705],  # xt = 1   # [3, 2] 中的t2至少包含一个aa
                        [21388, 5498, 45828],  # xt = 2 # [3, 2]中的t2不包含aa ,[2, 3]中的t2至少包含一个aa
                        [566, 13879, 4326, 29809, 4399], # xt = 3 # [2, 3] 中t2不包含aa
                        [1742, 14499,  1681, 140, 13100, 2252],# xt=4  # [1, 4] 中t2 不包含aa [1,5]中t2不包含aa, [0, 5] t2 至少包含一个aa,[0, 6] t2 至少包含一个aa'
                        [110, 3331, 9916, 355, 0],# xt=5 # [0, 5]中t2不包含aa [0, 6]中t2不包含aa
                        [327, 2653],# xt = 6
                        [16, 237], # xt = 7
                        [0], # xt = 8
                        [0], # xt = 9
                        ]
        '''
        # 对应上面频率
        # xts_zd = [[],
        #         [[3, 2], [4, 0]], # xt = 1   # [3, 2] 中的t2至少包含一个aa
        #         [[3, 1], [3, 2], [2, 3]], # xt = 2 # [3, 2]中的t2不包含aa ,[2, 3]中的t2至少包含一个aa
        #         [[3, 0], [2, 2], [2, 3], [1, 4], [1, 5]], # xt = 3 # [2, 3] 中t2不包含aa， [1, 4] 中t2 至少包含一个aa [1,5]中t2至少包含一个aa
        #         [[2, 1], [1, 3], [1, 4], [1, 5], [0, 5], [0, 6]],  # xt=4  # [1, 4] 中t2 不包含aa [1,5]中t2不包含aa, [0, 5] t2 至少包含一个aa,[0, 6] t2 至少包含一个aa'
        #         [[2, 0], [1, 2], [0, 4], [0, 5], [0, 6]], # xt=5 # [0, 5]中t2不包含aa [0, 6]中t2不包含aa
        #         [[1, 1], [0, 3]],  # xt = 6
        #         [[1, 0], [0, 2]],  # xt = 7
        #         [[0, 1]], # xt = 8
        #         [[0, 0]] # xt = 9
        #         ]

        # 例 [3, [1, 1]] 3表示3个t3 [1, 2]表示2个t2中至少1个aa
        #                           [-1, 1]表示1个t2，且不包含aa
        #                           [0, 2] 表示2个t2，且无限制
        xts_zd = [[],
                [[3, [1, 2]], [4, [0, 0]]], # xt = 1   # [3, 2] 中的t2至少包含一个aa
                [[3, [0, 1]], [3, [-1, 2]], [2, [1, 3]]], # xt = 2 # [3, 2]中的t2不包含aa ,[2, 3]中的t2至少包含一个aa
                [[3, [0, 0]], [2, [0, 2]], [2, [-1, 3]], [1, [1, 4]], [1, [1, 5]]], # xt = 3 # [2, 3] 中t2不包含aa， [1, 4] 中t2 至少包含一个aa [1,5]中t2至少包含一个aa
                [[2, [0, 1]], [1, [0, 3]], [1, [-1, 4]], [1, [-1, 5]], [0, [1, 5]], [0, [1, 6]]],  # xt=4  # [1, 4] 中t2 不包含aa [1,5]中t2不包含aa, [0, 5] t2 至少包含一个aa,[0, 6] t2 至少包含一个aa'
                [[2, [0, 0]], [1, [0, 2]], [0, [0, 4]], [0, [-1, 5]], [0, [-1, 6]]], # xt=5 # [0, 5]中t2不包含aa [0, 6]中t2不包含aa
                [[1, [0, 1]], [0, [0, 3]]],  # xt = 6
                [[1, [0, 0]], [0, [0, 2]]],  # xt = 7
                [[0, [0, 1]]], # xt = 8
                [[0, [0, 0]]] # xt = 9
                ]

        # 对应的概率 经验值
        # 宝做宝还原得出的频率
        # xts_zd_prob = [[0],
        #     [0.6850870348946703, 0.31491296510532973],
        #     [0.2941386803091564, 0.07561129906207883, 0.6302500206287648],
        #     [0.010683478359349931, 0.2619717246456143, 0.08165499537552615, 0.5626569017912758, 0.08303289982823382],
        #     [0.05213383611659783, 0.43391991380858325, 0.050308254025258875, 0.004189860537499252, 0.39205123600885855, 0.06739689950320225],
        #     [0.008022170361726954, 0.24292590431738623, 0.7231621936989499, 0.02588973162193699, 0.0],
        #     [0.10973154362416107, 0.890268456375839],
        #     [0.06324110671936758, 0.9367588932806324],
        #     [0],
        #     [0]]

        xts_zd_prob = [[1],
            [0.6989872872225813, 0.30101271277741865],
            [0.17033538434873038, 0.11448132420487044, 0.7151832914463991],
            [0.008259269234222103, 0.22223718466648712, 0.07078732381721878, 0.6133854026393751, 0.08533081964269683],
            [0.05276169409927786, 0.4107084770021469, 0.04033569709192635, 0.0038383969813284755, 0.4330232255546158, 0.059332509270704575],
            [0.01117952041477641, 0.24773169151004537, 0.7158133506156837, 0.02235904082955282, 0.002916396629941672],
            [0.1426459719142646, 0.8573540280857354],
            [0.06363636363636363, 0.9363636363636364],
            [1.0],
            [1.0]
        ]

        # 宝牌数量比
        # king_nums_ratio = [[0, 0, 0, 0, 0],
        #  [0.9497012359826471, 0.0487026274862896, 0.001432430220184988, 0.00016370631087828436, 0.0],
        #  [0.8996920198042961, 0.0834275466843398, 0.015243070445596663, 0.0016373630657674164, 0.0],
        #  [0.7886251505380522, 0.1726040169379589, 0.0364010722194165, 0.002369760304572472, 0.0],
        #  [0.7291296173974899, 0.23520146334027742, 0.03455108988364412, 0.0011178293785884864, 0.0],
        #  [0.7845430278378035, 0.19219524596415405, 0.022880386424304056, 0.0003813397737384009, 0.0],
        #  [0.8634333120612636, 0.1301850670070198, 0.006381620931716656, 0.0, 0.0],
        #  [0.9482758620689655, 0.05172413793103448, 0.0, 0.0, 0.0],
        #  [1.0, 0.0, 0.0, 0.0, 0.0],
        #  [1.0, 0, 0, 0, 0]]

        # shape  10 * 5 * 5 # 10 是向听数 5 是飞宝数 5 是宝牌的个数
        # 根据飞宝数调整宝牌数
        king_ratios_prob = [
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0.3626725082146769, 0.4115516611902154, 0.203986856516977, 0.021511500547645127, 0.0002774735304855787],
             [0.8261474269819193, 0.0976356050069541, 0.0717663421418637, 0.004450625869262865, 0.0],
             [0.8975609756097561, 0.06829268292682927, 0.03414634146341464, 0.0, 0.0],
             [0.6842105263157895, 0.3157894736842105, 0.0, 0.0, 0.0],
             [1, 0, 0, 0, 0]],
            [[0.4290052755796196, 0.4243023740108288, 0.13810217964736915, 0.008520755240871859, 6.941552131056504e-05],
             [0.7142857142857143, 0.2239858906525573, 0.06172839506172839, 0.0, 0.0],
             [0.7058823529411765, 0.29411764705882354, 0.0, 0.0, 0.0],
             [0.25, 0.75, 0.0, 0.0, 0.0],
             [1, 0, 0, 0, 0]],
            [[0.46769794786962937, 0.42504582644073857, 0.10233826619573479, 0.004917959493897259, 0.0],
             [0.825136612021858, 0.11475409836065574, 0.060109289617486336, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]],
            [[0.5322236919459141, 0.4069664902998236, 0.059082892416225746, 0.0017269253380364491, 0.0],
             [0.8194444444444444, 0.1527777777777778, 0.027777777777777776, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]],
            [[0.6767853243066172, 0.2875081895610395, 0.035269709543568464, 0.00043677658877484165, 0.0],
             [0.72, 0.24, 0.04, 0.0, 0.0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]],
            [[0.7957622130665097, 0.19364331959976458, 0.01059446733372572, 0.0, 0.0],
             [0.6666666666666666, 0.3333333333333333, 0.0, 0.0, 0.0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]],
            [[0.9186991869918699, 0.08130081300813008, 0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]]]

        def reg_prob(probability_relative_list):  #
            '''
            对输入的概率进行正则化，等比变化使和为1
            :param probability_relative_list:  相对概率
            :return:绝对概率，概率值相加为1
            '''
            probability_relative_list = np.array(probability_relative_list)
            total_prob = sum(probability_relative_list)  # 当前概率和
            if total_prob == 0:
                return probability_relative_list
            probability_abs_list = [probability_relative_list[index] / total_prob for index
                                    in range(len(probability_relative_list))]
            return probability_abs_list

        def simulate_xts_king_num(sum_xts, suit_len, king_num_in_wall, fei_king_num):
            # 模拟得出向听数和宝牌数
            # 宝只能作宝调
            xts_max = 9 - 2 * suit_len # 向听数的最大值 
            if xts_max < 0 or xts_max > 9:
                print(xts_max)
            r = random.uniform(0, sum_xts[xts_max])
            xts = 0
            # 获取随机得到的向听数
            for i in range(1, len(sum_xts)):
                if r <= sum_xts[i]:
                    xts = i
                    break
            # 根据飞宝数和当前轮数来调整xts
            if fei_king_num > 0 and self.round > 9:
                xts = 1
            king_prob = king_ratios_prob[xts][fei_king_num]
            # king_prob = king_nums_ratio[xts]

            temp_king_prob = king_prob[0:(king_num_in_wall+1)]

            # print(king_num_in_wall, temp_king_prob)

            temp_king_prob = reg_prob(temp_king_prob)
            # print(wall)
            king_in_hand_num = np.random.choice(len(temp_king_prob), 1, p=temp_king_prob)[0]

            return xts, king_in_hand_num, r # , reduction_flag

        # , king_in_hand_num， is_king_reduction
        def rand_N32(xts_zd, xts_zd_prob, xts, suit_len, r, discards_real):
            """
            随机出本次模拟的T3与T2的数量
            :param xts_zd: xts_zd的组合
            :param xts_zd_prob: 对应组合的概率
            :param sum_xts: 该轮向听数的累加表
            :param suit_len: 已有的副露数量
            :return: 随机得到的T3与T2的数量
            """
            xtSet = copy.deepcopy(xts_zd[xts])
            # cur_SetSize = len(xtSet)
            xtSet_prob = copy.deepcopy(xts_zd_prob[xts])
            for index, x in zip(range(len(xtSet)), xtSet):
                if x[0] < suit_len:
                    xtSet_prob[index] = 0
                    #  xtSet.remove(x)

                # 若上一次丢牌是飞宝，则手牌里一定有aa，将无t2的t3_t2组合概率置为0
                if len(discards_real) != 0 and discards_real[-1] == self.king_card and (x[1][1] == 0 or x[1][0] == -1):
                    xtSet_prob[index] = 0


            # 处理特殊情况， 当副露个数>组合中的t3个数时
            if sum(xtSet_prob) == 0:
                # 找出离r随机值最近的向听数的sum值
                if xts == 0:
                    pre_sum_value = 0
                else:
                    pre_sum_value = sum_xts[xts-1]
                if xts == len(sum_xts)-1:
                    next_sum_value = sum_xts[-1]
                else:
                    next_sum_value = sum_xts[xts+1]
                # new_xtSet_prob = [] # 新的t3t2分配概率表

                # pre_flag = False  # 向听数减少标志位flag
                if r - pre_sum_value <= next_sum_value - r:
                    if xts == 1:
                        new_xt = 1
                    else:
                        # pre_flag = True
                        new_xt = xts-1
                else:
                    if xts == len(sum_xts)-1:
                        new_xt = xts
                    else:
                        new_xt = xts + 1

                xtSet_prob = copy.deepcopy(xts_zd_prob[new_xt])
                xtSet = xts_zd[new_xt]

                for index, x in zip(range(len(xtSet)), xtSet):
                    if x[0] < suit_len:
                        xtSet_prob[index] = 0
                #如果还未符合条件
                # if sum(xtSet_prob) == 0:
                #     if pre_flag: new_xt -= 1 # 向听数再减去1
                #     else: new_xt -= 2
                #     xtSet = copy.deepcopy(xts_zd[new_xt])
                #     xtSet_prob = copy.deepcopy(xts_zd_prob[new_xt])
                #     for index, x in zip(range(len(xtSet)), xtSet):
                #         if x[0] < suit_len:
                #             xtSet_prob[index] = 0

            if sum(xtSet_prob) == 0:
                return [suit_len, [0, 0]]
            # np.random.choice(range(len(prob)), 1, p=prob)[0]
            prob = self.reg_prob(xtSet_prob)

            N32 = xtSet[np.random.choice(len(xtSet), 1, p=prob)[0]]

            return N32

        # 获取模拟T3后的wall
        def get_wall(P1_N32, suits, wall_, t3Set, king_card):
            """
            T3模拟算法
            由于T3属于固定牌，所有玩家不作区分处理，这里至直接模拟出所有玩家的T3，并生成模拟后的牌墙
            :param P1_N32: 玩家1的T3与T2数量
            :param wall_: 分配完后的牌墙
            :param t3Set: t3的集合
            :return: 玩家的T3模拟后的牌墙
            """

            def get_distributionTable(wall):
                """
                牌墙T3分配度计算方法 包括27+7种刻子，21种顺子
                :param wall: 牌墙的牌数量 1*34
                :return: T3分配度
                """
                t = [0] * (34 + 7 * 3)  # 34种刻子，21种顺子
                # 花色牌，万条筒27种
                for i in range(27):
                    if wall[i] >= 3:
                        # t[i] = 2
                        # t[i]=0
                        t[i] = float(wall[i])/2 #todo 有待商榷
                        if i % 9 + 1 == 1:  # 一万、条、筒
                            t[i] += 2
                            if min(wall[i], wall[i + 1], wall[i + 2]) == 0:
                                t[i] += 4
                            # else:
                            # t[i]+=3.0/(1+min(wall[i],wall[i+1],wall[i+2]))
                        elif i % 9 + 1 == 2:  # 二万、条、筒
                            t[i] += 1
                            if min([wall[i - 1], wall[i], wall[i + 1]]) + min([wall[i], wall[i + 1], wall[i + 2]]) == 0:
                                t[i] += 5
                            # t[i]+=3.0/(1+min([wall[i-1],wall[i],wall[i+1]])+min([wall[i],wall[i+1],wall[i+2]]))
                        elif i % 9 + 1 == 8:  # 八万、条、筒
                            t[i] += 1
                            if min([wall[i - 2], wall[i - 1], wall[i]]) + min([wall[i - 1], wall[i], wall[i + 1]]) == 0:
                                # t[i]+=3.0/(1+min([wall[i-2],wall[i-1],wall[i]])+min([wall[i-1],wall[i],wall[i+1]]))
                                t[i] += 5
                        elif i % 9 + 1 == 9:  # 九万、条、筒
                            t[i] += 2
                            if min([wall[i - 2], wall[i - 1], wall[i]]) == 0:
                                # t[i]+=3.0/(1+min([wall[i-2],wall[i-1],wall[i]]))
                                t[i] += 4
                        else:
                            if min([wall[i - 2], wall[i - 1], wall[i]]) + min(
                                    [wall[i - 1], wall[i], wall[i + 1]]) + min(
                                    [wall[i], wall[i + 1], wall[i + 2]]) == 0:
                                # t[i]+=3.0/(1+(min([wall[i-2],wall[i-1],wall[i]])+min([wall[i-1],wall[i],wall[i+1]])+min([wall[i],wall[i+1],wall[i+2]])))
                                t[i] += 6
                # 字牌 7种
                for i in range(27, 34):
                    if wall[i] > 3:
                        t[i] = 8
                #
                for i in range(34, 34 + 21):
                    index = MJ.translate16_33(int((i - 34) / 7) * 16 + (i - 34) % 7 + 1)
                    # if wall[index]==0 or wall[index+1]==0 or wall[index+2]==0:
                    #     t[i]=0
                    # c3 = [wall[index], wall[index + 1], wall[index + 2]]
                    # key = c3[0] * 100 + c3[1] * 10 + c3[2]
                    # t[i] = self.c3_d[key]
                    t[i] = min([wall[index], wall[index + 1], wall[index + 2]])

                return t

            wall = copy.copy(wall_)
            wall[MJ.convert_hex2index(king_card)] = 0  #将牌墙中的宝牌置为0
            # wall[MJ.convert_hex2index(self.king_card)] = 0  #将牌墙中的宝牌置为0
            t3_handacards = []
            for i in range(P1_N32[0] - len(suits)):
                # 生成T3分配度表  # 刻子顺子的分布度
                t3 = get_distributionTable(wall)
                t3_sum = copy.copy(t3)  # 累加表
                # 随机出要分配的T3
                for i in range(1, len(t3_sum)):
                    t3_sum[i] = t3_sum[i - 1] + t3_sum[i]
                if t3_sum[-1] == 0:
                    print("failed to simulate T3")
                    return [], wall_
                # 随机数
                r = random.uniform(0, t3_sum[-1])
                j = 0
                flag = False
                # 找到随机的T3的下标
                while j < len(t3_sum) and not flag:
                    if r <= t3_sum[j]:  # 下标
                        # index_set.append(i)
                        for card in t3Set[j]:
                            t3_handacards.append(card)
                            wall[MJ.convert_hex2index(card)] -= 1
                            flag = True
                    j += 1
                # print(t3_handacards)
            return t3_handacards, wall

        def simulate_t2(dzSet, dzEfc, efc_dzindex, N2, discards, wall):
            """
            模拟T2
            :param dzSet: 搭子的集合
            :param dzEfc: 搭子的有效牌
            :param efc_dzindex:有效牌下标
            :param N2: T2的数量
            :param discards: 弃牌
            :param wall: 牌墙
            :return: 返回t2的手牌，t2搭子下标， 危险度RT和wall
            """

            t2_handcards = []
            dz_set_indexs = [] # 记录搭子的下标

            P = [0] * (34 + 15 * 3)  # 根据已出牌，不再分配有该弃牌为有效牌的搭子组合
            for card in discards:
                for i in efc_dzindex[card]:
                    P[i] += 1
            RT = [[0] * 34, [0] * 34]  # 危险度表

            for y in range(N2[1]):
                Pdz_index = []  # 可分配搭子
                t2 = [0] * (34 + (15 * 3))  # 计算搭子的分配度表
                for i in range(len(wall)):
                    if N2[0] != -1:  # 不存在aa的情况
                        if wall[i] >= 2:
                            t2[i] = float(wall[i])/2
                    # elif wall[i]==4:
                    #     t2[i]= 4
                    # t2[i]=int(wall[i]/2)
                    # if wall[i] >= 2:
                    #     t2[i] = float(wall[i])/2 #todo 有待商榷
                    if y == 0 and N2[0] == 1:  # N2[0] == 1 一定要模拟到aa
                        continue

                    if i < 27:
                        color = int(i / 9)
                        if i % 9 + 1 == 8 or i % 9 + 1 == 1:  # 89 or 12
                            if sum([wall[MJ.convert_hex2index(e)] for e in
                                    dzEfc[33 + color * 8 + (i % 9) + 1]]) != 0:
                                t2[33 + color * 8 + (i % 9) + 1] = min(wall[i], wall[i + 1]) * 0.7
                                # t2[33 + color * 15 + (i % 9) + 1] = 1
                        elif i % 9 + 1 == 9:  # 9 不处理
                            pass
                        else:
                            if sum([wall[MJ.convert_hex2index(e)] for e in
                                    dzEfc[33 + color * 8 + (i % 9) + 1]]) != 0:
                                t2[33 + color * 8 + (i % 9) + 1] = min(wall[i], wall[i + 1]) * 0.8
                                # t2[33 + color * 15 + (i % 9) + 1] = 1

                            if sum([wall[MJ.convert_hex2index(e)] for e in
                                    dzEfc[33 + 24 + color * 7 + (i % 9) + 1]]) != 0:
                                t2[33 + 24 + color * 7 + (i % 9) + 1] = min(wall[i], wall[i + 2]) * 0.7
                                # t2[33 + color * 15 + 8 + (i % 9) + 1] = 1
                # Pc=copy.copy(t2)
                # 移除t2中的不可分表P
                Pc = [0] * (34 + (15 * 3))
                for x in range(len(t2)):
                    if P[x] == 0:
                        Pc[x] = t2[x]
                    else:
                        Pc[x] = max(0, t2[x] - P[x] * 1.5)
                # 生成随机数
                for j in range(1, len(Pc)):
                    Pc[j] = Pc[j] + Pc[j - 1]  # 累加
                    # if P[x] == 0 and t2[x] != 0:
                    #     Pdz_index.append(x)
                # print 'dz_index',Pdz_index
                r = random.uniform(0, Pc[-1])
                # 分配搭子
                for x in range(len(Pc)):
                    if r <= Pc[x]:
                        # 更新wall
                        dz_index = x
                        dz_set_indexs.append(dz_index)
                        for card in dzSet[dz_index]:
                            t2_handcards.append(card)
                            wall[MJ.convert_hex2index(card)] -= 1
                        # 0-33aa
                        if dz_index < 34:
                            for card in dzEfc[dz_index]:
                                RT[0][MJ.convert_hex2index(card)] += 1
                        else:  # 34+是ab/ac
                            for card in dzEfc[dz_index]:
                                RT[1][MJ.convert_hex2index(card)] += 1
                        break
                # dz_index = random.choice(Pdz_index)

            return t2_handcards, dz_set_indexs, RT, wall

        def distrib_king_and_updata_wall(t3_and_t2_handcards, king_nums, king_card, wall):  #在分配宝牌之后更新牌墙
            # t3和t2的组合
            t3_handcards, t2_handcards = t3_and_t2_handcards

            index_t2 = -1  # 凑成t2被替换的下标
            if len(t2_handcards) > 0 and king_nums > 0:  # t2大于零，宝牌也大于零一定是有个宝牌凑成的
                index_t2 = np.random.choice(len(t2_handcards), 1)[0]
                replace_card = t2_handcards[index_t2]  # 替换的牌
                t2_handcards[index_t2] = king_card  # 替换成宝牌
                wall[MJ.convert_hex2index(replace_card)] += 1  # 把被替换的牌加到牌墙中
                index_t2 += len(t3_handcards)

            # 把所有牌合并
            handcards_with_king = t3_handcards
            handcards_with_king.extend(t2_handcards)

            handcards_len = len(handcards_with_king)

            if (king_nums - 1) > 0 and handcards_len > 0:
                # 考虑重复模拟到与上面重复的t2下标的情况，需要重新模拟
                while True:
                    choice_index = np.random.choice(handcards_len, king_nums - 1)
                    if index_t2 not in choice_index:  # 没有重复情况
                        for index in choice_index:
                            replace_card = handcards_with_king[index]
                            wall[MJ.convert_hex2index(replace_card)] += 1
                            handcards_with_king[index] = king_card
                        break
            return handcards_with_king, wall

        def simulate_single(t2_dzSet_indexs, need_simu_num, wall):  # 平胡的孤张模拟
            # handcards_copy = copy.deepcopy(handcards)
            single_handcards = []
            #  origin_xt = MJ.wait_type_comm(handcards, suits)# .get_xts()  # 初始向听数
            # 只要不会减少初始向听数的牌，都认为是孤张
            # 计算孤张分配表
            def get_distribution_comm_single(t2_dzSet_indexs, single_handcards, wall):
                '''
                计算平胡孤张的分配表
                :param t2_dzSet_indexs: 模拟的搭子下标
                :param single_handcards: 当前模拟的孤张
                :param wall: 当前牌墙
                :return:  1*34的孤张分配度, 1*34分配度的累加和
                '''
                # dzSet
                t_comm_s = copy.deepcopy(wall)
                for dzset_index in t2_dzSet_indexs:
                    dz_efc_cards = dzEfc[dzset_index]
                    for card in dz_efc_cards:
                        t_comm_s[MJ.convert_hex2index(card)] = 0  # 转换成索引,并把有效牌置为0

                for single_card in single_handcards:
                    efc_dz_indexs = efc_dzindex[single_card]  # 孤张对应的有效牌的下标
                    for efc_dz_index in efc_dz_indexs:
                        dz_cards = dzSet[efc_dz_index]
                        for card in dz_cards:
                            t_comm_s[MJ.convert_hex2index(card)] = 0  # 转换成索引,并把有效牌置为0

                t_comm_s_sum = copy.deepcopy(t_comm_s)
                for j in range(1, 34):
                    t_comm_s_sum[j] = t_comm_s_sum[j]+t_comm_s_sum[j-1]
                return t_comm_s, t_comm_s_sum

            for j in range(need_simu_num):
                # 当前平胡孤张分配度
                _, cur_t_comm_single_sum = get_distribution_comm_single(t2_dzSet_indexs, single_handcards, wall)
                # 根据孤张分配来模拟孤张
                r = random.uniform(0, cur_t_comm_single_sum[-1])
                for i in range(34):
                    if r <= cur_t_comm_single_sum[i]: # 选择当前牌
                        single_handcards.append(MJ.translate33_16(i))
                        wall[i] -= 1  # 牌墙牌减少
                        break

            return single_handcards, wall

        # 未出现的宝牌数
        king_num_in_wall = wall[MJ.convert_hex2index(self.king_card)]

        # 模拟得到向听数,宝牌数
        xts, king_in_hand_num, r = simulate_xts_king_num(sum_xts, len(suits), king_num_in_wall, fei_king_num)

        # 判断生成的手牌中的宝牌数是否符会大于剩下的宝牌数

        N32 = rand_N32(xts_zd, xts_zd_prob, xts, len(suits), r, discards_real)  # T3和T2的数量

        # t3Set=get_t3info()
        t3_handcards, wall = get_wall(N32, suits, wall, t3Set, self.king_card)  # 模拟t3

        # dzSet,dzEfc,efc_dzindex=get_t2info()
        t2_handcards, t2_dzSet_indexs, RT1, wall = simulate_t2(dzSet, dzEfc, efc_dzindex, N32[1], discards_real, wall)  # 返回危险度表， 牌墙

        t3_and_t2_handcards = [t3_handcards, t2_handcards]

        # 计算需要模拟的孤张数量
        simu_t3_with_t2_len = len(t3_handcards) + len(t2_handcards)
        need_single_num = 13 - simu_t3_with_t2_len - 3 * len(suits)
        single_handcards = []

        # 考虑特殊情况 当t2为0时，宝牌必不能使用，此时拿一张牌做孤张
        if len(t2_handcards) == 0 and king_in_hand_num > 0:
            need_single_num -= 1
            # king_in_hand_num -= 1  # 宝牌数减一
            single_handcards.append(self.king_card)

        single_handcards_, wall = simulate_single(t2_dzSet_indexs, need_single_num, wall)
        single_handcards.extend(single_handcards_)

        # if len(t3_handcards) == 0 and len(t2_handcards) == 0:
        #     print("No")


        # 把宝牌替换手牌中的t3或者t2  宝牌的分配算法
        handcards_with_king, wall = distrib_king_and_updata_wall(t3_and_t2_handcards, king_in_hand_num, self.king_card, wall)

        # if wall[MJ.convert_hex2index(self.king_card)] < 0:
        #     print(king_in_hand_num, king_num_in_wall)

        wall[MJ.convert_hex2index(self.king_card)] = max(king_num_in_wall - int(king_in_hand_num), 0)

        comm_handcards = handcards_with_king
        comm_handcards.extend(single_handcards)

        comm_handcards.sort()

        # 可视化摸到的手牌，方便debug，正式代码环境中可以关闭
        # comm_handcards = [MJ.translate16_10(card) for card in comm_handcards]
        a = comm_handcards, wall, RT1
        if len(a) != 3:
            input()
        return comm_handcards, wall, RT1

    def simulate_qd(self, suits, discard_real, wall, sum_xts):
        # print("init wall is {}".format(wall))
        # 删除某一向听数情况下不必要的刻子，搭子组合
        # xts:4-->[0,3,0]
        # xts:5-->[1, 2, 0],[0, 2, 0],[0, 2, 1]
        # xts:6-->[1, 1, 0], [1, 1, 1]，[0, 1, 0],[0, 1, 1], [0, 1, 2]
        # xts:7-->[2, 0, 0],[1, 0, 0], [1, 0, 1],[1, 0, 2],[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]
        xts_zd = [
            [0, 7, 0],
            [[0, 6, 0]],  # xts:1
            [[1, 5, 0], [0, 5, 0], [0, 5, 1]],  # xts:2
            [[1, 4, 0], [1, 4, 1], [0, 4, 0], [0, 4, 1], [0, 4, 2]],  # xts:3
            [[2, 3, 0], [1, 3, 0], [1, 3, 1], [1, 3, 2], [0, 3, 1], [0, 3, 2], [0, 3, 3]],  # xts；4
            [[3, 2, 0], [2, 2, 0], [2, 2, 1], [1, 2, 1], [1, 2, 2], [1, 2, 3],
             [0, 2, 2], [0, 2, 3],
             [0, 2, 4]],  # xts:5
            [[3, 1, 0], [3, 1, 1], [2, 1, 0], [2, 1, 1], [2, 1, 2], [1, 1, 2], [1, 1, 3],
             [1, 1, 4], [0, 1, 3], [0, 1, 4], [0, 1, 5]],  # xts:6
            [[4, 0, 0], [3, 0, 0], [3, 0, 1], [3, 0, 2], [2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 0, 4],
             [1, 0, 3], [1, 0, 4], [1, 0, 5], [0, 0, 4],
             [0, 0, 5], [0, 0, 6]],  # xts:7
        ]
        # xts_zd = [
        #     [],
        #     [[0, 6, 0]],  # xts:1
        #     [[1, 5, 0], [0, 5, 0], [0, 5, 1]],  # xts:2
        #     [[1, 4, 0], [1, 4, 1], [0, 4, 0], [0, 4 ,1], [0, 4, 2]],  # xts:3
        #     [[2, 3, 0], [1, 3, 0],[1, 3, 1],[1, 3, 2],[0, 3, 0],[0, 3, 1],[0, 3, 2],[0, 3, 3]],  # xts；4
        #     [[3, 2, 0], [2, 2, 0], [2, 2, 1], [1, 2, 0], [1, 2 ,1],[1, 2 ,2],[1, 2, 3],[0, 2, 0],[0, 2, 1],[0, 2 ,2],[0, 2, 3],
        #     [0, 2, 4]],  # xts:5
        #     [[3, 1, 0], [3, 1, 1], [2, 1, 0], [2, 1, 1], [2, 1, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [0, 1, 0],
        #     [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]],  # xts:6
        #     [[4, 0, 0], [3, 0, 0],[3, 0, 1],[3, 0, 2],[2, 0, 0],[2, 0 ,1],[2, 0 ,2],[2, 0 ,3],[2, 0 ,4],[1, 0, 0],[1, 0, 1],
        #     [1, 0, 2],[1, 0, 3],[1, 0, 4],[1, 0, 5],[0, 0, 0],[0, 0, 1],[0, 0, 2],[0, 0, 3],[0, 0, 4],[0, 0, 5],[0, 0, 6]],  # xts:7
        # ]

        # -------------------随机出的T3与T2数量-start----------------------
        r = random.uniform(0, sum_xts[-1])
        xts = 0
        # 获取随机得到的向听数
        for i in range(0, len(sum_xts)):
            if r <= sum_xts[i]:
                xts = i
                break
        # print('xts:',xts)
        # 再random一组t23
        xtSet = copy.deepcopy(xts_zd[xts])
        P1_N32 = random.choice(xtSet)
        # print("P1_N32 is {}".format(P1_N32))
        # -------------start-2.获取模拟T3后的wall,以及模拟出来的T3-start-------------
        simulate_T3 = []
        # ----------2.1 生成T3分配度表,只考虑顺子,注意不要模拟到相同的的顺子-----------
        t3 = [0] * (34 + 7 * 3)  # 34种刻子，21种顺子
        for i in range(34, 34 + 21):
            index = MJ.translate16_33(int((i - 34) / 7) * 16 + (i - 34) % 7 + 1)
            t3[i] = min([wall[index], wall[index + 1], wall[index + 2]])

        for i in range(P1_N32[0] - len(suits)):
            t3_sum = copy.copy(t3)  # t3分配度累加表
            for i in range(1, len(t3_sum)):
                t3_sum[i] = t3_sum[i - 1] + t3_sum[i]
            if t3_sum[-1] == 0:
                print("failed to simulate T3 qd")
                # return wall
            # ----------------2.2 随机模拟T3---------------------
            r = random.uniform(0, t3_sum[-1])
            j = 0
            flag = False
            # 找到随机的T3的下标
            while j < len(t3_sum) and not flag:
                if r <= t3_sum[j]:  # 下标
                    # 记录t3
                    if t3Set[j] not in simulate_T3:
                        simulate_T3.append(t3Set[j])
                    # print('模拟得到的t3 is {}'.format(MJ.convert_hex2index(t3Set[j][0]))
                    for card in t3Set[j]:
                        if wall[MJ.convert_hex2index(card)] > 0:
                            wall[MJ.convert_hex2index(card)] -= 1
                        flag = True
                        t3[j] = 0  # 对这个模拟到的顺子的分配度置为0,在下一次模拟当中就不会模拟到重复的顺子
                j += 1

        # -----------------3.获取T2模拟之后的牌墙，危险度，以及模拟出来的T2----------------------
        # -------------3.1 根据弃牌，得到对应的弃牌对应的有效牌的标记数组，不再分配有该弃牌为有效牌的搭子组合,弃牌对应的t2分配度为0------
        P = [0] * (34 + 15 * 3)
        for card in discard_real:
            for i in efc_dzindex[card]:
                P[i] += 1
        RT = [[0] * 34, [0] * 34]  # 危险度表
        simulate_T2 = []  # 用于孤张模拟的T2
        hand_T2 = []  # 用于统计手牌的T2
        index_list = []
        count_duizi = 0
        for y in range(P1_N32[1] + P1_N32[2]):
            # ---------------------------------------3.2 计算搭子的分配度--------------------------------
            t2 = [0] * (34 + (15 * 3))  # 计算搭子的分配度表
            for i in range(len(wall)):
                if wall[i] >= 2:
                    t2[i] = 2
                if i < 27:
                    color = int(i / 9)
                    # 八万，八条，八筒或一万，一条，一筒时
                    if i % 9 + 1 == 8 or i % 9 + 1 == 1:  # 模拟八万九万，八筒九筒，八条九条或者一万二万，一条二条，一筒二筒的分配度
                        if sum([wall[MJ.convert_hex2index(e)] for e in
                                dzEfc[33 + color * 8 + (i % 9) + 1]]) != 0:
                            t2[33 + color * 8 + (i % 9) + 1] = min(wall[i], wall[i + 1]) * 0.7
                            # t2[33 + color * 15 + (i % 9) + 1] = 1
                    elif i % 9 + 1 == 9:
                        pass
                    else:
                        if sum([wall[MJ.convert_hex2index(e)] for e in
                                dzEfc[33 + color * 8 + (i % 9) + 1]]) != 0:
                            t2[33 + color * 8 + (i % 9) + 1] = min(wall[i], wall[i + 1]) * 0.8
                            # t2[33 + color * 15 + (i % 9) + 1] = 1

                        if sum([wall[MJ.convert_hex2index(e)] for e in
                                dzEfc[33 + 24 + color * 7 + (i % 9) + 1]]) != 0:
                            t2[33 + 24 + color * 7 + (i % 9) + 1] = min(wall[i], wall[i + 2]) * 0.7

            # ----------------3.3 移除t2中的不可分表,已经模拟得到的顺子或者隔了一张牌的那种搭子对应的分配度置为0-------
            Pc = [0] * (34 + (15 * 3))
            for x in range(len(t2)):
                if P[x] == 0:
                    Pc[x] = t2[x]
                for i in index_list:  # 已经模拟过的顺子对应的分配度为0
                    Pc[x] = 0

            Pc_sum = copy.copy(Pc)

            # ---------------------------------------3.4 T2分配度的累加表---------------------------
            for j in range(1, len(Pc_sum)):
                Pc_sum[j] = Pc_sum[j] + Pc_sum[j - 1]
            # --------------------------------------3.5 模拟搭子，得到模拟之后的牌墙和危险度，以及后续模拟T1用的搭子集合------------
            # if Pc_sum[34]==Pc_sum[35] and Pc_sum[35]==Pc_sum[36]:
            #     print("warning")
            Pc_duizi = Pc_sum[0:34]  # 用于对子的模拟的分配度累加表
            Pc_dazi = Pc_sum[34:]  # 用于对不是对子的搭子的模拟的分配度累加表

            # print("Pc_duizi is {}".format(Pc_duizi))
            # print("Pc_dazi is {}".format(Pc_dazi))

            if count_duizi < P1_N32[1]:
                r = random.uniform(0, Pc_duizi[-1])
                for x in range(len(Pc_duizi)):
                    if r <= Pc_duizi[x]:
                        # 更新wall
                        dz_index = x
                        # 统计手牌当中的T2
                        hand_T2.append(dzSet[dz_index])
                        for card in dzSet[dz_index]:
                            if wall[MJ.convert_hex2index(card)] > 0:
                                wall[MJ.convert_hex2index(card)] -= 1
                        # 0-33aa
                        # if dz_index < 34:
                        #     for card in dzEfc[dz_index]:
                        #         RT[0][MJ.convert_hex2index(card)] += 1
                        # else:  # 34+是aa/ab
                        #     for card in dzEfc[dz_index]:
                        #         RT[1][MJ.convert_hex2index(card)] += 1
                        break
            else:
                r = random.uniform(Pc_dazi[0], Pc_dazi[-1])
                # print("随机数：{}".format(r))
                # print("累加度分配表：{}".format(Pc_dazi))
                length = 34 + len(Pc_dazi)
                for x in range(0, len(Pc_dazi)):
                    if r <= Pc_dazi[x]:
                        # 更新wall
                        dz_index = x + 34
                        if dzSet[dz_index] not in simulate_T2 and dz_index > 33:
                            simulate_T2.append(dzSet[dz_index])
                        # 统计手牌当中的T2
                        hand_T2.append(dzSet[dz_index])

                        for card in dzSet[dz_index]:
                            if wall[MJ.convert_hex2index(card)] > 0:
                                wall[MJ.convert_hex2index(card)] -= 1
                        # 0-33aa
                        # if dz_index < 34:
                        #     for card in dzEfc[dz_index]:
                        #         RT[0][MJ.convert_hex2index(card)] += 1
                        # else:  # 34+是aa/ab
                        #     for card in dzEfc[dz_index]:
                        #         RT[1][MJ.convert_hex2index(card)] += 1
                        index_list.append(dz_index)  # 在下一个循环当中，index_list列表当中的值所对应的的搭子对应的分配度置为0,就可以不模拟到重复的搭子
                        break
            count_duizi += 1
        # ---------------------------------4.模拟孤张,孤张不重复---------------------------
        T3_T2_set = simulate_T3 + simulate_T2
        # 计算当前玩家的非孤张数
        P1_N1 = P1_N32[0] * 3 + P1_N32[1] * 2 + P1_N32[2] * 2
        # 得到非孤张牌的集合
        No_T1 = []  # 非孤张的集合
        for T3Set in T3_T2_set:
            for card in T3Set:
                if MJ.convert_hex2index(card) not in No_T1:
                    No_T1.append(MJ.convert_hex2index(card))
        No_T1.sort()
        # 移除掉绝对不可能是孤张的牌
        gz_wall = copy.copy(wall)
        for index in No_T1:
            gz_wall[index] = 0

        gz_card = []
        for i in range(13 - P1_N1):
            # 孤张分配度的累加和
            gz_sum = copy.copy(gz_wall)
            for i in range(1, len(gz_sum)):
                gz_sum[i] = gz_sum[i] + gz_sum[i - 1]
            # 模拟孤张,当孤张个数大于2时，注意模拟到的孤张不要有重复
            r = random.uniform(0, gz_sum[-1])
            for x in range(len(gz_sum)):
                if r < gz_sum[x]:
                    gz_card.append(x)
                    gz_wall[x] = 0  # 更新孤张的分配度，下次模拟就可以不用模拟到重复的孤张
                    break

        # 从牌墙当中减去孤张
        # print('wall:',wall)
        for i in gz_card:
            # print(MJ.translate33_10(i))
            # print(i)
            wall[i] = wall[i] - 1
        # print('wall:', wall)
        hand = []
        hand_T3_t2 = simulate_T3 + hand_T2
        for card in gz_card:
            hand.append(MJ.translate33_16(card))
        for cards in hand_T3_t2:
            for card in cards:
                hand.append(card)
        # print("用于模拟孤张所需要的T3_T2_set is {}".format(T3_T2_set))
        # print("模拟得到的孤张为gz_card is {}".format(gz_card))
        # print("模拟得到的手中的T3与T2为hand_t3_t2 is {}".format(hand_T3_t2))
        # print("手牌为hand is {}".format(hand))
        # print(hand, wall, RT)
        return hand, wall, RT


    def simulate_ssl(self, suits, discards_real, wall, sum_xts):
        """
        十三烂手牌的模拟方法
        将十三烂手牌拆成Z, L3, L2, L1 （字牌，147烂牌，14烂牌，非字牌的单张）和剩余孤张牌这5种牌型进行模拟，总共13张牌
        :param wall: 牌墙的牌数量 1*34
        :param discards_real: 玩家出牌
        :param sum_xts: 在某一轮不同向听数对应局数的求和表
        :return: handcards_ssl, wall, RT
        """
        # print('begin_wall:',wall)
        # 分配参数
        L2_fenpei_ratios = {5:1, 4:1, 3:0.9, 2:0.8, 1:0.7, 0:0.5} # L2牌型分配度根据其有效牌个数确定参数大小
        L1_fenpei_ratios = {6:0.9, 5:0.8, 4:0.7} # L1牌型分配度根据其有效牌个数确定参数大小
        discards_eff_L1_fenpei_ratios = 0.2 #  根据出牌的有效牌估计L1牌型的分配度
        # 根据出牌确定能将该出牌作为有效牌的L2的分配度
        discards_L2_fenbu_ratios = {1:[0,0,0,0.5,0.5,0.7],2:[0.5,0.5,0.7],3:[0.7],4:[0,0.5,0.7],5:[0.5,0.7,0,0.5],6:[0.7,0.5,0],7:[0.7],8:[0.7,0.5,0.5],9:[0.7,0.5,0,0.5,0,0]}
        
        # 确定烂牌牌组L3 
        L3 = [[1,4,7],[1,4,8],[1,4,9],[1,5,8],[1,5,9],[1,6,9],[2,5,8],[2,5,9],[2,6,9],[3,6,9]]
        '''
            # 烂牌牌组L2 
            # def get_L2():
            #     L2 = []
            #     L2_need = []
            #     # L2
            #     for i in range(1,10):
            #         for j in range(4, 10):
            #             if j - i >= 3:
            #                 L2.append([i,j])
            #     cards = [1,2,3,4,5,6,7,8,9]
            #     _L3 = [[1,4,7],[1,4,8],[1,4,9],[1,5,8],[1,5,9],[1,6,9],[2,5,8],[2,5,9],[2,6,9],[3,6,9]]
            #     # L2_need
            #     for L2_cards in L2:
            #         _need = []
            #         _L2_cards = copy.copy(L2_cards)
            #         for card in cards:
            #             _L2_cards.append(card)
            #             _L2_cards.sort()
            #             # print('_L2_cards:',_L2_cards)
            #             if _L2_cards in _L3:
            #                 _need.append(card)
            #             _L2_cards.remove(card)
            #         L2_need.append(_need)
            #     return L2, L2_need
            # L2, L2_need = get_L2()
            # L2 = [[1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [3, 6], [3, 7], [3, 8], [3, 9], [4, 7], [4, 8], [4, 9], [5, 8], [5, 9], [6, 9]]
            # L2_need = [[7, 8, 9], [8, 9], [9], [4], [4, 5], [4, 5, 6], [8, 9], [9], [], [5], [5, 6], [9], [], [], [6], [1], [1], [1], [1, 2], [1, 2], [1, 2, 3]]
        '''
        efc_L2_index = {1:[15,16,17,18,19,20],2:[18,19,20],3:[20],4:[3,4,5],5:[4,5,9,10],6:[5,10,14],7:[0],8:[0,1,6],9:[0,1,2,6,7,11]}
        
        # 烂牌牌组L1 
        L1 = [1,2,3,4,5,6,7,8,9]
        L1_need = [[4,5,6,7,8,9],[5,6,7,8,9],[6,7,8,9],
                    [1,7,8,9],[1,2,8,9],[1,2,3,9],
                    [1,2,3,4],[1,2,3,4,5],[1,2,3,4,5,6]]
        # 确定字牌集合Z [0x31,0x32,0x33,0x34,0x35,0x36,0x37]
        Z = [0x31,0x32,0x33,0x34,0x35,0x36,0x37]

        # 向听数对应的L, Z的数目表
        xts_ZL = [  [],
                    [[7,2,0,0],[7,1,1,1],[7,0,3,0],[6,2,0,1],[6,1,2,0],[5,2,1,0],[4,3,0,0]],
                    [[7,1,1,0],[7,0,2,1],[7,1,0,2],[6,2,0,0],[6,1,1,1],[6,0,3,0],[5,2,0,1],[5,1,2,0],[4,2,1,0],[3,3,0,0]],
                    [[7,1,0,1],[7,0,2,0],[7,0,1,2],[6,1,1,0],[6,0,2,1],[6,1,0,2],[5,2,0,0],[5,1,1,1],[5,0,3,0],[4,2,0,1],[4,1,2,0],[3,2,1,0],[2,3,0,0]],
                    [[7,1,0,0],[7,0,1,1],[6,1,0,1],[6,0,2,0],[6,0,1,2],[5,1,1,0],[5,0,2,1],[5,1,0,2],[4,2,0,0],[4,1,1,1],[4,0,3,0],[3,2,0,1],[3,1,2,0],[2,2,1,0],[1,3,0,0]],
                    [[7,0,1,0],[7,0,0,2],[6,1,0,0],[6,0,1,1],[5,1,0,1],[5,0,2,0],[5,0,1,2],[4,1,1,0],[4,0,2,1],[4,1,0,2],[3,2,0,0],[3,1,1,1],[3,0,3,0],[2,2,0,1],[2,1,2,0],[1,2,1,0],[0,3,0,0]],
                    [[7,0,0,1],[6,0,1,0],[6,0,0,2],[5,1,0,0],[5,0,1,1],[4,1,0,1],[4,0,2,0],[4,0,1,2],[3,1,1,0],[3,0,2,1],[3,1,0,2],[2,2,0,0],[2,1,1,1],[2,0,3,0],[1,2,0,1],[1,1,2,0],[0,2,1,0]],
                    [[7,0,0,0],[6,0,0,1],[5,0,1,0],[5,0,0,2],[4,1,0,0],[4,0,1,1],[3,1,0,1],[3,0,2,0],[3,0,1,2],[2,1,1,0],[2,0,2,1],[2,1,0,2],[1,2,0,0],[1,1,1,1],[1,0,3,0],[0,2,0,1],[0,1,2,0]],
                    [[6,0,0,0],[5,0,0,1],[4,0,1,0],[4,0,0,2],[3,1,0,0],[3,0,1,1],[2,1,0,1],[2,0,2,0],[2,0,1,2],[1,1,1,0],[1,0,2,1],[1,1,0,2],[0,2,0,0],[0,1,1,1],[0,0,3,0]],
                    [],[]
                ]
        
        # 根据当前轮数的向听数获得十三烂的Z,L3,L2,L1牌组的数目
        r=random.uniform(0,sum_xts[-1])
        xts=0
        #获取随机得到的向听数
        for i in range(0,len(sum_xts)):
            if r<=sum_xts[i]:
                xts=i
                break
        # print('xts:',xts)
        #获得了xts对应的十三烂牌组
        xtSet=copy.deepcopy(xts_ZL[xts])
        if xtSet==[]: # 向听数超过了8
            rand_ZL = [1,1,0,1]
        else:
            def get_enable_L3_color():
                num = 3
                enable_color = [0,1,2]
                for color in range(3):
                    _t3 = [0] * 10
                    for i in range(10):
                        card1 = MJ.translate16_33(color*16 + L3[i][0])
                        card2 = MJ.translate16_33(color*16 + L3[i][1])
                        card3 = MJ.translate16_33(color*16 + L3[i][2])
                        _t3[i] = min([wall[card1],wall[card2],wall[card3]])
                    _t3_sum = copy.copy(_t3)
                    for i in range(1, len(_t3_sum)):
                        _t3_sum[i] = _t3_sum[i - 1] + _t3_sum[i]
                    if _t3_sum[-1]==0:
                        num -= 1
                        enable_color.remove(color)
                return num, enable_color
            num, enable_color = get_enable_L3_color()
            ZL_nums = copy.copy(xts_ZL[xts])
            # print('num:',num)
            # print('ZL_nums: ',ZL_nums)
            # 剔除绝对无法形成的组合
            ZL_nums_ = list(filter(lambda x:x[1]<=num and sum([wall[i]>=1 for i in range(27,34)]) >= x[0], ZL_nums))
            # print('after ZL_nums: ',ZL_nums_)
            if ZL_nums_ == []:
                new_xts = xts+3 if xts+3 <= 8 else 8 
                ZL_nums = copy.copy(xts_ZL[new_xts])
                ZL_nums_ = list(filter(lambda x:x[1]<=num and sum([wall[i]>=1 for i in range(27,34)]) >= x[0], ZL_nums))
                if ZL_nums_ == []:
                    ZL_nums_ = [[0,1,1,0]]
            # print('after ZL_nums: ',ZL_nums_)
            rand_ZL = random.choice(ZL_nums_)
        # print('rand_ZL:',rand_ZL)
        
        # 十三烂手牌
        handcards_ssl = []

        # Z 分配
        Z_fenpei_ = [0x31,0x32,0x33,0x34,0x35,0x36,0x37]
        Z_fenpei = copy.copy(Z_fenpei_)
        for Z in Z_fenpei_:
            if wall[Z-22] <= 0:
                Z_fenpei.remove(Z)
        for _ in range(rand_ZL[0]):
            Z = random.choice(Z_fenpei)
            Z_fenpei.remove(Z)
            wall[MJ.convert_hex2index(Z)] -= 1
            handcards_ssl.append(Z)
        
        # ssl的不靠牌必须要是不同花色的
        color_fenpei = [0, 1, 2] # 对应万，条，筒
        
        # L3 分配
        for _ in range(rand_ZL[1]):
            if len(enable_color) == 0:
                print('simulate L3 failed, {}/{}'.format(_+1, rand_ZL[1]))
                continue
            color = random.choice(enable_color)
            color_fenpei.remove(color)
            enable_color.remove(color)
            # L3的分配表
            t3 = [0] * (10) 
            for i in range(10):
                card1 = MJ.translate16_33(color*16 + L3[i][0])
                card2 = MJ.translate16_33(color*16 + L3[i][1])
                card3 = MJ.translate16_33(color*16 + L3[i][2])
                t3[i] = min([wall[card1],wall[card2],wall[card3]])
            # print('t3:', t3, 'color:',color)
            # 根据L3分配表进行分配
            t3_sum = copy.copy(t3) #累加表
            #随机出要分配的L3
            for i in range(1, len(t3_sum)):
                t3_sum[i] = t3_sum[i - 1] + t3_sum[i]
            if t3_sum[-1]==0:
                print ("failed to simulate L3")
                # return 
            #随机数
            r = random.uniform(0, t3_sum[-1])
            j = 0
            flag = False
            #找到随机的L3的下标
            while j < len(t3_sum) and not flag:
                if r <= t3_sum[j]:  # 下标
                    card1 = MJ.translate16_33(color*16 + L3[j][0])
                    card2 = MJ.translate16_33(color*16 + L3[j][1])
                    card3 = MJ.translate16_33(color*16 + L3[j][2])
                    handcards_ssl.append(color*16 + L3[j][0])
                    handcards_ssl.append(color*16 + L3[j][1])
                    handcards_ssl.append(color*16 + L3[j][2])
                    wall[card1] -= 1
                    wall[card2] -= 1
                    wall[card3] -= 1
                    flag = True
                j += 1
        
        # L2 分配
        L2_allocated_j = []
        # 根据已出牌，不再分配有该弃牌为有效牌的L2组合
        t2_all_color = [1] * (21*3)
        for card in discards_real:
            if card & 0xf0 < 3: # card 要是花色牌
                huase = card & 0xf0
                for i in efc_L2_index[card%16]:
                    t2_i_ratios = 1 - (1 / len(L2_need[i]))
                    t2_all_color[huase*21+i] *= t2_i_ratios
        # print('t2_all_color:', t2_all_color)
        for _ in range(rand_ZL[2]):
            # L2 分配表
            t2 = [0] * 21
            # 剔除无法分配L2的花色
            enable_t2_color = copy.copy(color_fenpei)
            for color in color_fenpei:
                for i in range(21):
                    card1 = MJ.translate16_33(color*16 + L2[i][0])
                    card2 = MJ.translate16_33(color*16 + L2[i][1])
                    t2[i] = min([wall[card1],wall[card2]])
                if sum(t2) == 0:
                    enable_t2_color.remove(color)
            # 选择一种花色的L2
            if len(enable_t2_color) == 0:
                print('simulate L2 failed, {}/{}'.format(_+1, rand_ZL[2]))
                continue
            color = random.choice(enable_t2_color)
            color_fenpei.remove(color)
            for i in range(21):
                card1 = MJ.translate16_33(color*16 + L2[i][0])
                card2 = MJ.translate16_33(color*16 + L2[i][1])
                # print('card1:{},card2:{}, ratio:{}'.format(card1,card2,L2_fenpei_ratios[len(L2_need[i])]))
                t2[i] = min([wall[card1],wall[card2]]) * L2_fenpei_ratios[len(L2_need[i])]
            #减少L2 分配表t2中的分配可能性不高的L2组合分布概率
            t2_ = [0] * 21
            for x in range(len(t2)):
                t2_[x]=t2[x] * t2_all_color[color*21+x]
            # 根据L2分配表进行分配
            t2_sum = copy.copy(t2_) #累加表
            #随机出要分配的L2
            for i in range(1, len(t2_sum)):
                t2_sum[i] = t2_sum[i - 1] + t2_sum[i]
            if t2_sum[-1]==0:
                print ("failed to simulate L2 and resimulate")
                t2_sum = copy.copy(t2) # 不排除有效牌
                for i in range(1, len(t2_sum)):
                    t2_sum[i] = t2_sum[i - 1] + t2_sum[i]
                # return 
            #随机数
            r = random.uniform(0, t2_sum[-1])
            j = 0
            flag = False
            #找到随机的L2的下标
            while j < len(t2_sum) and not flag:
                if r <= t2_sum[j]:  # 下标
                    card1 = MJ.translate16_33(color*16 + L2[j][0])
                    card2 = MJ.translate16_33(color*16 + L2[j][1])
                    handcards_ssl.append(color*16 + L2[j][0])
                    handcards_ssl.append(color*16 + L2[j][1])
                    wall[card1] -= 1
                    wall[card2] -= 1
                    flag = True
                    L2_allocated_j.append([color,j])
                j += 1
        
        # L1 分配
        L1_allocated_j = []
        # 根据已出牌，不再分配有该弃牌为有效牌的L1牌
        t1_all_color = [0] * (9*3)
        for card in discards_real:
            if card & 0xf0 < 3: # card 要是花色牌
                huase = card & 0xf0
                for i in L1_need[card%16 - 1]:
                    t1_all_color[huase*9 + L1.index(i)] += 1
        # print('t1_all_color:', t1_all_color)
        for _ in range(rand_ZL[3]):
            # L1 分配表
            t1 = [0] * (9)
            color = random.choice(color_fenpei)
            color_fenpei.remove(color)
            for i in range(9):
                card1 = MJ.translate16_33(color*16 + L1[i])
                t1[i] = wall[card1] * L1_fenpei_ratios[len(L1_need[i])]
            #移除L1 分配表t1中的不可分配的L1组合.不是完全移除，而是分配度给的很低
            t1_ = [0] * 9
            for x in range(len(t1)):
                if t1_all_color[color*9+x]==0:
                    t1_[x]=t1[x]
                else:
                    t1_[x]=t1[x]*discards_eff_L1_fenpei_ratios
            # 根据L1分配表进行分配
            t1_sum = copy.copy(t1_) #累加表
            #随机出要分配的L1
            for i in range(1, len(t1_sum)):
                t1_sum[i] = t1_sum[i - 1] + t1_sum[i]
            if t1_sum[-1]==0:
                print ("failed to simulate L1 and resimulate")
                t1_sum = copy.copy(t1) # 不排除有效牌
                for i in range(1, len(t1_sum)):
                    t1_sum[i] = t1_sum[i - 1] + t1_sum[i]
                # return 
            #随机数
            r = random.uniform(0, t1_sum[-1])
            j = 0
            flag = False
            #找到随机的L1的下标
            while j < len(t1_sum) and not flag:
                if r <= t1_sum[j]:  # 下标
                    card1 = MJ.translate16_33(color*16 + L1[j])
                    handcards_ssl.append(color*16 + L1[j])
                    wall[card1] -= 1
                    flag = True
                    L1_allocated_j.append([color,j])
                j += 1

        # 剩余牌分配表
        rest = [True] * 34
        for j in L2_allocated_j:
            for need_card in L2_need[j[1]]:
                rest[9*j[0]+need_card-1] = False
        for j in L1_allocated_j:
            for need_card in L1_need[j[1]]:
                rest[9*j[0]+need_card-1] = False
        Z_fenpei_rest = Z_fenpei_
        for zi_pai in [0x31,0x32,0x33,0x34,0x35,0x36,0x37]:
            if zi_pai in Z_fenpei_rest:
                rest[zi_pai-22] = False
        # 剩余牌分配
        for _ in range(13-len(handcards_ssl)):
            # print('handcards_ssl:',handcards_ssl)
            for i in range(len(rest)):
                if rest[i]:
                    rest[i] = wall[i]
                else:
                    rest[i] = 0
            # print('rest:',rest)
            # 根据r分配表进行分配
            r_sum = copy.copy(rest) #累加表
            #随机出要分配的r 孤张牌
            for i in range(1, len(r_sum)):
                r_sum[i] = r_sum[i - 1] + r_sum[i]
            # print('r_sum:',r_sum)
            if r_sum[-1]==0:
                print ("failed to simulate r")
                return 
            #随机数
            r = random.uniform(0, r_sum[-1])
            j = 0
            flag = False
            #找到随机的r的下标
            while j < len(r_sum) and not flag:
                if r <= r_sum[j]:  # 下标
                    card1 = j
                    handcards_ssl.append(int(card1 / 9) * 16 + card1 % 9 + 1)
                    wall[card1] -= 1
                    flag = True
                j += 1
            
        RT = [[0]*34,[0]*34] #危险度表
        # print('handcards_ssl:',handcards_ssl)
        # print('after_wall:',wall)
        # print('-------------------end----------------------')
        return handcards_ssl, wall, RT


    def simulate_jy(self, suits, discards_real, wall, sum_xts):
        '''
        模拟91胡牌类型。传入sum_xt
        :param sum_xts:  统计的向听数
        :param wall:  牌墙
        :return: 返回危险度、模拟后的牌墙
        '''
        r = random.uniform(0, sum_xts[-1])
        xts = 0
        # 获取随机得到的向听数
        for i in range(0, len(sum_xts)):
            if r <= sum_xts[i]:
                xts = i
                break
        # print('xts:',xts)
        # xts = 10
        # 91的有效牌，也是危险牌
        RT91 = [[0] * 34, [0] * 34]
        simulate_handcards_index = []  # 模拟的手牌
        suits_len = len(suits)  # 副露长度,此时的副露都是1或9或字牌
        need_simulation_eff_cards_len = 14 - xts - suits_len * 3  # 需要模拟手牌中有效牌的数量

        effic_card_to_single = 0 #当91的有效牌全部模拟完后，剩下需要模拟的个数需要转成孤张

        break_flag = False # 跳出最外层循环标志

        for _ in range(need_simulation_eff_cards_len):  # 模拟 手牌中的有效牌
            t_91 = [0] * (6 + 7)  # 1、9 各两张 + 7张字牌
            for i in range(0, len(t_91)):
                if i < 6:  # 1、9牌    0-5
                    if i % 2 == 0:  # 0是1万、条、筒
                        t_91[i] = wall[(i//2) * 9] # 数量作为分配度
                    else: # ，1是9万、条、筒
                        t_91[i] = wall[((i+1)//2) * 8 + i//2]
                else:  # 6-12
                    t_91[i] = wall[i + 27 - 6]  # 字牌

            t_91_sum = [0] * (6 + 7)
            for index in range(len(t_91)): # copy t_91 to t_91_sum
                if t_91_sum[index] == 0:
                    t_91_sum[index] = t_91[index]
            for index in range(1, len(t_91_sum)): # 生成91的分配度表
                t_91_sum[index] = t_91_sum[index] + t_91_sum[index - 1]

            # 生成随机数选择九幺有效牌
            r = random.uniform(0, t_91_sum[-1])
            select_t91_index = -1 # 选择91牌的下标
            for i in range(0, len(t_91_sum)):
                if r <= t_91_sum[i]:
                    select_card_index = i
                    card_index = -1  # 牌的下标
                    if select_card_index < 6:  # 1、9牌    0-5
                        if select_card_index % 2 == 0:  # 0是1万、条、筒
                            card_index = (select_card_index // 2) * 9  #
                        else:  # ，1是9万、条、筒
                            card_index = ((select_card_index + 1) // 2) * 8 + select_card_index // 2
                    else:  # 6-12
                        card_index = select_card_index + 27 - 6  # 字牌

                    if wall[card_index] == 0:
                        effic_card_to_single = need_simulation_eff_cards_len - len(simulate_handcards_index)
                        break_flag = True
                        break
                    wall[card_index] -= 1 # 从牌墙减一张对应牌
                    simulate_handcards_index.append(card_index)
                    break

            if break_flag:
                break

        single_cards_index = []  # 存储模拟的孤张
        for _ in range(xts - 1 + effic_card_to_single):  # 模拟孤张，随机
            t_91_single = [0] * (7 * 3)  # 2-8万、条、筒
            for i in range(len(t_91_single)):
                if 0 <= i < 7:  # 2-8万
                    t_91_single[i] = wall[i+1] # 数量作为分配度
                elif 7 <= i < 14:  # 2-8 条
                    t_91_single[i] = wall[i+3]
                else: # 2-8筒
                    t_91_single[i] = wall[i+5]
            t_91_single_sum = [0] * (7 * 3)

            for index in range(len(t_91_single_sum)):  # copy t_91_single to t_91_single_sum
                if t_91_single_sum[index] == 0:
                    t_91_single_sum[index] = t_91_single[index]

            for index in range(1, len(t_91_single_sum)): # 生成91孤张的分配度表
                t_91_single_sum[index] = t_91_single_sum[index] + t_91_single_sum[index - 1]

            # 生成随机数选择九幺的孤张
            single_card_index = -1
            r_single = random.uniform(0, t_91_single_sum[-1])
            for i in range(0, len(t_91_single_sum)):
                if r_single <= t_91_single_sum[i]:
                    if 0 <= i < 7:  # 2-8万
                        single_card_index = i + 1  # 数量作为分配度
                    elif 7 <= i < 14:  # 2-8 条
                        single_card_index = i + 3
                    else:  # 2-8筒
                        single_card_index = i + 5
                    single_cards_index.append(single_card_index)
                    wall[single_card_index] -= 1
                    break
        # 计算危险牌
        for card_index in set(simulate_handcards_index):
            if simulate_handcards_index.count(card_index) > 1:
                RT91[0][card_index] = 1

        # 可视化摸到的手牌，方便debug，正式代码环境中可以关闭
        simulate_handcards_index.extend(single_cards_index)
        simulate_handcards = [MJ.translate33_16(index) for index in simulate_handcards_index]
        simulate_handcards.sort()
        return simulate_handcards,  wall, RT91


    def simulate_handcards(self, type, suits, discards_real, wall, sum_xts, fei_king_num):
        
        if type == 0:
            return self.simulate_ph(suits, discards_real, wall, sum_xts, fei_king_num)
        if type == 1:
            return self.simulate_qd(suits, discards_real, wall, sum_xts)
        if type == 2:
            return self.simulate_ssl(suits, discards_real, wall, sum_xts)
        if type == 3:
            return self.simulate_jy(suits, discards_real, wall, sum_xts)

    def reg_prob(self, probability_relative_list):  #
        '''
        对输入的概率进行正则化，等比变化使和为1
        :param probability_relative_list:  相对概率
        :return:绝对概率，概率值相加为1
        '''
        probability_relative_list = np.array(probability_relative_list)
        total_prob = sum(probability_relative_list)  # 当前概率和
        if total_prob == 0:
            return probability_relative_list
        probability_abs_list = [probability_relative_list[index] / total_prob for index
                                in range(len(probability_relative_list))]
        return probability_abs_list

    # 根据统计和出牌确定胡牌倾向：
    def hu_prefer(self, discards_real, suits, tongji_prob=[0.7873, 0.0134, 0.0752, 0.1241]):
        '''
        胡牌倾向比例判断
        :param discards_real: 实际弃牌
        :param suits: 副露
        :param tongji: 统计概率
        :param round: 轮数，模拟第几轮
        :return: 牌型比例
        '''
        # 统计在有副露下和没有副露下轮数与胡牌类型比例之间的关系
        tongji_prob = [
                [[0, 0, 0, 0],
                [0.8732970027247956, 0.0, 0.0, 0.12670299727520437],
                [0.8825543916196615, 0.0, 0.0, 0.11744560838033843],
                [0.8829688535453943, 0.0, 0.0, 0.1170311464546057],
                [0.8881018375936762, 0.0, 0.0, 0.11189816240632379],
                [0.8925258183091196, 0.0, 0.0, 0.10747418169088045],
                [0.8915429309231763, 0.0, 0.0, 0.10845706907682376],
                [0.8905524419535629, 0.0, 0.0, 0.10944755804643715],
                [0.8872642685054416, 0.0, 0.0, 0.11273573149455844],
                [0.8854337152209493, 0.0, 0.0, 0.11456628477905073],
                [0.8843157894736842, 0.0, 0.0, 0.11568421052631579],
                [0.8819392112534539, 0.0, 0.0, 0.11806078874654609],
                [0.8836126629422719, 0.0, 0.0, 0.11638733705772812],
                [0.8780922224421136, 0.0, 0.0, 0.1219077775578864],
                [0.88259526261586, 0.0, 0.0, 0.11740473738414006],
                [0.8854389721627409, 0.0, 0.0, 0.1145610278372591],
                [0.8782523318605793, 0.0, 0.0, 0.12174766813942071],
                [0.8898614150255288, 0.0, 0.0, 0.11013858497447118],
                [0.8966267682263329, 0.0, 0.0, 0.10337323177366703],
                [0.896, 0.0, 0.0, 0.104],
                [0.9088541666666666, 0.0, 0.0, 0.09114583333333333],
                [0.900990099009901, 0.0, 0.0, 0.09900990099009901],
                [0.9230769230769231, 0.0, 0.0, 0.07692307692307693],
                [0.9629629629629629, 0.0, 0.0, 0.037037037037037035],
                [1.0, 0.0, 0.0, 0.0]],
                [[0, 0, 0, 0],
                [0.7694162507710425, 0.015028318286323108, 0.08557169292883979, 0.12998373801379465],
                [0.7491077329808328, 0.017250495703899537, 0.1, 0.1336417713152677],
                [0.7200931951474251, 0.0215312926809673, 0.12187675745159476, 0.13649875472001285],
                [0.6826695583596214, 0.025532334384858045, 0.14520899053627762, 0.1465891167192429],
                [0.6335181762168823, 0.030560690080098582, 0.17584719654959952, 0.16007393715341958],
                [0.5770537368746139, 0.034898085237801114, 0.2124768375540457, 0.17557134033353922],
                [0.5334114888628371, 0.03751465416178194, 0.24091441969519342, 0.18815943728018758],
                [0.47633434038267874, 0.04355488418932528, 0.2797079556898288, 0.20040281973816718],
                [0.4335974643423138, 0.04469096671949287, 0.3061806656101426, 0.21553090332805072],
                [0.39575530586766544, 0.046192259675405745, 0.32251352476071576, 0.23553890969621308],
                [0.3572621035058431, 0.04897050639955482, 0.34501947690595436, 0.24874791318864775],
                [0.33233308327081773, 0.0555138784696174, 0.36534133533383345, 0.24681170292573143],
                [0.3067608476286579, 0.056508577194752774, 0.3773965691220989, 0.2593340060544904],
                [0.28175182481751826, 0.07153284671532846, 0.3897810218978102, 0.2569343065693431],
                [0.27740492170022374, 0.0894854586129754, 0.378076062639821, 0.2550335570469799],
                [0.259375, 0.103125, 0.3875, 0.25],
                [0.2641509433962264, 0.11320754716981132, 0.37735849056603776, 0.24528301886792453],
                [0.2740740740740741, 0.0962962962962963, 0.37037037037037035, 0.25925925925925924],
                [0.24324324324324326, 0.13513513513513514, 0.33783783783783783, 0.28378378378378377],
                [0.3125, 0.09375, 0.4375, 0.15625],
                [0.3076923076923077, 0.07692307692307693, 0.38461538461538464, 0.23076923076923078],
                [0.25, 0.0, 0.5, 0.25]]
               ]

        def hu_91_prefer(suits):  # 判断副露是否符合91胡牌方向
            _suits = copy.deepcopy(suits)
            for suit in _suits:
                if suit[0] != suit[1]:  # 保证是刻子
                    return False
                else:  # 保证刻子是符合九幺类型
                    if not (suit[0] & 0xf0 == 0x30 or suit[0] & 0x0f == 0x01 or suit[0] & 0x0f == 0x09):
                        return False
            return True



        def statistics_special_cards_ratio(discards):
            # 统计特殊牌在弃牌中的占比
            count_nums = [0] * 3
            discards_len = len(discards)  # 弃牌个数

            if discards_len == 0: return count_nums  # 如果当前弃牌为空
            for card in discards:
                if card & 0xf0 == 0x30: # 字牌
                    count_nums[2] += 1
                else:
                    if card & 0x0f == 1: # 1万、条、筒
                        count_nums[0] += 1
                    if card & 0x0f == 9: # 9万、条、筒
                        count_nums[1] += 1
            # 返回统计牌数
            count_ratio = [count / discards_len for count in count_nums]
            return count_ratio

        def statistics_19_cards_num(discards):
            discards_len = len(discards)  # 弃牌个数
            num_19_cards = 0
            if discards_len == 0:
                return 0
            for card in discards:
                if card & 0x0f == 1:  # 1万、条、筒
                    num_19_cards += 1
                if card & 0x0f == 9:  # 9万、条、筒
                    num_19_cards += 1
            return num_19_cards

        def get_N32_num_ratio(discards_real):
            # 获得n32的个数，用于判断是否往十三烂方向胡牌

            def get_32N(cards_real):
                cards = copy.deepcopy(cards_real)
                cards.sort()
                kz = []
                sz = []
                aa = []
                ab = []
                ac = []

                lastCard = 0
                for card in cards:
                    if card == lastCard:
                        continue
                    else:
                        lastCard = card
                    if cards.count(card) >= 3:
                        kz.append([card, card, card])
                        for i in range(3):
                            cards.remove(card)
                    if cards.count(card) >= 2:
                        aa.append([card, card])
                        for i in range(2):
                            cards.remove(card)
                    if card + 1 in cards and card + 2 in cards and card in cards:
                        sz.append([card, card + 1, card + 2])
                        for i in range(3):
                            cards.remove(card+i)
                    if card + 1 in cards and card in cards:
                        ab.append([card, card + 1])
                        for i in range(2):
                            if card+i not in cards:
                                print(card, card+i, cards)
                            cards.remove(card+i)
                    if card + 2 in cards and card in cards:
                        ac.append([card, card + 2])
                        cards.remove(card)
                        cards.remove(card+2)

                return (len(kz) + len(sz))*3 + (len(aa) + len(ab) + len(ac)) * 2

            cards_wan, cards_tiao, cards_tong, cards_zi = MJ.split_type_s(
                discards_real)
            n32 = get_32N(cards_wan) + get_32N(cards_tiao) + \
                get_32N(cards_tong) + get_32N(cards_zi)
            n32_ratio = n32 / len(discards_real)
            return n32_ratio

        # 获取该轮中的有无副露胡牌倾向分布
        hu_type_ratios_have_fulu, hu_type_ratios_nofulu = tongji_prob[0][self.round+1], tongji_prob[1][self.round+1]

        is_have_fulu = len(suits) != 0  # 有无副露判断

        if is_have_fulu:
            hu_type_ratios = hu_type_ratios_have_fulu
        else:
            hu_type_ratios = hu_type_ratios_nofulu

        ph, qd, ssl, jy = hu_type_ratios[0], hu_type_ratios[1], hu_type_ratios[2], hu_type_ratios[3]

        # 获取1，9，字牌 出牌个数比例
        special_cards_ratio = statistics_special_cards_ratio(discards_real)

        special_cards_ratio_sum = sum(special_cards_ratio)  # 特殊牌总和

        # 根据副露情况排除一些胡牌倾向
        if is_have_fulu:  # 有副露必定不是七对、十三烂
            # 判断副露中是否有非1、9及字牌的副露
            #  qd, ssl = 0, 0
            if not hu_91_prefer(suits):  # 无法胡91
                jy = 0
            else:
                if statistics_19_cards_num(discards_real) >= 2: # 丢19牌的数目>=2
                    jy = 0
                else :
                    jy = (1 - min(1, special_cards_ratio_sum*1.5)) * jy
        else:
            # 字牌的出牌占比
            zi_pai_ratio = special_cards_ratio[2]

            # 1、9万、条、筒占比
            r19 = special_cards_ratio_sum - zi_pai_ratio

            # 根据弃牌动态修改胡牌倾向
            if len(discards_real) > 0 and get_N32_num_ratio(discards_real) > 0.8:
                ssl = 0.8
            else:
                ssl = (1 - min(1, zi_pai_ratio*1 + r19*2)) * ssl  # 考虑到十三烂可以扔字牌情况
                # ssl = ssl * (1 + get_N32_num_ratio(discards_real))
            if statistics_19_cards_num(discards_real) >= 2: # 丢19牌的数目>=2
                    jy = 0
            else :
                jy = (1 - min(1, special_cards_ratio_sum*1.5)) * jy

        prob = self.reg_prob([ph, qd, ssl, jy])
        prefer_index = np.random.choice(range(len(prob)), 1, p=prob)[0]  #根据概率选择胡牌倾向

        # 若每个胡牌类型的概率为1，则直接选择该胡牌牌型
        for ratio in [ph, qd, ssl, jy]:
            if ratio == 1:
                prefer_index = ratio
        
        return prefer_index  # 0:"ph",1:"qd",2:"ssl",3:"jy"

    def getWTandRT(self):
        """
        计算牌墙牌分布表T_selfmo和危险度表RT.
        1.根据出牌和统计确定对手玩家的胡牌倾向
        2.根据胡牌倾向模拟对应的牌型的手牌
        3.计算三个对手玩家模拟手牌后的wall, RT1, RT2, RT3
        4.求模拟M次的T_selfmo,RT1, RT2, RT3平均值
        :return: 自摸概率表T_selfmo和危险度表RT1, RT2, RT3
        """

        # 总表
        wall_pred = [0] * 34
        wall_sum = [0] * 34
        RT1_sum = [[0] * 34, [0] * 34]
        RT2_sum = [[0] * 34, [0] * 34]
        RT3_sum = [[0] * 34, [0] * 34]

        for dn in range(self.M):

            wall = copy.copy(self.leftNum)
            wall_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28,
                          29, 31, 32, 33, 34, 35, 36, 37]
            # print('wall before:', list(zip(wall_index, wall)))
            # --- add ---
            opp1_hu_prefer = self.hu_prefer(self.discards1, self.discardsOp1)
            opp2_hu_prefer = self.hu_prefer(self.discards2, self.discardsOp2)
            opp3_hu_prefer = self.hu_prefer(self.discards3, self.discardsOp3)
            # opp1_hu_prefer = 0
            # opp2_hu_prefer = 0
            # opp3_hu_prefer = 0
            opps_hu_prefer = [opp1_hu_prefer, opp2_hu_prefer, opp3_hu_prefer]
            # print(opps_hu_prefer)
            # 生成三家对手的生成递增表，用于随机向听数生成
            sum_xts_opps = []
            for i in range(3):
                # 游戏轮数得到向听数
                xts_round = self.xts_round[opps_hu_prefer[i]][self.round]
                sum_xts = copy.copy(xts_round)
                # 生成递增表，用于随机数生成
                for i in range(1, len(xts_round)):
                    sum_xts[i] = sum_xts[i] + sum_xts[i - 1]
                sum_xts_opps.append(sum_xts)
            # 根据对手的胡牌倾向模拟手牌
            opp1_handcards, wall, RT1 = self.simulate_handcards(opp1_hu_prefer, self.discardsOp1, self.discards1, wall,
                                                                sum_xts_opps[0], self.fei_king1)
            # print('king:', self.king_card)
            # print('--------opp1:----------')
            # print('opp1 hu_prefer:', opp1_hu_prefer)
            # print('opp1:', opp1_handcards, 'opp1_suites:', self.discardsOp1)
            # print('opp1:', [MJ.translate16_10(c) for c in opp1_handcards], 'opp1_suites:', [
            #     [MJ.translate16_10(c) for c in suit] for suit in self.discardsOp1])
            # print('wall after1:', list(zip(wall_index, wall)))
            opp2_handcards, wall, RT2 = self.simulate_handcards(opp2_hu_prefer, self.discardsOp2, self.discards2, wall,
                                                                sum_xts_opps[1], self.fei_king2)
            # print('--------opp2:----------')
            # print('opp2 hu_prefer:', opp2_hu_prefer)
            # print('opp2:', opp2_handcards, 'opp2_suites:', self.discardsOp2)
            # print('opp2:', [MJ.translate16_10(c) for c in opp2_handcards], 'opp2_suites:', [
            #     [MJ.translate16_10(c) for c in suit] for suit in self.discardsOp2])
            # print('wall after2:', list(zip(wall_index, wall)))
            # print('opp3 hu type:', opp3_hu_prefer)
            opp3_handcards, wall, RT3 = self.simulate_handcards(opp3_hu_prefer, self.discardsOp3, self.discards3, wall,
                                                                sum_xts_opps[2], self.fei_king3)
            # print('--------opp3:----------')
            # print('opp3 hu_prefer:', opp3_hu_prefer)
            # print('opp3:', opp3_handcards, 'opp3_suites:', self.discardsOp3)
            # print('opp3:', [MJ.translate16_10(c) for c in opp3_handcards], 'opp3_suites:', [
            #     [MJ.translate16_10(c) for c in suit] for suit in self.discardsOp3])
            # print('wall after3:', list(zip(wall_index, wall)))

            # print('opp1:', opp1_handcards)
            # print('opp2:', opp2_handcards)
            # print('opp3:', opp3_handcards)
            # 累加本次分配的值wall_sum和RT_sum
            # 给wallsum赋值
            # 给RT赋值
            for i in range(len(wall)):
                wall_sum[i] += wall[i]
                for j in range(2):
                    RT1_sum[j][i] += RT1[j][i]
                    RT2_sum[j][i] += RT2[j][i]
                    RT3_sum[j][i] += RT3[j][i]

        # 求M次模拟的平均值
        s_w = sum(wall_sum)
        T_selfmo = [0] * 34
        for i in range(len(wall_sum)):
            T_selfmo[i] = max(0, float(wall_sum[i]) / s_w)
            wall_pred[i] =max(0, float(wall_sum[i]) /self.M)
            for j in range(2):
                RT1_sum[j][i] = float(RT1_sum[j][i]) / self.M
                RT2_sum[j][i] = float(RT2_sum[j][i]) / self.M
                RT3_sum[j][i] = float(RT3_sum[j][i]) / self.M

        return wall_pred, T_selfmo, RT1_sum, RT2_sum, RT3_sum
                
        
# if __name__ == '__main__':
#
#     request = {"discards": [[55, 51, 40, 49], [52, 55, 49, 9, 4, 54, 22], [55, 41, 53, 38, 24, 41, 36, 22], [3, 49, 53, 17, 50, 9, 38]], "discards_real": [[55, 51, 2, 35, 36, 40, 49, 5], [52, 55, 49, 34, 9, 4, 1, 54, 22], [55, 41, 53, 38, 24, 41, 36, 22], [18, 3, 24, 49, 51, 53, 17, 50, 9, 38]], "discards_op": [
#         [[17, 18, 19], [22, 23, 24]], [[35, 35, 35], [51, 51, 51], [36, 37, 38], [5, 6, 7]], [[33, 34, 35]], [[2, 2, 2], [1, 1, 1, 1]]], "king_card": 25, "user_cards": {"hand_cards": [7, 8, 9, 37, 37, 39, 40, 34], "operate_cards": [[17, 18, 19], [22, 23, 24]]}, "catch_card": 34, "seat_id": 0, "round": 9, "fei_king": 0}
#
#     user_cards = request.get('user_cards', {})
#     catch_card = request.get('catch_card', 0)
#     king_card = request.get('king_card', 0)
#     fei_king = request.get('fei_king', 0)
#     seat_id = request.get('seat_id', 0)
#     discards = request.get('discards', [])
#     discards_op = request.get('discards_op', [])
#     remain_num = request.get('remain_num', 136)
#
#     round = request.get('round', [])
#     hand_cards = user_cards.get('hand_cards', [])
#     operate_cards = user_cards.get('operate_cards', [])
#     hand_cards.sort()
#     [e.sort() for e in operate_cards]
#     start_time = time.time()
#     DFM = DefendModel(cards=hand_cards, suits=operate_cards, king_card=king_card, fei_king=fei_king,
#                       discards=discards, discardsOp=discards_op, discardsReal=discards, round=round,
#                       seat_id=seat_id, xts_round=xts_round, M=250)
#     wall_pred, T_selfmo, RT1, RT2, RT3 = DFM.getWTandRT()
#     print('cost time:',time.time() - start_time)
