# -*- coding:utf-8 -*-
# 不同牌型向听数计算
import copy
from mah_tool.so_lib.sr_xt_ph import pinghu


# 1 平胡向听数  任航师兄版本
def wait_types_comm_king(tile_list, suits, jing_card=0):
    xt_ph = pinghu(tile_list, suits, jing_card).get_xts()
    # 向听数此时为0，为胡牌情况
    if xt_ph == 0:
        return 0
    elif jing_card not in tile_list: # 没有宝牌 ====
        return xt_ph
    else:
        # 当xt_ph>0时，宝又在手牌中，需要考虑宝还原的情况
        xt_ph_no_king = pinghu(tile_list, suits, 0).get_xts()
        return min(xt_ph, xt_ph_no_king) # 返回有宝牌和没宝牌时向听数的最小值===

'''
    计算七对向听数，进行深拷贝，将深拷贝的宝牌移除，如果有副露则不是七对，对深拷贝进行排序并初始化向听数为7，
    如果其中某张牌的数量大于2则向听数-1，最后返回向听数减去宝牌数后的值和0的较大者
'''
# 2 七对的向听数判断
def wait_types_7(tile_list, suits=[], jing_card=0):
    _tile_list = copy.deepcopy(tile_list) # 深拷贝===

    jing_count = _tile_list.count(jing_card) # 获得精牌数===
    for i in range(jing_count):
        _tile_list.remove(jing_card)

    if suits != []:
        wait_num = 7  # 如果副露有牌，则不能做七对
        return wait_num
    else:
        wait_num = 7  # 表示向听数
        _tile_list.sort()  # L是临时变量，传递tile_list的值
        L = set(_tile_list)
        for i in L:
            # print("the %d has %d in list" % (i, tile_list.count(i)))
            if _tile_list.count(i) >= 2:
                wait_num -= 1

        return max(0, wait_num - jing_count)


# 返回去精牌后的手牌，四个牌的数量，三个牌数量，两个牌数量，精牌数量
def get_four_three_two_card_jing_nums(tile_list, jing_card=0):
    _tile_list = copy.deepcopy(tile_list)
    jing_count = _tile_list.count(jing_card)

    for i in range(jing_count):
        _tile_list.remove(jing_card)

    si_card_num = 0
    san_card_num = 0
    er_card_num = 0
    L = list(set(_tile_list))
    L.sort(key=_tile_list.index)

    for i in L:
        _count = _tile_list.count(i)
        if _count == 4:
            si_card_num += 1
        if _count == 3:
            san_card_num += 1
        if _count == 2:
            er_card_num += 1

    return _tile_list, si_card_num, san_card_num, er_card_num, jing_count


# 2-2 豪华七对的向听数判断
def wait_types_haohua7(tile_list, suits=[], jing_card=0):
    _tile_list = copy.deepcopy(tile_list)

    if len(suits) > 0 or len(_tile_list) != 14:  # 当副露不为空时,不是七对
        return 7

    wait_nums = 7
    _tile_list, si_card_num, san_card_num, er_card_num, jing_count = get_four_three_two_card_jing_nums(_tile_list,
                                                                                                       jing_card)
    wait_nums -= (si_card_num * 2 + san_card_num + er_card_num)  # 减去向听数

    signal_nums = len(_tile_list) - si_card_num * 4 - san_card_num * 3 - er_card_num * 2 + max((san_card_num - 1),
                                                                                               0) + jing_count  # 精牌也算单张

    # 如果没有四个相同的牌，需要增加向听数
    if si_card_num == 0:
        if san_card_num == 0:  # 只有aa， 需要增加1个向听
            if signal_nums < 2:  # 单张不满足2张，需要拆对， 向听+1
                wait_nums += 2
            else:
                wait_nums += 1
        else:  # 有刻子时
            if signal_nums < 1:  # 也需要拆对 +1
                wait_nums += 1
    return max(0, wait_nums - jing_count)


# 3 十三烂的向听数判断
def wait_types_13(tile_list, suits=[], jing_card=0):  # 十三烂中仅作宝还原
    # 十三烂的向听数判断，手中十四张牌中，序数牌间隔大于等于3，字牌没有重复所组成的牌形
    # 先计算0x0,0x1,0x2中的牌，起始位a，则a+3最多有几个，在wait上减，0x3计算不重复最多的数
    wait_13lan = {
        'thirteen_waiting0': 0,
        'thirteen_waiting1': 0,
        'thirteen_waiting2': 0,
        'thirteen_waiting3': 0,
        'thirteen_waiting4': 0,
        'thirteen_waiting5': 0,
        'thirteen_waiting6': 0,
        'thirteen_waiting7': 0,
        'thirteen_waiting8': 0,
        'thirteen_waiting9': 0,
        'thirteen_waiting10': 0,
        'thirteen_waiting11': 0,
        'thirteen_waiting12': 0,
        'thirteen_waiting13': 0,
        'thirteen_waiting14': 0,
    }
    wait_num = 14  # 表示向听数
    max_num_wait = 0
    if suits != []:
        wait_num = 14
        return wait_num
    else:
        L = set(tile_list)  # 去除重复手牌
        L_num0 = []  # 万数牌
        L_num1 = []  # 条数牌
        L_num2 = []  # 筒数牌
        for i in L:
            if i & 0xf0 == 0x30:
                # 计算字牌的向听数
                wait_num -= 1
            if i & 0xf0 == 0x00:
                L_num0.append(i & 0x0f)
            if i & 0xf0 == 0x10:
                L_num1.append(i & 0x0f)
            if i & 0xf0 == 0x20:
                L_num2.append(i & 0x0f)
        wait_num -= calculate_13(L_num0)
        # 减去万数牌的向听数
        wait_num -= calculate_13(L_num1)
        # 减去条数牌的向听数
        wait_num -= calculate_13(L_num2)
        # 减去筒数牌的向听数
        # print(L)
        # print(L_num0)
        # print(L_num1)
        # print(L_num2)
        # print(wait_num)
        wait_13lan['thirteen_waiting' + str(wait_num)] = 1
        # print(wait_13lan)
        return wait_num


# 4 九幺的向听数判断
def wait_types_19(tile_list, suits, jing_card=0):
    # 九幺的向听数判断，由一、九这些边牌、东、西、南、北、中、发、白这些风字牌中的任意牌组成的牌形。以上这些牌可以重复
    wait_19 = {
        'one_nine_waiting0': 0,
        'one_nine_waiting1': 0,
        'one_nine_waiting2': 0,
        'one_nine_waiting3': 0,
        'one_nine_waiting4': 0,
        'one_nine_waiting5': 0,
        'one_nine_waiting6': 0,
        'one_nine_waiting7': 0,
        'one_nine_waiting8': 0,
        'one_nine_waiting9': 0,
        'one_nine_waiting10': 0,
        'one_nine_waiting11': 0,
        'one_nine_waiting12': 0,
        'one_nine_waiting13': 0,
        'one_nine_waiting14': 0,
    }
    wait_num = 14  # 表示向听数
    _suits = copy.deepcopy(suits)
    for i in _suits:
        if i[0] != i[1]:
            return 14
        else:
            if i[0] & 0xf0 == 0x30 or i[0] & 0x0f == 0x01 or i[0] & 0x0f == 0x09:
                wait_num -= 3
            else:
                return 14  # 如果非1和9及字牌的刻子

    for i in tile_list:
        if i & 0x0f == 0x01 or i & 0x0f == 0x09 or i & 0xf0 == 0x30:
            wait_num -= 1
    wait_19['one_nine_waiting' + str(wait_num)] = 1
    # print(wait_19)
    return wait_num


def calculate_13(tiles):
    # 计算十三浪的数牌最大向听数
    if len(tiles) == 0:
        return 0
    if len(tiles) == 1:
        return 1
    if len(tiles) == 2:
        if tiles[0] + 3 <= tiles[1]:
            return 2
        else:
            return 1
    if len(tiles) >= 3:
        return max((tiles.count(1) + tiles.count(4) + tiles.count(7)),
                   (tiles.count(1) + tiles.count(4) + tiles.count(8)),
                   (tiles.count(1) + tiles.count(4) + tiles.count(9)),
                   (tiles.count(1) + tiles.count(5) + tiles.count(8)),
                   (tiles.count(1) + tiles.count(5) + tiles.count(9)),
                   (tiles.count(1) + tiles.count(6) + tiles.count(9)),
                   (tiles.count(2) + tiles.count(5) + tiles.count(8)),
                   (tiles.count(2) + tiles.count(5) + tiles.count(9)),
                   (tiles.count(2) + tiles.count(6) + tiles.count(9)),
                   (tiles.count(3) + tiles.count(6) + tiles.count(9)))
