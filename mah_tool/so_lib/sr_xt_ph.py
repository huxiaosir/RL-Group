# -*- coding: utf-8 -*-
import copy


class pinghu:
    def __init__(self, cards, suits, kingCard):
        cards.sort()
        self.cards = cards
        self.suits = suits

        self.kingCard = kingCard

        if kingCard != None:
            self.kingNum = cards.count(kingCard)
        else:
            self.kingNum = 0

    # 花色分离，输出为原来的牌
    def split_type_s(self, cards):
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

    def get_32N(self, cards):
        cards.sort()
        kz = []
        sz = []
        aa = []
        ab = []
        ac = []
        lastCard = 0
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
    def in_cards(self, t32=[], cards=[]):
        for card in t32:
            if card not in cards:
                return False
        return True

    # 生成所有的分类情况
    def extract_32N(self, cards=[], t32_branch=[], t32_set=[]):
        t32N = self.get_32N(cards=cards)

        if len(t32N) == 0:
            t32_set.extend(t32_branch)
            # t32_set.extend([cards])
            t32_set.append(0)
            t32_set.extend([cards])
            return t32_set
        else:
            for t32 in t32N:
                if self.in_cards(t32=t32, cards=cards):
                    cards_r = copy.copy(cards)
                    for card in t32:
                        cards_r.remove(card)
                    t32_branch.append(t32)
                    self.extract_32N(cards=cards_r, t32_branch=t32_branch, t32_set=t32_set)
                    if len(t32_branch) >= 1:
                        t32_branch.pop(-1)

    # 计算代价
    '''
    sub=[[]]*4
    sub[0] kz
    sub[1] sz
    sub[2] aa
    sub[3] ２N
    sub[4] 得分
    sub[5] 废牌
    搜索理论一：如果同一分支的废牌（保留牌）一致，那么他们属于同一分支的不同分割类型，其有效牌、废牌共用
    '''

    def tree_expand(self, cards):
        all = []  # 全部的情况
        t32_set = []
        self.extract_32N(cards=cards, t32_branch=[], t32_set=t32_set)
        # print ('tree expand',t32_set)
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
                        sz.append(t)
                    # print (sub)
                elif len(t) == 2:
                    if t[1] == t[0]:
                        aa.append(t)
                    else:
                        t2N.append(t)

            else:
                '修改，使计算时间缩短'
                leftCards = t32_set[i + 1]
                efc_cards = self.get_effective_cards(dz_set=t2N)  # t2N中不包含ａａ
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

    # 获取有效牌,输入为搭子集合,
    def get_effective_cards(self, dz_set=[]):
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

    def zi_expand(self, cards):
        cardList = []
        for i in range(7):
            cardList.append([])
        ziCards = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37]
        for card in ziCards:
            index = (card & 0x0f) - 1
            # print(index)

            if cards.count(card) == 4:
                # 此结构为[3N,2N,leftCards]
                cardList[index].append([[[card, card, card]], [], [], [], 0, [card]])
            elif cards.count(card) == 3:
                cardList[index].append([[[card, card, card]], [], [], [], 0, []])
                cardList[index].append([[], [], [[card, card]], [], 0, [card]])
            elif cards.count(card) == 2:

                cardList[index].append([[], [], [[card, card]], [], 0, []])
            elif cards.count(card) == 1:
                cardList[index].append([[], [], [], [], 0, [card]])
            else:
                cardList[index].append([[], [], [], [], 0, []])
            # print(index,cardList[index],card)

        ziBranch = []
        for c1 in cardList[0]:
            for c2 in cardList[1]:
                for c3 in cardList[2]:
                    for c4 in cardList[3]:
                        for c5 in cardList[4]:
                            for c6 in cardList[5]:
                                for c7 in cardList[6]:
                                    branch = []
                                    for n in range(6):
                                        branch.append(c1[n] + c2[n] + c3[n] + c4[n] + c5[n] + c6[n] + c7[n])
                                    ziBranch.append(branch)

        # print('ziBranch',ziBranch)
        return ziBranch

    def get_xts_info(self, type="ph"):
        """
                get_xts_information
                :return:
                """
        cards = self.cards
        suits = self.suits
        kingCard = self.kingCard

        if kingCard == None:
            kingCard = self.kingCard
        outOfKingCards = copy.copy(cards)
        kingNum = 0
        if kingCard != None:
            kingNum = cards.count(kingCard)
            for i in range(kingNum):
                outOfKingCards.remove(kingCard)

        # 花色分离
        cards_wan, cards_tiao, cards_tong, cards_zi = self.split_type_s(outOfKingCards)
        wan_expd = self.tree_expand(cards=cards_wan)
        tiao_expd = self.tree_expand(cards=cards_tiao)
        tong_expd = self.tree_expand(cards=cards_tong)
        zi_expd = self.zi_expand(cards=cards_zi)

        all = []
        for i in wan_expd:
            for j in tiao_expd:
                for k in tong_expd:
                    for m in zi_expd:
                        branch = []
                        # 将每种花色的4个字段合并成一个字段
                        for n in range(6):
                            branch.append(i[n] + j[n] + k[n] + m[n])
                        all.append(branch)

        for i in range(len(all)):
            t3N = all[i][0] + all[i][1]
            all[i][4] = 14 - (len(t3N) + len(suits)) * 3
            # 有将牌
            has_aa = False
            if len(all[i][2]) > 0:
                has_aa = True

            if has_aa and kingNum == 0:  # has do 当２Ｎ与３Ｎ数量小于4时，存在没有减去相应待填数，即废牌也会有１张作为２Ｎ或３Ｎ的待选位,
                # print()all_src
                if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1 >= 4:

                    all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 2
                else:
                    all[i][4] -= (len(all[i][2]) + len(all[i][3]) - 1) * 2 + 2 + 4 - (
                            len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) - 1)  # 0717 17:24
            # 无将牌
            else:
                if len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]) >= 4:

                    all[i][4] -= (4 - (len(suits) + len(t3N))) * 2 + 1

                else:
                    all[i][4] -= (len(all[i][2]) + len(all[i][3])) * 2 + 1 + 4 - (
                            len(suits) + len(t3N) + len(all[i][2]) + len(all[i][3]))
            all[i][4] -= kingNum
            if all[i][4] < 1:
                # print(all[i][4])
                all[i][4] = 0
        if type == "pph":
            all.sort(key=lambda k: (-len(k[0]), -len(k[2]), k[4], len(k[-1])))
        else:
            all.sort(key=lambda k: (k[4], len(k[-1])))
        return all[0]

    def get_xts(self, type="ph"):
        all_info = self.get_xts_info(type)
        efc_kingNum = self.kingNum
        if type == "pph":  # 碰碰胡
            if len(self.suits) != 0:
                for suit in self.suits:
                    if suit[0] != suit[1]:
                        return 14

            kz = len(all_info[0])
            aa = len(all_info[2])
            if aa + kz >= 5:
                pph_xt = 9 - 2 * kz - (5 - kz)
                if efc_kingNum > 0:
                    efc_kingNum -= 1
            else:
                pph_xt = 9 - 2 * kz - aa
            return max(0, pph_xt - efc_kingNum)

        return all_info[4]


# if __name__ == '__main__':
#     print(pinghu([1, 2, 3, ], [[7, 8, 9], [20, 20, 20], [18, 19, 20]], 6).get_xts())
#     print(pinghu([1, 5, 6, 7, 20, 21, 22], [[7, 8, 9], [18, 19, 20]], 1).get_xts())
#     import time
#     t1 = time.time()
#     print (pinghu([1, 1,  2, 3, 4,  6, 7, 8,  7, 8, 9,  7, 8, 9],[], 6).get_xts())
#     print (pinghu([1, 1,  2, 3, 4,  6, 7, 8,  17, 18, 19,  37, 38, 39],[], 0).get_xts())
#
#     print(pinghu([1, 2,  3, 3, 3,  5, 7, 8,  17, 18, 19,  37, 38, 39], [], 0).get_xts_info())
#     print(pinghu([1, 1, 1,2,2,2, 3, 3, 3,  4, 5, 6,  17,18], [], 17).get_xts_pph())
#     print(pinghu([2, 3,3, 35, 35,35, 36, 37, 37, 39,  40, 40,40], [], 35).get_xts("pph"))
#     print(pinghu([1, 1,  3, 3, 3,  6, 7, 8,  17, 18, 19,  37, 38, 39], [], 1).get_32N())
#     print(time.time() - t1)
