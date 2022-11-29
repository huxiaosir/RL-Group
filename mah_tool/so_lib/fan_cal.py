# -*- coding:utf-8 -*-
from mah_tool.so_lib.sr_xt_ph import pinghu


# 计算可能的番型
# [清一色、门清、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]

class FanList(object):
    def __init__(self, choosePaiXing=0, handcards=[], suits=[], jingCard=0, feiking_num=0):
        '''
        :param choosePaiXing:  the index of [平胡  碰碰胡  九幺　七对 十三烂]
        :param handcards:
        :param suits:
        :param jingCard:
        '''
        self.choosePaiXing = choosePaiXing
        self.handcards = handcards
        self.suits = suits
        self.jingCard = jingCard
        self.all_suits_cards = self.__merge_suits()
        self.feiking_num = feiking_num

    def __merge_suits(self):
        all_suits_cards = []
        for suit in self.suits:
            all_suits_cards.extend(suit)
        return all_suits_cards

    def isQingYiSe(self):
        w = 0
        ti = 0
        to = 0
        z = 0
        for card in self.handcards + self.all_suits_cards:
            if card & 0xf0 == 0x00:
                w = 1
            if card & 0xf0 == 0x10:
                ti = 1
            if card & 0xf0 == 0x20:
                to = 1
            if card & 0xf0 == 0x30:
                z = 1
        if w + ti + to + z <= 1:
            return True
        return False

    def isMenQing(self):
        return len(self.suits) == 0

    def isBaoDiao(self):
        if self.jingCard in self.handcards:
            if self.choosePaiXing <= 1:
                return pinghu(self.handcards, self.suits, self.jingCard).get_xts() < \
                       pinghu(self.handcards, self.suits, 0).get_xts()
            else:
                return True
        else:
            return False

    def isBaoHuanYuan(self):
        if self.jingCard in self.handcards:
            return pinghu(self.handcards, self.suits, self.jingCard).get_xts() == \
                   pinghu(self.handcards, self.suits, 0).get_xts()
        else:
            return False

    def isDanDiao(self):
        return len(self.suits) == 4

    def isQingXing(self):
        d, n, x, b, z, f, b = 0, 0, 0, 0, 0, 0, 0
        for card in self.handcards + self.all_suits_cards:
            if card & 0xf0 == 0x30:
                if card & 0x0f == 1:
                    d = 1
                elif card & 0x0f == 2:
                    n = 1
                elif card & 0x0f == 3:
                    x = 1
                elif card & 0x0f == 4:
                    b = 1
                elif card & 0x0f == 5:
                    z = 1
                elif card & 0x0f == 6:
                    f = 1
                else:
                    b = 1
        return d + n + x + b + z + f + b == 7

    def getFanList(self):
        '''
        return the may fans base on the choosePaiXing
        :return:
        '''
        # [清一色、门清、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        # choosePaiXing[平胡  碰碰胡  九幺　七对 十三烂]
        fanList = [0] * 10

        if self.isQingYiSe():  fanList[0] = 1
        if self.isMenQing(): fanList[1] = 1

        if self.choosePaiXing <= 1:  # pinghu or pengpenghu
            if self.choosePaiXing == 1:  # 平胡宝吊不翻倍
                if self.isBaoDiao(): fanList[2] = 1  # baodiao
            if self.isBaoHuanYuan(): fanList[3] = 1
            if self.isDanDiao(): fanList[4] = 1

        if self.choosePaiXing == 3:  # qidui
            if self.isBaoDiao(): fanList[2] = 1  # baodiao

        if self.choosePaiXing == 2 or self.choosePaiXing == 4:  # jiuyao or ssl
            if self.isQingXing(): fanList[5] = 1

        if self.feiking_num > 0:
            for i in range(self.feiking_num):
                fanList[6 + i] = 1

        return fanList

# if __name__ == '__main__':
#     # test
#     fan_list= Fan(1,[1, 2, 3, 6, 35],[[36, 36, 36], [29, 29, 29], [9, 9, 9]],35,2).getFanList()
#     print(fan_list)
