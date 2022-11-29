import os
from mah_tool.so_lib import shangraoMJ_v2, sr_xt_ph
from mah_tool.so_lib.fan_cal import FanList

current_path = os.path.dirname(os.path.abspath(__file__))  # 返回当前文件所在的目录
"""
        pinghu
        类变量初始化
        :param cards: 手牌　
        :param suits:副露
        :param leftNum:剩余牌数量列表
        :param discards:弃牌
        :param discards_real:实际弃牌
        :param discardsOp:场面副露
        :param round:轮数
        :param remainNum:牌墙剩余牌数量
        :param seat_id:座位号
        :param kingCard:宝牌
        :param fei_king:飞宝数
        :param op_card:动作操作牌
"""


class SearchInfo(object):

    @staticmethod
    def getSearchInfo(cards=[], suits=[], king_card=None, discards=[], discards_op=[], fei_king=0, remain_num=136,
                      round=0, seat_id=0):

        result = shangraoMJ_v2.recommend_card_rf(cards=cards, suits=suits, king_card=king_card, discards=discards,
                                                 discards_op=discards_op, fei_king=fei_king, remain_num=remain_num,
                                                 round=round, seat_id=seat_id)

        # [平胡  碰碰胡 九幺　七对 十三烂]
        paixing = result[0]

        def getSzKzInSuits(suits):
            kz = 0
            sz = 0
            for suit in suits:
                if suit[0] == suit[1]:
                    kz += 1
                else:
                    sz += 1
            return kz, sz

        if paixing == 0:  # pinghu
            pinghu_info = sr_xt_ph.pinghu(cards, suits, king_card).get_xts_info()
            kz, sz = getSzKzInSuits(suits)
            if sz == 0:
                if (len(pinghu_info[0]) + kz) >= 3 and len(pinghu_info[1]) == 0:  # choose pengpengHu
                    paixing = 1  #
        else:
            paixing += 1

        # [清一色、门清、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        fanList = FanList(paixing, cards, suits, king_card, fei_king).getFanList()

        return paixing, fanList


if __name__ == '__main__':
    result = SearchInfo.getSearchInfo([3, 3, 3, 6, 7], [[36, 36, 36], [29, 29, 29], [9, 9, 9]], 2)
    # result = SearchInfo.getSearchInfo([1, 1, 2, 51, 52, 53, 54, 55], [[49, 49, 49], [50, 50, 50]], 2, fei_king=3)
    print(result)
