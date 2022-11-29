# -*- coding:utf-8 -*-
# 特征编码（常规，418*34*1）
import copy
import numpy as np
# from feature.extract import get_param
from mah_tool.suphx_extract_features import tool
from mah_tool.suphx_extract_features.search_tree_ren import SearchInfo
import torch


# 模仿suphx对牌进行编码
def suphx_cards_feature_code(cards_, channels):
    '''
    对牌集进行特征编码
    :param cards_:  牌或者牌集
    :param channels: 通道数
    :return:
    '''
    cards = copy.deepcopy(cards_)    #深拷贝，防止修改原本的数据

    if not isinstance(cards, list):  # 如果是一张牌，则不是一个list   如果只有一张牌 则不是list  需要将这张牌也转换成list
        cards = [cards]

    features = []
    for channel in range(channels):
        # #遍历所有的通道数，对于每个通道，将手牌中的每一种牌将feature中的该牌的相应位置置为1，然后在cards中去掉该牌
        # #最终返回channnels * 34
        # 去重，编写为channels×34的样式
        S = set(cards)      #将手牌先去重，每次从这个集合中拿一张牌去编码，之后在原来的手牌中去掉这张牌
        feature = [0] * 34
        for card in S:
            card_index = tool.translate3(card) # 将牌转换成 0~33
            cards.remove(card)
            feature[card_index] = 1 # 将feature中card对应的位置(0~33) 置为1
        features.append(feature)
    return features


def suphx_data_feature_code(data, channels=4, data_type="cards_set"):
    '''
    返回对数据按数据类型编码的特征
    :param data: 数据
    :param channel： 通道数
    :param type: 数据类型  optional ["cards_set", "seq_discards", "dummy"]
    :return:
    '''

    # cards 为16进制
    data_copy = copy.deepcopy(data)
    features = []
    if data_type == "cards_set":
        features.extend(suphx_cards_feature_code(data_copy, channels))
    elif data_type == "seq_discards":
        seq_discards_features = []  # 弃牌的features,四个玩家的弃牌顺序，
        seq_len = 30  # 每个玩家弃牌的最大手数为30手
        for player_discard_seq in data_copy:
            cur_seq_discards_features = []  # 当前玩家的弃牌序列
            for i in range(len(player_discard_seq)):
                cur_seq_discards_features.extend(suphx_cards_feature_code(player_discard_seq[i], channels))

            seq_discards_features.extend(cur_seq_discards_features)  # 把当前已有的序列添加到features中
            need_pad_len = seq_len - len(cur_seq_discards_features)  # 需要填充的长度  填充成30*channels*34

            pad_features = [[0] * 34 for _ in range(need_pad_len)]
            seq_discards_features.extend(pad_features)  # 每个玩家的弃牌都是 30*channels * 34
        features.extend(seq_discards_features) # features是一个 4*30*channels * 34
    elif data_type == "dummy":  # 哑变量编码  此时的data为整数
        assert isinstance(data_copy, int)
        dummy_features = [[0] * 34 for _ in range(channels)] # dummy_features是一个 有channels行 每一行都是一个含有34个0的list
        if 0 < data_copy <= channels:
            dummy_features[data_copy - 1] = [1] * 34
        elif data_copy == 0:
            # pass  当为0时，哑变量全为零
            pass
        else:
            print("INFO[ERROR]")
        features.extend(dummy_features) # channels * 34
    elif data_type == "look_ahead":  # 暂时空着
        pass

    return features


def calculate_king_sys_suphx(handCards0, fulu_, king_card,
                             discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
                             dealer_flag, search=False):
    '''

    返回不加前瞻特征及隐藏特征的特征
    :param state: 集成的状态信息
    :param seat_id: agent的座位id
    :param search: 开启前瞻搜索特征
    :param global_state: 是否编码隐藏信息特征
    :param dropout_prob: 对隐藏信息特征的dropout的概率
    :return:
    '''

    # 所有特征
    # features = [[0]*34 for _ in range(419)]
    features = []
    # 手牌特征
    handcards_features = suphx_data_feature_code(handCards0, 4) # 高手玩家手牌的特征 是一个 4 * 34
    features.extend(handcards_features)

    # 副露特征
    fulu_features = []
    for fulu in fulu_: # 副露是一个[[[],[]],[]]
        action_features = []
        fulu_len = len(fulu)  # 当前玩家副露的长度
        for action in fulu: # 当前玩家副露里的每个组合
            action_features.extend(suphx_data_feature_code(action, 4)) # 动作的特征是一个4 * 34
        # 需要padding  副露的长度最大为4
        action_padding_features = [[0] * 34 for _ in range(4) for _ in range(4 - fulu_len)]
        action_features.extend(action_padding_features) # 填充完成后 每个玩家的action_features 是一个4 * 4 * 34

        fulu_features.extend(action_features)
    features.extend(fulu_features) # fulu_features 是一个4 * 4 * 4 * 34

    # 宝牌特征
    king_features = suphx_data_feature_code(king_card, 1) # 宝牌特征是一个 1 * 34
    features.extend(king_features)

    # 隐藏信息特征

    # 所有弃牌的顺序信息
    seq_discards_features = suphx_data_feature_code(discards_seq, 1, data_type="seq_discards") # 所有玩家弃牌的特征是 4 * 30 * 1 * 34
    features.extend(seq_discards_features)

    # 剩余牌数特征
    remian_cardsnums_features = suphx_data_feature_code(remain_card_num, 120, data_type="dummy") # 剩余牌特征是一个120 * 34 且第remain_card_num行全为1
    features.extend(remian_cardsnums_features)

    # 自己拥有的宝牌数
    self_king_num_features = suphx_data_feature_code(self_king_num, 4, data_type="dummy") # 特征是一个 4 * 34 且第self_king_num行全为1
    features.extend(self_king_num_features)

    # 所有玩家飞宝数
    all_palyer_fei_king_num_features = []
    for fei_king_num in fei_king_nums:
        all_palyer_fei_king_num_features.extend(suphx_data_feature_code(fei_king_num, 4, data_type="dummy")) # 特征为4 * 4 * 34
    features.extend(all_palyer_fei_king_num_features)

    # 当前手数
    cur_round_features = suphx_data_feature_code(round_, 30, data_type="dummy") # 当前手数特征为一个 30 * 34
    features.extend(cur_round_features)

    # 庄家特征
    dealer_features = []
    for flag in dealer_flag:
        dealer_features.extend(suphx_data_feature_code(flag, 1, data_type="dummy")) # 庄家特征是一个 4 * 34
    features.extend(dealer_features)

    #开启搜索特征
    search_features = [[0] * 34 for _ in range(55)] # 搜索特征是一个 55 * 34
    if search:
        # paixing -> [平胡  碰碰胡 九幺　七对 十三烂]
        # fanList -> [清一色、门清、宝吊、宝还原、单吊、七星、飞宝1、飞宝2、飞宝3、飞宝4]
        paixing, fanList = SearchInfo.getSearchInfo(handCards0, fulu_[0], king_card, discards_seq, fulu_,
                                                    fei_king_nums[0], remain_card_num, round_ - 1, 0)
        search_features[paixing * 11] = [1] * 34
        for fan_index in range(len(fanList)):
            if fanList[fan_index] == 1:
                search_features[paixing * 11 + 1 + fan_index] = [1] * 34
    features.extend(search_features)

    return torch.tensor(features, dtype=torch.float).reshape(418, 34, 1)


# zengw 20.11.11
def card_preprocess_sr_suphx(handCards0, fulu_, king_card, discards_seq, remain_card_num,
                             self_king_num, fei_king_nums, round_, dealer_flag=[1, 0, 0, 0],
                             search=True):
    '''
    上饶麻将特征提取,模仿suphx
    说明:
    1.牌都是用16进制进行表示，参数需要预先处理好
    2.当前玩家的位置放在第一位  eg  当前玩家座位为0时:[0,1,2,3], 当前玩家座位为1时:[1,2,3,0], 当前玩家座位为2时:[2,3,0,1], ..[3,2,1,0]
    :param handCards0: 当前要编码玩家的手牌 -> [] 1维list ==先知道高手玩家的位置，再根据位置获取handcards中的列表==
    :param fulu_: 四个玩家的副露 -> [[[7,8,9],[17,17,17]], [], [], []] 3维list   位置参考说明2   ==discards_op==
    :param king_card:  宝牌 -> 1 int  ==king_card==
    :param all_player_handcards:四个玩家的手牌 -> [[],[],[],[]]  2维list   位置参考说明2  当后面三个玩家为空时，隐藏完美信息 ==根据当前玩家位置进行调整==
    :param card_library:  牌库的牌 -> [] 1维list 当为空时，隐藏完美信息
    :param all_palyer_king_nums: 四个玩家手中的宝牌数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2  当后面三个玩家为0时，隐藏完美信息
    :param discards_seq:  四个玩家真实弃牌顺序-> [[], [], [], []] 2维list   位置参考说明2
    :param remain_card_num: 牌墙剩余牌 -> int
    :param self_king_num: 当前玩家的宝牌数 -> int
    :param fei_king_nums: 所有玩家的飞宝数 -> [0,0,0,0] 长度为4的一维list 位置参考说明2
    :param round_: 当前轮（手）数 -> int
    :param dealer_flag: 庄家flag，默认当前玩家为庄家 -> [1,0,0,0]
    :param search: 是否采用搜索树 默认开启
    :param global_state:是否开启隐藏信息特征，默认关闭
    :param dropout_prob: 对隐藏信息的dropout率，默认为0
    :return: 编码好的三维特征 455×34×1
    '''

    features = calculate_king_sys_suphx(handCards0, fulu_, king_card, discards_seq, remain_card_num, self_king_num,
                                        fei_king_nums, round_, dealer_flag, search)

    features = np.array(features)
    features = features.T
    features = np.expand_dims(features, 0)
    features = features.transpose([2, 1, 0])  # 更换位置  转换成c × 34 × 1的格式

    return features.tolist()

# if __name__ == '__main__':
#     # test
#     path = '../original_dataset/chi_dataset/13.json'
#     feature = card_preprocess_sr_suphx(*get_param(path))
#     print(np.array(feature).shape)
#     print(feature)

# if __name__ == '__main__':
# 	# test
# 	handCards0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19]
# 	fulu_ = [[[18, 18, 18]], [], [[41, 41, 41], [20, 20, 20], [21, 22, 23]], []]
# 	king_card = 5
# 	all_player_handcards = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 19], [2, 2, 2, 6, 6, 6, 7, 8, 9, 49, 49, 50, 51, 52],
# 							[33, 35, 38, 39, 40], []]
# 	card_library = [53, 53, 2, 2, 9, 9, 9]
# 	all_palyer_king_nums = [0, 0, 0, 0]
# 	discards_seq = [[2, 3], [3, 2], [1, 4], [4, 1]]
# 	remain_card_num = 83
# 	self_king_num = 0
# 	fei_king_nums = [1, 0, 0, 0]
# 	round_ = 2
#
# 	discards_real_list = [2, 3, 1, 4, 3, 2, 4, 1]
# 	dealer_flag = [1, 0, 0, 0]
#
# 	featrues = card_preprocess_sr_suphx(handCards0, fulu_, king_card, all_player_handcards, card_library,
# 										all_palyer_king_nums,
# 										discards_seq, remain_card_num, self_king_num, fei_king_nums, round_,
# 										dealer_flag)
# 	print(featrues.shape)
