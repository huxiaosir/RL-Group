#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 17:02
# @Author  : Joisen
# @File    : test.py
from mah_tool.suphx_extract_features import tool
import numpy as np
if __name__ == '__main__':
    # card = 55
    # card_index = tool.translate3(card)
    # print(card_index)
    feature = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # feature = [[0] * 9] * 4
    # feature[1][1] = 1
    # print(feature)
    handCards = [
        [
          2,
          4,
          5,
          5,
          17,
          19,
          23,
          25,
          25,
          33,
          34,
          37,
          51
        ],
        [
          1,
          8,
          8,
          9,
          22,
          22,
          22,
          36,
          41,
          49,
          50,
          54,
          55
        ],
        [
          7,
          7,
          17,
          20,
          21,
          23,
          24,
          24,
          38,
          39,
          49,
          52,
          52
        ],
        [
          3,
          4,
          18,
          19,
          20,
          21,
          23,
          23,
          34,
          36,
          36,
          38,
          40
        ]
      ]
    # handCard = [x for a in handCards for x in a]
    # print(handCard)

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
      ]
    eat = [[],[],[],[]]
    pong = [[],[],[],[]]
    gang = [[],[],[],[]]
    for i in range(len(fulu)):
        for fulu_ in fulu[i]:
            if len(fulu_) == 3:
                if fulu_[0] == fulu_[1]:
                    pong[i].append(fulu_)
                else:
                    eat[i].append(fulu_)
            elif len(fulu_) == 4:
                gang[i].append(fulu_)

    print(eat)
    print(pong)
    print(gang)