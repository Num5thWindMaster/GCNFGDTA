# -*- coding: utf-8 -*-
# @Time    : 2024/7/4 19:24
# @Author  : HaiqingSun
# @OriginalFileName: data_preprocessing
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import json


def count_pn_kiba_num():
    with open("./davis-kiba/kiba.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(len(lines))
        count = 0
        pp_count = 0
        np_count = 0
        for line in lines:
            count += 1
            if float(line.split()[-1]) >= 12.1:
                pp_count += 1
            else:
                np_count += 1
        print('total, pp, np:', (count, pp_count, np_count))


if __name__ == "__main__":
    count_pn_kiba_num()
