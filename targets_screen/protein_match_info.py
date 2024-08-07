# -*- coding: utf-8 -*-
# @Time    : 2024/7/3 16:08
# @Author  : HaiqingSun
# @OriginalFileName: protein_match_info
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import pandas as pd
import json

df = pd.read_csv('top3_with_mol_info.csv', encoding='utf-8')

with open('../predict_data/kibaproteins.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    protein_dict = dict(json.loads(content))

reversed_dict = {value: key for key, value in protein_dict.items()}

for i, value in df['am_seq'].items():
    # 获取F列中匹配值的索引
    match_index = reversed_dict[value]
    # 将E列对应行中的内容粘贴到B列
    df.at[i, 'UniprotID'] = match_index

# 保存到新的Excel文件
df.to_csv('top3_with_protein_info.csv',index=False, encoding='utf_8_sig')
