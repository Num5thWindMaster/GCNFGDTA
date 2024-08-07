# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 19:17
# @Author  : HaiqingSun
# @OriginalFileName: mol_match_info
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import pandas as pd

# 读取Excel文件
df = pd.read_csv('top3.csv', encoding='utf-8')
df2 = pd.read_csv('mol_info.csv', encoding='utf-8')

v = df2['SMILES'].values
# 遍历C列
for i, value in df['smiles'].items():
    # 检查C列的值是否存在于F列

    if value in v:
        # 获取F列中匹配值的索引
        match_index = df2[df2['SMILES'] == value].index[0]
        # 将E列对应行中的内容粘贴到B列
        df.at[i, 'Name(CN)'] = df2.at[match_index, '化合物名']
        df.at[i, 'Name'] = df2.at[match_index, 'Name']

# 保存到新的Excel文件
df.to_csv('top3_with_mol_info.csv',index=False, encoding='utf_8_sig')
