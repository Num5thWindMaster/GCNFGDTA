# -*- coding: utf-8 -*-
# @Time    : 2024/7/3 16:57
# @Author  : HaiqingSun
# @OriginalFileName: protein_match_details
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import pandas as pd

# 读取Excel文件
df = pd.read_csv('top3_with_protein_info.csv', encoding='utf-8')
df2 = pd.read_csv('idmapping_2024_07_03.csv', encoding='utf-8')

v = df2['From'].values
# 遍历C列
for i, value in df['UniprotID'].items():
    # 检查C列的值是否存在于F列

    if value in v:
        # 获取F列中匹配值的索引
        match_index = df2[df2['From'] == value].index[0]
        # 将E列对应行中的内容粘贴到B列
        df.at[i, 'Entry'] = df2.at[match_index, 'Entry Name']
        df.at[i, 'Protein names'] = df2.at[match_index, 'Protein names']
        df.at[i, 'Gene Names'] = df2.at[match_index, 'Gene Names']

# 保存到新的Excel文件
df.to_csv('top3_with_all_info_details.csv',index=False, encoding='utf_8_sig')