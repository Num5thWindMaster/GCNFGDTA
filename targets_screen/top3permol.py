# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 18:33
# @Author  : HaiqingSun
# @OriginalFileName: top3permol
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import pandas as pd

# 读取CSV文件
df = pd.read_csv('pretop3.csv')

# 获取第一列作为序号列
# sequence_col = df.iloc[:, 0]
sequence_col = df['index']
smi_col = df['smiles']

# 创建一个列表来存储结果
filtered_rows = []

# 遍历序号列，保留最多连续三个递增的序号所在的行
count = 1
jump_flag = False
filtered_rows.append(df.iloc[0].tolist())
for i in range(1, len(sequence_col)):
    # if sequence_col[i - 1] + 1 != sequence_col[i] or smi_col[i] != smi_col[i-1]:
    if smi_col[i] != smi_col[i - 1]:
        count = 1
        filtered_rows.append(df.iloc[i].tolist())
    # elif sequence_col[i-1] + 1 == sequence_col[i] and smi_col[i] == smi_col[i-1]:
    elif smi_col[i] == smi_col[i - 1]:
        count += 1
        if count <= 3:
            filtered_rows.append(df.iloc[i].tolist())

# 将结果转换为DataFrame
filtered_df = pd.DataFrame(filtered_rows, columns=df.columns)

# 保存到新的CSV文件
filtered_df.to_csv('top3.csv', index=False)
