# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 1:01
# @Author  : HaiqingSun
# @OriginalFileName: deal
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import json
import os

test_file = './test.txt'
target_file = './kiba_proteins_col.txt'
smi_file = './SMIbatch.txt'
protein_dict = {}

# 读取目标文件中的 JSON 字典
# with open(target_file, 'r') as json_file:
#     target_data = json.load(json_file)

# 处理 smi_file 中的每一行和 target_file 中的每个值
output_lines = []

with open(smi_file, 'r', encoding='utf-8') as smi_lines, open(target_file, 'r', encoding='utf-8') as tgt_lines:
    smis = smi_lines.readlines()
    tgts = tgt_lines.readlines()
    for smi_line in smis:
        smi_line = smi_line.strip()  # 去除行尾的换行符
        for tgt_line in tgts:
            tgt_line = tgt_line.strip()
            output_line = smi_line + ' ' + tgt_line
            output_lines.append(output_line)

# with open(smi_file, 'r') as smi_lines:
#     for smi_line in smi_lines:
#         smi_line = smi_line.strip()  # 去除行尾的换行符
#
#         for value in target_data.values():
#             output_line = smi_line + ' ' + str(value)
#             output_lines.append(output_line)

# 将结果写入输出文件
with open(test_file, 'w') as output:
    output.write('\n'.join(output_lines))