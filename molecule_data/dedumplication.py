# -*- coding: utf-8 -*-
# @Time    : 2024/7/1 1:31
# @Author  : HaiqingSun
# @OriginalFileName: dedumplication
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import csv

def process_csv(input_file, output_file):
    unique_records = {}  # 用于存储唯一记录的字典

    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取标题行
        key_index = 0  # 第一列的索引
        min_value_index = 5  # 第六列的索引

        for row in reader:
            key = row[key_index]
            value = float(row[min_value_index])

            if key in unique_records:
                if value < int(unique_records[key][min_value_index]):
                    unique_records[key] = row
            else:
                unique_records[key] = row

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入标题行
        writer.writerows(unique_records.values())  # 写入唯一记录的值

    print("output:", output_file)

# 示例用法
input_file = 'input.csv'
output_file = 'output.csv'
process_csv(input_file, output_file)