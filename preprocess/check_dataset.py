import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np

with open('datasets/MQuAKE-CF-3k-gpt2-medium-corrupted.json', 'r') as f:
    mquake = json.load(f)
#
# identical = []
count = 0
for item in mquake:
    flag = True
    if not item['gpt2-medium_identical']:
        flag=False
    for single_hop in item['single_hops']:
        if not single_hop['gpt2-medium_identical']:
            flag=False
    if flag:
        count+=1
print(count)

#
#
# count = 0
# for item in identical:
#     if item['wiki_count'] + item['dolma_count'] < 10:
#         count += 1
#         print(item['wiki_count'] + item['dolma_count'])
# print(count)



global_wrong = []
wrong_info = ['single_552', 'global_843', 'global_941', 'single_1045', 'single_1061', 'single_1087', 'single_1089', 'single_1198', 'single_1204', 'single_1228',
              'global_1282', 'single_1362', 'single_1480', 'single_1486', 'single_1593', 'single_1765', 'single_1790', 'single_1794', 'single_1831', 'single_1836',
              'single_1837', 'single_1486', 'single_1593', 'single_1765', 'single_1790', 'single_1794', 'single_1831', 'single_1836', 'single_1837', 'single_1897',
              'global_1952', 'global_1977', 'global_2022', 'global_2035', 'global_2059', 'global_2116', 'global_2154', 'global_2177', 'global_2212', 'global_2250',
              'global_2286', 'single_2322', 'single_2326', 'global_2371', 'global_2390', 'single_2569', 'global_2661', 'global_2684', 'global_2716', 'global_2810',
              'global_2916', 'global_2932', 'global_2998', 'global_11', 'global_277', 'global_308', 'global_477', 'global_572', 'global_654', 'single_655', 'single_662',
              'global_709', 'global_716', 'global_843', 'global_941', 'global_974', 'global_1008', 'single_1045', 'single_1087', 'single_1089', 'single_1104', 'global_1139',
              'single_1146', 'single_1228', 'global_1248', 'global_1351', 'single_1362', 'global_1379', 'global_1387', 'single_1405', 'single_1410', 'single_1480',
              'single_1486', 'global_1493', 'single_1526', 'single_1593', 'single_1783', 'single_1790', 'single_1794', 'single_1836', 'single_1837', 'global_1956',
              'global_1960', 'global_1977', 'global_2002', 'global_2035', 'global_2059', 'global_2071', 'global_2073', 'single_2078', 'global_2093', 'global_2097',
              'global_2118', 'global_2154', 'global_2177', 'global_2212', 'global_2217', 'global_2250', 'global_2286', 'global_2312', 'global_2314', 'global_2333',
              'global_2340', 'global_2369', 'global_2377', 'global_2390', 'global_2447', 'global_2488', 'global_2515', 'global_2582', 'global_2619', 'global_2661',
              'global_2676', 'global_2683', 'single_2704',
              'global_2716', 'global_2759', 'global_2760', 'global_2794', 'global_2810', 'single_2890', 'global_2916', 'global_2967', 'global_2999']
for s in wrong_info:
    info = s.split('_')
    global_wrong.append(int(info[1]))
print(len(global_wrong))

# [843, 2035, 2177, 2250, 2716, 2810, 2916]
