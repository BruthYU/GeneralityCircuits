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
wrong_info = ['global_843', 'single_1045', 'single_1087', 'single_1089', 'single_1228', 'single_1362', 'single_1480', 'single_1486', 'single_1593', 'single_1790', 'single_1794', 'single_1836', 'single_1837', 'global_2035', 'global_2177', 'global_2250', 'global_2716', 'global_2810', 'global_2916']
for s in wrong_info:
    info = s.split('_')
    if info[0] == 'global':
        global_wrong.append(int(info[1]))
print(global_wrong)

# [843, 2035, 2177, 2250, 2716, 2810, 2916]
