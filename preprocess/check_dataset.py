import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np
with open('datasets/MQuAKE-1R-circuit.json', 'r') as f:
    mquake = json.load(f)

gptj_identical = []
identical_count = []

gptj_different = []
different_count = []
for item in mquake:
    if item['gptj_identical']:
        gptj_identical.append(item)
        identical_count.append(item['wiki_count'] + item['dolma_count'])
    else:
        gptj_different.append(item)
        different_count.append(item['wiki_count'] + item['dolma_count'])
print(np.sort(identical_count))
print(np.sort(different_count))
pass