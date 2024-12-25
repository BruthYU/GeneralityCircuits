import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
    mquake = json.load(f)

with open('datasets/gptj-answers-CF-1R-single.json', 'r') as f:
    single = json.load(f)
with open('datasets/gptj-answers-CF-1R-multi.json', 'r') as f:
    multi = json.load(f)

subject_match = 0
for x,y in zip(single,multi):
    if x['subject'] in x['question']:
        subject_match+=1

print(subject_match)
pass