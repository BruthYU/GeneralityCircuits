import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm



with open('datasets/gptj-answers-CF-3k.json', 'r') as f:
    dataset = json.load(f)
with open('datasets/gptj-answers-CF-3k-multi.json', 'w') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
pass