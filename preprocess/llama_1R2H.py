import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm



with open('datasets/MQuAKE-1R2H.json', 'r') as f:
    dataset = json.load(f)
pass

