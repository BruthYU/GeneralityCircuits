import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm



with open('datasets/MQuAKE-CF-circuits.json', 'r') as f:
    dataset = json.load(f)
pass