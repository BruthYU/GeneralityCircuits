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

with open('prompts/rel-prompts.json', 'r') as f:
    rel_prompts = json.load(f)

for x in rel_prompts.keys():
    QAs = rel_prompts[x].split('\n')
    rel_prompts[x] = QAs[0] + '\n' +QAs[1]
with open('prompts/rel-prompts-short.json', 'w') as f:
    json.dump(rel_prompts, f, indent=4)
pass