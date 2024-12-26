import pdb

import torch
from transformer_lens import HookedTransformer
from functools import partial
import torch.nn.functional as F
from eap.metrics import logit_diff, nll_loss_diff, logit_diff_metric
import transformer_lens.utils as utils
from eap.graph import Graph
from eap.dataset import EAPDataset
from eap.attribute import attribute
import time
from rich import print as rprint
import pandas as pd
from eap.evaluate import evaluate_graph, evaluate_baseline, get_circuit_logits
from utils import load_model, get_model
import os
from typing import List
import json
from eap.graph import circuits_Union, circuits_Intersection
from tqdm import tqdm

with open('preprocess/prompts/rel-prompts.json','r') as f:
    rel_prompts = json.load(f)
with open('preprocess/prompts/multihop-prompts.txt','r') as f:
    multihop_prompt = f.read()

with open('.preprocess/datasets/MQuAKE-1R-circuit.json', 'r') as f:
    mquake = json.load(f)


#load model
from utils import load_model, get_model
MODEL_PATH = "/mnt/ssd2/models/gpt-j-6B"
hf_model, tokenizer = get_model(MODEL_PATH, device="cuda")



for item in tqdm(mquake):
    for single_hop in item['single_hops']:
        pass




