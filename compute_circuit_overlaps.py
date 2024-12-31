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
from eap.graph import circuits_List_Union, circuits_List_Intersection
from tqdm import tqdm
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
with open('preprocess/prompts/rel-prompts.json','r') as f:
    rel_prompts = json.load(f)
with open('preprocess/prompts/multihop-prompts.txt','r') as f:
    multihop_prompt = f.read()

with open('preprocess/datasets/MQuAKE-CF-3k-gpt2-medium-corrupted.json', 'r') as f:
    mquake = json.load(f)


#load model
from utils import load_model, get_model
short_name ="gpt2-medium"
# MODEL_NAME = "EleutherAI/gpt-j-6B"
# MODEL_PATH = "/mnt/ssd2/models/gpt-j-6B"
hf_model, tokenizer = load_model(short_name)
model = get_model(short_name, hf_model=hf_model, tokenizer=tokenizer)




wrong_idx = [552, 843, 941, 1045, 1061, 1087, 1089, 1198, 1204, 1228, 1282, 1362, 1480, 1486, 1593, 1765, 1790, 1794, 1831, 1836,
             1837, 1486, 1593, 1765, 1790, 1794, 1831, 1836, 1837, 1897, 1952, 1977, 2022, 2035, 2059, 2116, 2154, 2177, 2212, 2250,
             2286, 2322, 2326, 2371, 2390, 2569, 2661, 2684, 2716, 2810, 2916, 2932, 2998, 11, 277, 308, 477, 572, 654, 655, 662, 709,
             716, 843, 941, 974, 1008, 1045, 1087, 1089, 1104, 1139, 1146, 1228, 1248, 1351, 1362, 1379, 1387, 1405, 1410, 1480, 1486, 1493,
             1526, 1593, 1783, 1790, 1794, 1836, 1837, 1956, 1960, 1977, 2002, 2035, 2059, 2071, 2073, 2078, 2093, 2097, 2118, 2154, 2177, 2212,
             2217, 2250, 2286, 2312, 2314, 2333, 2340, 2369, 2377, 2390,
             2447, 2488, 2515, 2582, 2619, 2661, 2676, 2683, 2704, 2716, 2759, 2760, 2794, 2810, 2890, 2916, 2967, 2999]

identical = []
for idx, item in enumerate(mquake):
    if not item['gpt2-medium_identical'] or idx in wrong_idx:
        continue

    # Compute multi-hop question circuit
    mh_clean_question = multihop_prompt + "\n\nQ: " + item['questions'][0] + ' A:'
    mh_corrupted_question = multihop_prompt + "\n\nQ: " + item['corrupted_question'] + ' A:'

    # mh_clean_question = item['questions'][0]
    # mh_corrupted_question = item['corrupted_question']

    sentences = [mh_clean_question, mh_corrupted_question]
    labels = [item[f'{short_name}_answer'], item[f'corrupted_{short_name}_answer']]

    data = (sentences, labels)
    mh_g = Graph.from_model(model)



    attribute(model, mh_g, data, partial(nll_loss_diff, mean=True, loss=True),
              method='EAP-IG-tokens', ig_steps=10)


    mh_g.apply_topn(5000, absolute=True)

    # mh_g.prune_dead_nodes()




    single_hop_circuits = []
    for single_hop in item['single_hops']:
        rel_prompt = rel_prompts[single_hop['relation_id']]
        clean_question = rel_prompt + "\n\nQ: " + single_hop['question'] + ' A:'
        corrupted_question = rel_prompt + "\n\nQ: " + single_hop['corrupted_question'] + ' A:'
        # clean_question = single_hop['question']
        # corrupted_question = single_hop['corrupted_question']

        sentences = [clean_question, corrupted_question]
        labels = [single_hop[f'{short_name}_answer'], single_hop[f'corrupted_{short_name}_answer']]


        data = (sentences, labels)

        g = Graph.from_model(model)

        attribute(model, g, data, partial(nll_loss_diff, mean=True, loss=True),
                  method='EAP-IG-tokens', ig_steps=10)

        g.apply_topn(5000, absolute=True)

        # g.prune_dead_nodes()

        single_hop_circuits.append(g)

        # intersect(one single-hop, multi-hop)
        clean_graph = Graph.from_model(model)
        intersect_graph = circuits_List_Intersection(clean_graph, [g, mh_g])

        edge_overlap_rate = intersect_graph.count_included_edges() / (g.count_included_edges() +
                                                     mh_g.count_included_edges() - intersect_graph.count_included_edges())
        node_overlap_rate = intersect_graph.count_included_nodes() / (g.count_included_nodes() +
                                                     mh_g.count_included_nodes() - intersect_graph.count_included_nodes())
        single_hop['edge_overlap_rate'] = edge_overlap_rate
        single_hop['node_overlap_rate'] = node_overlap_rate

    # intersect(all single-hops, multi-hop)
    clean_graph = Graph.from_model(model)
    union_graph = circuits_List_Union(clean_graph, single_hop_circuits)
    clean_graph = Graph.from_model(model)
    intersect_graph = circuits_List_Intersection(clean_graph, [union_graph, mh_g])
    edge_overlap_rate = intersect_graph.count_included_edges() / (union_graph.count_included_edges() +
                                                                  mh_g.count_included_edges() - intersect_graph.count_included_edges())
    node_overlap_rate = intersect_graph.count_included_nodes() / (union_graph.count_included_nodes() +
                                                                  mh_g.count_included_nodes() - intersect_graph.count_included_nodes())

    item['edge_overlap_rate'] = edge_overlap_rate
    item['node_overlap_rate'] = node_overlap_rate
    identical.append(item)
    # print(f"----[Item {idx}]---- edge_overlap_rate: {edge_overlap_rate} | "
    #       f"node_overlap_rate: {node_overlap_rate} | wiki & dolma count: {item['wiki_count'] + item['dolma_count']}")

    print(f"----[Item {idx}]---- edge_overlap_rate: {edge_overlap_rate} | wiki & dolma count: {item['wiki_count'] + item['dolma_count']}")



with open(f"preprocess/datasets/{short_name}-overlap-results", 'w') as f:
    json.dump(identical, f, ensure_ascii=False, indent=4)




