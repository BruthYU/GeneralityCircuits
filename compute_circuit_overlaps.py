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

with open('preprocess/prompts/rel-prompts-short.json','r') as f:
    rel_prompts = json.load(f)
with open('preprocess/prompts/multihop-prompts-short.txt','r') as f:
    multihop_prompt = f.read()

with open('preprocess/datasets/MQuAKE-1R-circuit.json', 'r') as f:
    mquake = json.load(f)


#load model
from utils import load_model, get_model
MODEL_NAME = "EleutherAI/gpt-j-6B"
MODEL_PATH = "/mnt/ssd2/models/gpt-j-6B"
hf_model, tokenizer = load_model(MODEL_PATH)
model = get_model(MODEL_NAME, hf_model=hf_model, tokenizer=tokenizer)



for idx, item in enumerate(mquake):

    # Compute multi-hop question circuit
    # mh_clean_question = multihop_prompt + "\n\nQ: " + item['questions'][0] + ' A:'
    # mh_corrupted_question = multihop_prompt + "\n\nQ: " + item['corrupted_question'] + ' A:'

    mh_clean_question = item['questions'][0]
    mh_corrupted_question = item['corrupted_question']

    sentences = [mh_clean_question, mh_corrupted_question]
    labels = [item['gptj_answer'], item['corrupted_gptj_answer']]
    data = (sentences, labels)
    mh_g = Graph.from_model(model)
    attribute(model, mh_g, data, partial(nll_loss_diff, mean=True, loss=True),
              method='EAP-IG-tokens', ig_steps=100)

    mh_g.apply_topn(6500, absolute=True)
    mh_g.prune_dead_nodes()



    single_hop_circuits = []
    for single_hop in item['single_hops']:
        rel_prompt = rel_prompts[single_hop['relation_id']]
        # clean_question = rel_prompt + "\n\nQ: " + single_hop['question'] + ' A:'
        # corrupted_question = rel_prompt + "\n\nQ: " + single_hop['corrupted_question'] + ' A:'
        clean_question = single_hop['question']
        corrupted_question = single_hop['corrupted_question']

        sentences = [clean_question, corrupted_question]
        labels = [single_hop['gptj_answer'], single_hop['corrupted_gptj_answer']]
        data = (sentences, labels)

        g = Graph.from_model(model)
        attribute(model, g, data, partial(nll_loss_diff, mean=True, loss=True),
                  method='EAP-IG-tokens', ig_steps=1)

        g.apply_topn(6500, absolute=True)
        g.prune_dead_nodes()
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
    print(f"----[Item {idx}]---- edge_overlap_rate: {edge_overlap_rate} | node_overlap_rate: {node_overlap_rate}")



with open("preprocess/datasets/overlap-results", 'w') as f:
    json.dump(mquake, f, ensure_ascii=False, indent=4)




