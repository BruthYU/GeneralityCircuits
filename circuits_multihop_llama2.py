import torch
from transformer_lens import HookedTransformer
from functools import partial
import torch.nn.functional as F
from eap.metrics import logit_diff, direct_logit
import transformer_lens.utils as utils
from eap.graph import Graph
from eap.dataset import EAPDataset
from eap.attribute import attribute
import time
import torch
from transformer_lens import HookedTransformer
from functools import partial
import torch.nn.functional as F
from eap.metrics import logit_diff, direct_logit
import transformer_lens.utils as utils
from eap.graph import Graph
from eap.graph import circuits_Union, circuits_Intersection, circuits_Intersection_Relax
from eap.dataset import EAPDataset
from eap.attribute import attribute
import time
from rich import print as rprint
import pandas as pd
from eap.evaluate import evaluate_graph, evaluate_baseline,get_circuit_logits
from utils import load_model, get_model
import os
import pdb
# from huggingface_hub import login
# login(token = "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")
# os.environ['http_proxy'] = '127.0.0.1:7890'
# os.environ['https_proxy'] = '127.0.0.1:7890'
import pdb
'''
Load Model
'''
LLAMA_2_7B_CHAT_NAME = "meta-llama/Llama-2-7b-chat-hf"
LLAMA_2_7B_CHAT_PATH = "/mnt/ssd2/models/llama-2-7b-chat-hf"
hf_model, tokenizer = load_model(LLAMA_2_7B_CHAT_PATH)
model = get_model(LLAMA_2_7B_CHAT_NAME, hf_model=hf_model, tokenizer=tokenizer)


model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
# 改了这个地方后面绘图应该会报错
if "use_hook_mlp_in" in model.cfg.to_dict():
    model.set_use_hook_mlp_in(True)




clean_subject = 'Eiffel Tower' # France, Europe
corrupted_subject = 'the Great Walls' # China, Asian

single_prompt_template = 'The name of country where {} is located is'
single_labels = ['France', 'China']

hop_prompt_template = 'The country where {} is located belongs to the continent of'
hop_labels = ['Europe', 'Asian']

rephrase_prompt_template = "The {} is situated in the country of"



def get_component_logits(logits, model, answer_token, top_k=10):
    logits = utils.remove_batch_dim(logits)
    # print(heads_out[head_name].shape)
    probs = logits.softmax(dim=-1)
    token_probs = probs[-1]
    answer_str_token = model.to_string(answer_token)
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
    # Janky way to get the index of the token in the sorted list - I couldn't find a better way?
    correct_rank = torch.arange(len(sorted_token_values))[
        (sorted_token_values == answer_token).cpu()
    ].item()
    # answer_ranks = []
    # answer_ranks.append((answer_str_token, correct_rank))
    # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
    # rprint gives rich text printing
    rprint(
        f"Performance on answer token:\n[b]Rank: {correct_rank: <8} Logit: {logits[-1, answer_token].item():5.2f} Prob: {token_probs[answer_token].item():6.2%} Token: |{answer_str_token}|[/b]"
    )
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{model.to_string(sorted_token_values[i])}|"
        )
    # rprint(f"[b]Ranks of the answer tokens:[/b] {answer_ranks}")

def get_pad_data(model, clean_subject, corrupted_subject, prompt_template, labels):
    clean = prompt_template.format(clean_subject)
    corrupted = prompt_template.format(corrupted_subject)

    if not model.to_tokens(clean).shape[-1] == model.to_tokens(corrupted).shape[-1]:
        toks = model.to_tokens([clean, corrupted])
        clean, corrupted = tuple(model.to_string(toks[:, 1:]))




    country_idx = model.tokenizer(labels[0], add_special_tokens=False).input_ids[0]
    corrupted_country_idx = model.tokenizer(labels[1], add_special_tokens=False).input_ids[0]
    label = [[country_idx, corrupted_country_idx]]
    label = torch.tensor(label)
    data = ([clean], [corrupted], label)
    return data


def get_knowledge_circuits(model, data, topn=5000, is_multihop=False):
    print("--------------------------Get Knowledge Circuits--------------------------------")
    '''
    Running EAP-IG
    '''
    g = Graph.from_model(model)
    start_time = time.time()
    # Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
    attribute(model, g, data, partial(logit_diff, loss=True, mean=True), method='EAP-IG-case', ig_steps=100)
    # attribute(model, g, data, partial(direct_logit, loss=True, mean=True), method='EAP-IG-case', ig_steps=30)
    # attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP-IG', ig_steps=30)
    g.apply_topn(topn, absolute=True)
    g.prune_dead_nodes()
    print(f'Nodes in graph: {g.count_included_nodes()} | Edges in graph: {g.count_included_edges()}')
    # g.to_json('graph.json')
    #
    # gz = g.to_graphviz()
    # gz.draw(f'graph.png', prog='dot')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"函数执行时间：{execution_time}秒")


    logits = get_circuit_logits(model, g, data)
    label = hop_labels[0] if is_multihop else single_labels[0]
    get_component_logits(logits, model, answer_token=model.to_tokens(label, prepend_bos=False)[0], top_k=5)

    baseline = evaluate_baseline(model, [data], partial(logit_diff, loss=False, mean=False)).mean().item()
    results = evaluate_graph(model, g, [data], partial(logit_diff, loss=False, mean=False)).mean().item()
    print(f"Original performance was {baseline}; the circuit's performance is {results}")

    return g

single_data = get_pad_data(model, clean_subject, corrupted_subject, single_prompt_template, single_labels)
hop_data = get_pad_data(model, clean_subject, corrupted_subject, hop_prompt_template, hop_labels)
rephrase_data = get_pad_data(model, clean_subject, corrupted_subject, rephrase_prompt_template, single_labels)


g1 = get_knowledge_circuits(model, single_data, topn=5000)
g2 = get_knowledge_circuits(model, hop_data, topn=5000, is_multihop=True)





print("[-------------------------Union & Intersection-------------------------------]")

print(f'g1 Edges Num: {g1.count_included_edges()} | g1 Nodes Num: {g1.count_included_nodes()}')
print(f'g2 Edges Num: {g2.count_included_edges()} | g2 Nodes Num: {g2.count_included_nodes()}')

clean_graph = Graph.from_model(model)
intersection_graph = circuits_Intersection_Relax(clean_graph, g1, g2, ratio=0.2)
print(f'--- Relax Edges Num: {intersection_graph.count_included_edges()} | Relax Nodes Num: {intersection_graph.count_included_nodes()}---')

print("---Single Question Evaluation---")
print(single_prompt_template.format(clean_subject))
logits = get_circuit_logits(model, intersection_graph, single_data)
get_component_logits(logits, model, answer_token=model.to_tokens(single_labels[0], prepend_bos=False)[0], top_k=5)

print("---Hop Question Evaluation---")
print(hop_prompt_template.format(clean_subject))
logits = get_circuit_logits(model, intersection_graph, hop_data)
get_component_logits(logits, model, answer_token=model.to_tokens(hop_labels[0], prepend_bos=False)[0], top_k=5)

print("---Rephrase Question Evaluation---")
print(rephrase_prompt_template.format(clean_subject))
logits = get_circuit_logits(model, intersection_graph, rephrase_data)
get_component_logits(logits, model, answer_token=model.to_tokens(single_labels[0], prepend_bos=False)[0], top_k=5)


# clean_graph = Graph.from_model(model)
# union_graph = circuits_Union(clean_graph, g1, g2)
# print(f'---Union Edges Num: {union_graph.count_included_edges()} | Union Nodes Num: {union_graph.count_included_nodes()}---')
# logits = get_circuit_logits(model, union_graph, single_data)
# get_component_logits(logits, model, answer_token=model.to_tokens(single_labels[0], prepend_bos=False)[0], top_k=5)
#
#
#
#
# clean_graph = Graph.from_model(model)
# intersection_graph = circuits_Intersection(clean_graph, g1, g2)
# print(f'---Intersection Edges Num: {intersection_graph.count_included_edges()} | Intersection Nodes Num: {intersection_graph.count_included_nodes()}---')
# logits = get_circuit_logits(model, intersection_graph, single_data)
# get_component_logits(logits, model, answer_token=model.to_tokens(single_labels[0], prepend_bos=False)[0], top_k=5)







