import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import re
with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)




def replace_word(text, old_word, new_word):
    # 创建正则表达式模式，以匹配单词边界
    pattern = r'\b{}\b'.format(re.escape(old_word))
    # 使用re.sub进行替换
    return re.sub(pattern, new_word, text)

single_rewrite_dataset = []
relation_id_dict = {}
for d in dataset:
    for r in d["requested_rewrite"]:
        relation_id = r["relation_id"]
        if relation_id not in relation_id_dict.keys():
            relation_id_dict[relation_id] = []
        relation_id_dict[relation_id].append(r)
    if len(d["requested_rewrite"])==1 and len(d['single_hops'])==2:
        single_rewrite_dataset.append(d)


MQuAKE_circuits = []


for s in single_rewrite_dataset:
    SR = s["requested_rewrite"][0] # single_hop_question
    relation_id = SR["relation_id"]

    clean_subject = SR["subject"]
    singlehop_prompt_relation = SR['prompt']
    singlehop_prompt_question = replace_word(SR['question'], clean_subject, "{}")
    clean_edited_label = SR["target_new"]
    clean_label = SR["target_true"]

    same_relations = relation_id_dict[relation_id]

    sample_times = 0
    CR = None
    for idx in range(len(same_relations)):
        CR = same_relations[idx]
        if CR["subject"] != clean_subject and abs(len(clean_subject) - len(CR["subject"])) < 4:
            break

    if CR is None:
        raise ValueError("Length requirement not satisfied")

    corrupted_subject = CR["subject"]
    corrupted_label = CR["target_true"]

    multihop_questions = s["questions"]
    if clean_subject not in multihop_questions[0]:
        continue
    multihop_prompts = [replace_word(q, clean_subject, '{}') for q in multihop_questions]
    multihop_clean_labels = [s['answer']] + s['answer_alias']
    multihop_clean_edited_labels = [s['new_answer']] + s['new_answer_alias']

    info = {
        "singlehop_prompt_sro": singlehop_prompt_relation,
        "singlehop_prompt_question": singlehop_prompt_question,
        "singlehop_clean_subject": clean_subject,
        "singlehop_clean_label": clean_label,
        "singlehop_clean_edited_label": clean_edited_label,
        "singlehop_corrupted_subject": corrupted_subject,
        "singlehop_corrupted_label": corrupted_label,
        "multihop_prompts": multihop_prompts,
        "multihop_clean_labels": multihop_clean_labels,
        "multihop_clean_edited_labels": multihop_clean_edited_labels
    }

    MQuAKE_circuits.append(info)











with open('datasets/MQuAKE-CF-3k-circuits.json', 'w') as f:
    json.dump(MQuAKE_circuits, f)

