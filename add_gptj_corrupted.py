import torch
from typing import List
import json
from dataset.mquake import mqauke_Dataset
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
import pandas as pd
from tqdm import tqdm
import random
random.seed(10)
#load dataset
with open('./preprocess/datasets/MQuAKE-CF-3k-gptj.json', 'r') as f:
    mquake = json.load(f)
with open('preprocess/prompts/rel-prompts.json','r') as f:
    rel_prompts = json.load(f)
with open('preprocess/prompts/multihop-prompts.txt', 'r') as f:
    multihop_prompt = f.read()
# get relation id dict
relation_id_dict = {}
for item in mquake:
    for single_hop in item['single_hops']:
        relation_id = single_hop['relation_id']
        if relation_id not in relation_id_dict.keys():
            relation_id_dict[relation_id] = []

        info = {
            'subject': single_hop['subject'],
            'question': single_hop['question'],
            'original_answers': [single_hop['answer']] + single_hop['answer_alias']
        }
        relation_id_dict[relation_id].append(info)

#load model
from utils import load_model, get_model
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_PATH = "/mnt/ssd2/models/gpt-j-6B"
hf_model, tokenizer = load_model(MODEL_PATH, device="cuda")
stop_words_ids = [
    tokenizer.encode(stop_word)[0] for stop_word in ["\n"]]

class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, token_id_list: List[int] = None):
        self.token_id_list = token_id_list

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

stopping_criteria = StoppingCriteriaList()
stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=stop_words_ids))

# get items with one rewrite
one_rewrite = []
for item in mquake:
    if len(item['requested_rewrite'])==1:
        one_rewrite.append(item)


for item in tqdm(one_rewrite):
    # get & set single_hop questions
    for single_hop in item['single_hops']:
       relation_id = single_hop['relation_id']
       clean_subject, clean_question = single_hop['subject'], single_hop['question']
       corrupted_subject = clean_subject
       corrupted_case = None
       while corrupted_subject == clean_subject:
           index = random.randint(0, len(relation_id_dict[relation_id]) - 1)
           corrupted_case = relation_id_dict[relation_id][index]
           corrupted_subject = corrupted_case['subject']

       corrupted_question = clean_question.replace(clean_subject, corrupted_subject)

       rel_prompt = rel_prompts[relation_id]
       prompt = [rel_prompt + "\n\nQ: " + corrupted_question + ' A:']

       inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
       generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,
                                        stopping_criteria=stopping_criteria)
       generate_str = \
       tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
       generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]

       single_hop['corrupted_question'],single_hop['corrupted_gptj_answer'] \
           = corrupted_question, generate_ans

    # get & set multi_hop questions
    explict_index = item['explicit_single_hops_index']
    single_gt = item['single_hops'][explict_index]
    relation_id = single_gt['relation_id']
    clean_subject, clean_question = single_gt['subject'], item['questions'][0]
    corrupted_subject = clean_subject
    corrupted_case = None
    while corrupted_subject == clean_subject:
        index = random.randint(0, len(relation_id_dict[relation_id]) - 1)
        corrupted_case = relation_id_dict[relation_id][index]
        corrupted_subject = corrupted_case['subject']
    corrupted_question = clean_question.replace(clean_subject, corrupted_subject)

    prompt = [multihop_prompt + "\n\nQ: " + corrupted_question + ' A:']
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
    generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,stopping_criteria=stopping_criteria)
    generate_str = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]
    item['corrupted_question'], item['corrupted_gptj_answer'] \
        = corrupted_question, generate_ans

with open("preprocess/datasets/MQuAKE-1R-corrupted.json", 'w') as f:
    json.dump(one_rewrite, f, ensure_ascii=False, indent=4)