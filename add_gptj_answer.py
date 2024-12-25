import torch
from typing import List
import json
from dataset.mquake import mqauke_Dataset
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
import pandas as pd
from tqdm import tqdm

#load dataset
with open('preprocess/datasets/MQuAKE-CF-3k.json', 'r') as f:
    mquake = json.load(f)
with open('preprocess/prompts/rel-prompts.json','r') as f:
    rel_prompts = json.load(f)
with open('preprocess/prompts/multihop-prompts.txt','r') as f:
    multihop_prompt = f.read()

wiki_df = pd.read_json('./preprocess/datasets/count_paras_wiki.json')
dolma_df = pd.read_json('./preprocess/datasets/count_paras_dolma.json')


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

for item in tqdm(mquake):
    zip_item = zip(item['single_hops'], item['orig']['triples'],item['orig']['triples_labeled'])
    for single_hop, triple, triple_labeled in zip_item:
        # set single_hop subject
        single_hop['subject'] = triple_labeled[0]
        # set relation_id & get single hop prompt
        relation_id = triple[1]
        single_hop['relation_id'] = relation_id
        rel_prompt = rel_prompts[relation_id]
        prompt = [rel_prompt + "\n\nQ: " + single_hop['question'] + ' A:']

        # get single hop anwsers
        answer = single_hop['answer']
        answer_alias = single_hop['answer_alias']
        all_answer = [answer] + answer_alias

        # get & set single hop gptj anwsers
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
        generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,
                                         stopping_criteria=stopping_criteria)
        generate_str = \
        tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]
        single_hop["gptj_answer"] = generate_ans
        single_hop["gptj_identical"] = generate_ans in all_answer



    # relate rewrite to single_hops
    for rewrite in item['requested_rewrite']:
        rewrite['single_hops_index'] = -1
        for idx, triple_labeled in enumerate(item['orig']['triples_labeled']):
            if rewrite['subject']==triple_labeled[0]:
                rewrite['single_hops_index'] = idx
                break

    # relate multi-hop question to single_hops (original or new)
    item['explicit_single_hops_index'] = -1
    for index, single_hop in enumerate(item['single_hops']):
        if single_hop['subject'] in item['questions'][0]:
            item['explicit_single_hops_index'] = index

    # get & set multi-hop answer
    prompt = [multihop_prompt + "\n\nQ: " + item['questions'][0] + ' A:']
    answer = item['answer']
    answer_alias = item['answer_alias']
    all_answer = [answer] + answer_alias

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
    generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,stopping_criteria=stopping_criteria)
    generate_str = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]
    item['gptj_answer'] = generate_ans
    item['gptj_identical'] = generate_ans in all_answer

    # get appearance count in corpus
    wiki_line = wiki_df[wiki_df['idx'] == item['case_id'] - 1]
    dolma_line = dolma_df[dolma_df['idx'] == item['case_id'] - 1]
    item['wiki_count'] = wiki_line.appearance_count.item()
    item['dolma_count'] = dolma_line.appearance_count.item()

with open("preprocess/datasets/MQuAKE-CF-3k-gptj.json", 'w') as f:
    json.dump(mquake, f, ensure_ascii=False, indent=4)

pass
