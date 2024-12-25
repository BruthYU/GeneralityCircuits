import pdb

import torch
from typing import List
import json
from dataset.mquake import mqauke_Dataset
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
import random
import pandas as pd
random.seed(10)




#load data
with open('preprocess/datasets/MQuAKE-CF-3k.json', 'r') as f:
    mquake = json.load(f)
with open('preprocess/datasets/gptj-answers-CF-1R-single.json', 'r') as f:
    single = json.load(f)
with open('preprocess/datasets/gptj-answers-CF-1R-multi.json', 'r') as f:
    multi = json.load(f)
with open('preprocess/prompts/rel-prompts.json','r') as f:
    rel_prompts = json.load(f)
with open('preprocess/prompts/multihop-prompts.txt', 'r') as f:
    multihop_prompt = f.read()

wiki_df = pd.read_json('./preprocess/datasets/count_paras_wiki.json')
dolma_df = pd.read_json('./preprocess/datasets/count_paras_wiki.json')


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

#build new data
relation_id_dict = {}
for item in mquake:
    for rewrite in item['requested_rewrite']:
        relation_id = rewrite['relation_id']
        if relation_id not in relation_id_dict:
            relation_id_dict[relation_id] = []
        for single_hop in item['single_hops']:
            if rewrite['target_true']['str'] == single_hop['answer']:
                single_gt = single_hop
        info = {
            'subject': rewrite['subject'],
            'question': single_gt['question'],
            'original_answers': [single_gt['answer']] + single_gt['answer_alias']
        }
        relation_id_dict[relation_id].append(info)



gptj_circuit = []
for s, m in zip(single, multi):

    # single_hop clean and corrupted
    relation_id = s['relation_id']
    clean_subject, clean_question, clean_gptj_answer, clean_original_answers, clean_identical = \
        s['subject'], s['question'], s['gptj_answer'], s['original_answers'], s['identical']
    replace_subject = clean_subject
    corrupted_case = None
    while replace_subject == clean_subject:
        index = random.randint(0, len(relation_id_dict[relation_id])-1)
        corrupted_case = relation_id_dict[relation_id][index]
        replace_subject = corrupted_case['subject']

    corrupted_subject = replace_subject
    corrupted_question = clean_question.replace(clean_subject, corrupted_subject)

    rel_prompt = rel_prompts[relation_id]
    prompt = [rel_prompt + "\n\nQ: " + corrupted_question + ' A:']

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
    generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,stopping_criteria=stopping_criteria)
    generate_str = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]

    single_info = {
        'clean': s,
        'corrupted': {
            "subject": corrupted_subject,
            "question": corrupted_question,
            "gptj_answer": generate_ans,
        }
    }

    # multi_hop clean and corrupted
    relation_id = m['relation_id']
    clean_subject, clean_question, clean_gptj_answer, clean_original_answers, clean_identical = \
        s['subject'], s['question'], s['gptj_answer'], s['original_answers'], s['identical']
    replace_subject = clean_subject
    corrupted_case = None
    while replace_subject == clean_subject:
        index = random.randint(0, len(relation_id_dict[relation_id])-1)
        corrupted_case = relation_id_dict[relation_id][index]
        replace_subject = corrupted_case['subject']
    corrupted_subject = replace_subject

    clean_subject, clean_question, clean_gptj_answer, clean_original_answers, clean_identical = \
        m['subject'], m['question'], m['gptj_answer'], m['original_answers'], m['identical']
    corrupted_question = clean_question.replace(clean_subject, corrupted_subject)

    pdb.set_trace()

    prompt = [multihop_prompt + "\n\nQ: " + corrupted_question + ' A:']
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
    generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,stopping_criteria=stopping_criteria)
    generate_str = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]

    multi_info = {
        'clean': m,
        'corrupted': {
            "subject": corrupted_subject,
            "question": corrupted_question,
            "gptj_answer": generate_ans,
        }
    }

    case_id = s['case_id']
    wiki_line = wiki_df[wiki_df['idx']==case_id-1]
    dolma_line = dolma_df[dolma_df['idx']==case_id-1]
    pdb.set_trace()
    assert s['case_id'] == m['case_id'] and wiki_line.subject.item().strip() == s['subject'].lower(), "not equal"
    info = {
        'case_id': s['case_id'],
        'single_info': single_info,
        'multi_info': multi_info,
        'wiki_count': wiki_line.appearance_count.item(),
        'dolma_count': dolma_line.appearance_count.item()

    }
    gptj_circuit.append(info)


with open("preprocess/datasets/1R_clean_corrupted.json", 'w') as f:
    json.dump(gptj_circuit, f, ensure_ascii=False, indent=4)





    pass



