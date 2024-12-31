import pdb

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
with open('./preprocess/datasets/MQuAKE-CF-3k-gpt2-medium.json', 'r') as f:
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


P407_template = 'Which language was {} written in?'
P407_subjects = ['Journey to the West', 'Dream of the Red Chamber', 'Romance of the Three Kingdoms']
P407_answer = ['Chinese', 'zh', 'zho', 'Chinese language']
for p407_subject in P407_subjects:
    p407_info = {
        'subject': p407_subject,
        'question': P407_template.format(p407_subject),
        'original_answers': ['Chinese', 'zh', 'zho', 'Chinese language']
    }
    relation_id_dict['P407'].append(p407_info)

for idx, x in enumerate(relation_id_dict['P27']):
    if x['subject'] == 'England':
        print(idx)

#load model
from utils import load_model, get_model
short_name = "gpt2-medium"
model_name ="gpt2-medium"
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# MODEL_PATH = "/mnt/ssd2/models/gpt-j-6B"
hf_model, tokenizer = load_model(model_name, device="cuda")
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

def sample_corrupted_triple(clean_subject, relation_id, clean_object):
    corrupted_subject = clean_subject
    corrupted_case = None
    sample_times = 0
    while corrupted_subject == clean_subject or clean_object in corrupted_case['original_answers']:
        index = random.randint(0, len(relation_id_dict[relation_id]) - 1)
        corrupted_case = relation_id_dict[relation_id][index]
        corrupted_subject = corrupted_case['subject']
        sample_times += 1
        if sample_times > 1000:
            print(relation_id)
            break
    return corrupted_case

wrong_info = ['single_552', 'global_843', 'global_941', 'single_1045', 'single_1061',
              'single_1087', 'single_1089', 'single_1198', 'single_1204', 'single_1228', 'global_1282', 'single_1362', 'single_1480',
              'single_1486', 'single_1593', 'single_1765', 'single_1790', 'single_1794', 'single_1831', 'single_1836', 'single_1837',
              'single_1486', 'single_1593', 'single_1765', 'single_1790', 'single_1794', 'single_1831', 'single_1836', 'single_1837',
              'single_1897', 'global_1952', 'global_1977', 'global_2022', 'global_2035', 'global_2059', 'global_2116', 'global_2154',
              'global_2177', 'global_2212', 'global_2250', 'global_2286', 'single_2322', 'single_2326', 'global_2371', 'global_2390',
              'single_2569', 'global_2661', 'global_2684', 'global_2716', 'global_2810', 'global_2916', 'global_2932', 'global_2998']
for idx, item in enumerate(tqdm(mquake)):

    for single_hop in item['single_hops']:
       relation_id = single_hop['relation_id']
       clean_subject, clean_question = single_hop['subject'], single_hop['question']

       generate_ans = single_hop[f'{short_name}_answer']
       corrupted_question = ""


       generate_count = 0
       while generate_ans == single_hop[f'{short_name}_answer']:
           corrupted_case = sample_corrupted_triple(clean_subject, relation_id, single_hop[f'{short_name}_answer'])

           corrupted_question = clean_question.replace(clean_subject, corrupted_case['subject'])

           rel_prompt = rel_prompts[relation_id]
           prompt = [rel_prompt + "\n\nQ: " + corrupted_question + ' A:']

           inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
           generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,
                                            stopping_criteria=stopping_criteria)
           generate_str = \
           tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
           generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]
           generate_count +=1
           if generate_count >100:
               wrong_info.append(f"single_{idx}")
               break



       single_hop['corrupted_question'],single_hop[f'corrupted_{short_name}_answer'] \
           = corrupted_question, generate_ans

    # get & set multi_hop questions
    explict_index = item['explicit_single_hops_index']
    single_gt = item['single_hops'][explict_index]
    relation_id = single_gt['relation_id']
    clean_subject, clean_question = single_gt['subject'], item['questions'][0]
    generate_ans = item[f'{short_name}_answer']


    corrupted_question = ""
    generate_count = 0
    while generate_ans == item[f'{short_name}_answer']:
        corrupted_case = sample_corrupted_triple(clean_subject, relation_id, item[f'{short_name}_answer'])
        corrupted_question = clean_question.replace(clean_subject, corrupted_case['subject'])

        prompt = [multihop_prompt + "\n\nQ: " + corrupted_question + ' A:']
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
        generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,stopping_criteria=stopping_criteria)
        generate_str = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]
        generate_count += 1
        if generate_count > 100:
            wrong_info.append(f"global_{idx}")
            break


    item['corrupted_question'], item[f'corrupted_{short_name}_answer'] \
        = corrupted_question, generate_ans

print(wrong_info)
with open(f"preprocess/datasets/MQuAKE-CF-3k-{short_name}-corrupted.json", 'w') as f:
    json.dump(mquake, f, ensure_ascii=False, indent=4)