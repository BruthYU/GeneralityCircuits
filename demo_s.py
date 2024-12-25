import pdb

import torch
from typing import List
import json
from dataset.mquake import mqauke_Dataset
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings



with open('preprocess/prompts/rel-prompts.json','r') as f:
    rel_prompts = json.load(f)
#load dataset
mquake_dataset = mqauke_Dataset('./preprocess/datasets/MQuAKE-1R.json')
pass

# load model
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




# stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[11]))
gptj_answers = []
successful_single = 0
for item in mquake_dataset:
    requested_rewrite = item['requested_rewrite'][0]
    rel_prompt = rel_prompts[requested_rewrite['relation_id']]

    single_gt = item['single_hops'][0]
    for single_hop in item['single_hops']:
        if requested_rewrite['target_true']['str'] == single_hop['answer']:
            single_gt = single_hop
    prompt = [rel_prompt + "\n\nQ: " + single_gt['question'] + ' A:']

    answer = single_gt['answer']
    answer_alias = single_gt['answer_alias']
    all_answer = [answer] + answer_alias


    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
    generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,stopping_criteria=stopping_criteria)
    generate_str = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]

    gptj_answers.append({
        'case_id': item['case_id'],
        'subject': item['requested_rewrite'][0]['subject'],
        'relation_id': item['requested_rewrite'][0]['relation_id'],
        'question': single_gt['question'],
        'gptj_answer': generate_ans,
        'original_answers': all_answer,
        'identical': generate_ans in all_answer
    })





    if generate_ans in all_answer:
        successful_single += 1
        print(f"Success Match {successful_single}: {generate_ans}")

with open("preprocess/datasets/gptj-answers-CF-1R-single.json", 'w') as f:
    json.dump(gptj_answers, f, ensure_ascii=False, indent=4)

    # for ans in all_answer:
    #     if ans in generate_ans:
    #         successful_single += 1
    #         print(f"Success Match {successful_single}:")


print(f'-------Single Hop Acc: {successful_single / len(mquake_dataset)}')