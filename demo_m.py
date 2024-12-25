import pdb

import torch
from typing import List
import json
from dataset.mquake import mqauke_Dataset
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings



with open('preprocess/prompts/multihop-prompts.txt','r') as f:
    multihop_prompt = f.read()
#load dataset
mquake_dataset = mqauke_Dataset('./preprocess/datasets/MQuAKE-CF-3k.json')
pass

# load model
from utils import load_model, get_model

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
successful_multi = 0
for item in mquake_dataset:



    prompt = [multihop_prompt + "\n\nQ: " + item['questions'][0] + ' A:']

    answer = item['answer']
    answer_alias = item['answer_alias']
    all_answer = [answer] + answer_alias


    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
    generate_ids = hf_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id,stopping_criteria=stopping_criteria)
    generate_str = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    generate_ans = generate_str.strip().split('\n')[-1].split("A: ")[-1]
    gptj_answers.append({
        'case_id': item['case_id'],
        'subject': item['requested_rewrite'][0]['subject'],
        'question': item['questions'][0],
        'gptj_answer': generate_ans,
        'original_answers': all_answer,
        'identical': generate_ans in all_answer
    })




    if generate_ans in all_answer:
        successful_multi += 1
        print(f"Success Match {successful_multi}: {generate_ans}")

    with open("preprocess/datasets/gptj-answers-CF-3k.json", 'w') as f:
        json.dump(gptj_answers, f, ensure_ascii=False)


print(f'-------Multi Hop Acc: {successful_multi / len(mquake_dataset)}')