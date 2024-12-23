import pdb

import torch
from typing import List
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings


class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时，立即停止生成
    ---------------
    ver: 2023-08-02
    by: changhongyu
    """

    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        # 储存scores会额外占用资源，所以直接用input_ids进行判断
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list




from utils import load_model, get_model
subjects = ['Tesla', 'NVIDIA', 'Alibaba', 'Tencent', 'Baidu', 'Daji']
labels = ['Elon Musk', 'Jensen Huang']

prompt_template = 'The name of the CEO of {} is'

LLAMA_2_7B_CHAT_NAME = "meta-llama/Llama-2-7b-chat-hf"
LLAMA_2_7B_CHAT_PATH = "/mnt/ssd2/models/llama-2-7b-hf"
hf_model, tokenizer = load_model(LLAMA_2_7B_CHAT_PATH, device="cuda")



# Tokenize without special tokens
# sample_sentence = "hello-hello."
# tokenized_output_no_special = tokenizer(sample_sentence, add_special_tokens=False)
# print(tokenized_output_no_special["input_ids"])
# print("Tokenized Text:", [tokenizer.decode([x]) for x in tokenized_output_no_special["input_ids"]])

stopping_criteria = StoppingCriteriaList()
stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[29892, 29991, 29889]))
# stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[11]))

hop_prompt_template = 'The name of the CEO of {} is'

prompt = [hop_prompt_template.format(subjects[-1])]

inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
generate_ids = hf_model.generate(**inputs, stopping_criteria=stopping_criteria)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])