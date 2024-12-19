
from utils import load_model, get_model


LLAMA_2_7B_CHAT_NAME = "meta-llama/Llama-2-7b-chat-hf"
LLAMA_2_7B_CHAT_PATH = "/mnt/ssd2/models/llama-2-7b-chat-hf"
hf_model, tokenizer = load_model(LLAMA_2_7B_CHAT_PATH, device="cuda")

clean_subject = 'Eiffel Tower' # France, Europe
corrupted_subject = 'the Great Walls' # China, Asian

single_prompt_template = 'The name of country where {} is located is'
hop_prompt_template = 'The country where {} is located belongs to the continent of'

prompt = [hop_prompt_template.format(clean_subject)]
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(hf_model.device)
generate_ids = hf_model.generate(**inputs, max_new_tokens=2)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])