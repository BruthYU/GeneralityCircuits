from functools import partial
from typing import List, Optional, Union
import sys
sys.path.append('..')
import torch
import transformers
from transformer_lens import ActivationCache, HookedTransformer, SVDInterpreter
torch.set_grad_enabled(False)
from utils import *
from plot_utils import *
print("Disabled automatic differentiation")

# load model
device = "cuda:4"
# NBVAL_IGNORE_OUTPUT
gpt2_medium = HookedTransformer.from_pretrained(
    model_name= "gpt2-medium",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
    refactor_factored_attn_matrices=True,
)

gpt2_medium = HookedTransformer.from_pretrained(
    model_name= "tinyllama",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device=device,
    refactor_factored_attn_matrices=True,
)
# Get the default device used

tinyllama_path = 'path_to_checkpoint_of_tinyllama'
hf_model = transformers.AutoModelForCausalLM.from_pretrained(tinyllama_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(tinyllama_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = HookedTransformer.from_pretrained(
    model_name= "tinyllama",
    hf_model=hf_model,
    tokenizer=tokenizer,
    local_path=tinyllama_path,
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=True,
    device=device,
)


China = ComponentAnalyzer(gpt2_medium,'The official language of France is',' French','France')
# China.get_min_rank_at_subject(gpt2_medium.W_U, China.answer_token)
draw_rank_logits(gpt2_medium,China)
draw_attention_pattern(China,gpt2_medium,20,6)