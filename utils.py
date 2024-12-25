import dataclasses
from functools import partial
import wandb
import os
import logging
import torch
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)

import transformers

from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


GPT_J_NAME_SHORT = "gpt-j-6B"  # A useful alias for the CLI.
GPT_J_NAME = "EleutherAI/gpt-j-6B"

GPT_NEO_X_NAME_SHORT = "neox"
GPT_NEO_X_NAME = "EleutherAI/gpt-neox-20b"

LLAMA_13B_NAME = "llama-13b"
LLAMA_30B_NAME = "llama-30b"
LLAMA_NAME_SHORT = "llama"

def load_model(
    name: str, device = torch.device('cpu'), fp16: Optional[bool] = None
):
    """Load the model given its string name.

    Args:
        name: Name of the model or path to it.
        device: If set, send model to this device. Defaults to CPU.
        fp16: Whether to use half precision. If not set, depends on model.
    Returns:
        ModelAndTokenizer: Loaded model and its tokenizer.
    """
    if name == GPT_J_NAME_SHORT:
        name = GPT_J_NAME
    elif name == GPT_NEO_X_NAME_SHORT:
        name = GPT_NEO_X_NAME
    elif name == LLAMA_NAME_SHORT:
        name = LLAMA_13B_NAME

    # I usually save randomly initialized variants under the short name of the
    # corresponding real model (e.g. gptj_random, neox_random), so check here
    # if we are dealing with *any* variant of the big model.
    is_gpt_j_variant = name == GPT_J_NAME or GPT_J_NAME_SHORT in name
    is_neo_x_variant = name == GPT_NEO_X_NAME or GPT_NEO_X_NAME_SHORT in name
    is_llama_variant = (
        name in {LLAMA_13B_NAME, LLAMA_30B_NAME} or LLAMA_NAME_SHORT in name
    )

    if fp16 is None:
        fp16 = is_gpt_j_variant or is_neo_x_variant or is_llama_variant

    torch_dtype = torch.float16 if fp16 else None

    kwargs: dict = dict(torch_dtype=torch_dtype)
    if is_gpt_j_variant:
        kwargs["low_cpu_mem_usage"] = True
        if fp16:
            kwargs["revision"] = "float16"

    logger.info(f"loading {name} (fp16={fp16})")

    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(torch_dtype)
    model.to(device)
    model.eval()
    print(is_llama_variant)
    if is_llama_variant:
        tokenizer = transformers.LlamaTokenizerFast.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token = "</s>"
        tokenizer.pad_token_id = tokenizer.eos_token_id = 2
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_model(name, hf_model, tokenizer, device="cuda",local_path=None) -> HookedTransformer:
    if 'Llama' in name or 'llama' in name:
        tl_model =HookedTransformer.from_pretrained(name, hf_model=hf_model, device="cpu", fold_ln=False, dtype=torch.float16,
                                                    local_path=local_path, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)
    else:
        tl_model = HookedTransformer.from_pretrained(name, hf_model=hf_model, tokenizer=tokenizer,local_path=local_path)
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True) 
    #改了这个地方后面绘图应该会报错
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    # logger.info(
    #     f"dtype: {tl_model.type}, device: {tl_model.device}, memory: {tl_model.get_memory_footprint()}"
    # )
    return tl_model