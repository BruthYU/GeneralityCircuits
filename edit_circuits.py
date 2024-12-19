import hydra
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import FTHyperParams
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import json


def test_FT_Qwen():
    prompts = ['你是谁']
    ground_truth = ['我是通义千问,由阿里云开发的大预言模型']
    target_new = ['我是张三']
    hparams = FTHyperParams.from_hparams('./hparams/FT/qwen-7b')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new
    )

    # Save the edited model
    edited_model_path = "./edited_model"
    os.makedirs(edited_model_path, exist_ok=True)
    edited_model.save_pretrained(edited_model_path)

    # Save the metrics
    metrics_path = "./metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    return metrics_path, edited_model_path

    import torch

    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("edited_model", trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use auto mode, automatically select precision based on the device.
    model = AutoModelForCausalLM.from_pretrained("edited_model", device_map="auto", trust_remote_code=True).eval()

    # Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
    # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

    # 1st dialogue turn
    def generate_response(model, tokenizer, input_text, history=None, device='cuda'):
        if history is None:
            history = []

        # addinput text to history.
        history.append(input_text)

        #  move the model and the input tensor to the same device.
        model.to(device)

        # Encode history as input.
        inputs = tokenizer.encode(" ".join(history), return_tensors='pt').to(device)

        # Invoke model to generate response.
        outputs = model.generate(inputs, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated response.
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Add the response to the history.
        history.append(response)

        return response, history

        # Example Usage
        input_text = "你是谁"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        response, history = generate_response(model, tokenizer, input_text, device=device)
        print("Response:", response)
        print("History:", history)


def main():
    metrics_path, edited_model_path = test_FT_Qwen()
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()