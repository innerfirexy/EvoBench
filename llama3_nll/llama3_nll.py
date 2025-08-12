# %%
import os
import platform
import socket
import pickle
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn
print('torch version:', torch.__version__)

# %%
# Load model and tokenizer
# Note: You need to adjust the model path according to your environment
model_name = 'llama3-8b-instruct'
if platform.system() == 'Darwin':
    model_path = f'/Users/xy/models/{model_name}'  # Please modify according to actual situation
elif socket.gethostname() == 'l40-server':
    model_path = f'/data1/model/{model_name}'  # Please modify according to actual situation
elif socket.gethostname() == 'a5880-server':
    model_path = f'/data2/model/{model_name}'  # Please modify according to actual situation
assert os.path.exists(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Use CPU for computation (if no GPU available)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             device_map='auto', 
                                             trust_remote_code=True).eval()


# %%
# Independent function for computing NLLs
def text_to_nlls(text, tokenizer, model):
    device = model.device
    ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=True).to(device)

    # Forward pass
    try:
        output = model(ids)
    except Exception:
        raise
    logits = output.logits.to(device)
    logits = logits.permute(0, 2, 1) # reshape logits from (B, L, V) to (B, V, L)
    shift_logits = logits[:, :, :-1]
    shift_targets = ids[:, 1:]

    # NLL
    loss_fn = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)
    try:
        nlls = loss_fn(log_softmax(shift_logits), shift_targets)
        nlls = nlls.squeeze(0)
    except Exception:
        raise

    return nlls.detach().cpu().numpy()


# %%
def compute_json_nll(json_file: str, tokenizer, model):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # key: original => human
    human_texts = data['original']
    human_nlls = []
    for text in tqdm(human_texts, leave=False):
        nlls = text_to_nlls(text, tokenizer, model)
        human_nlls.append(nlls)
    # key: sampled => model
    model_texts = data['sampled']
    model_nlls = []
    for text in tqdm(model_texts, leave=False):
        nlls = text_to_nlls(text, tokenizer, model)
        model_nlls.append(nlls)

    return human_nlls, model_nlls


# %%
def exp_llama3():
    input_dir = '../LlaMA3/'
    
    # 列出所有 JSON 文件（排除 args 子目录）
    json_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            json_files.append(os.path.join(input_dir, file))
    # print(f'Found {len(json_files)} JSON files')
    
    # Compute NLLs
    output_dir = './LlaMA3/'
    os.makedirs(output_dir, exist_ok=True)
    for json_file in tqdm(json_files):
        human_nlls, model_nlls = compute_json_nll(json_file, tokenizer, model)
        assert len(human_nlls) == len(model_nlls)
        # save
        basename = os.path.basename(json_file)
        basename_prefix = basename.split('.')[0]
        human_nlls_path = os.path.join(output_dir, f'{basename_prefix}_human.txt')
        with open(human_nlls_path, 'w') as f:
            for nlls in human_nlls:
                f.write(' '.join(f'{nll:.5f}' for nll in nlls) + '\n')
        model_nlls_path = os.path.join(output_dir, f'{basename_prefix}_model.txt')
        with open(model_nlls_path, 'w') as f:
            for nlls in model_nlls:
                f.write(' '.join(f'{nll:.5f}' for nll in nlls) + '\n')

# %%
if __name__ == '__main__':
    exp_llama3()
