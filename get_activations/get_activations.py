# Pyvene method of getting activations
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json
# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_tqa_wice, tokenized_tqa_med
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llava_7B')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--dataset_length', type=int, default=300)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save', type=int, default=1)
    args = parser.parse_args()
    
    tokenizer = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model_llava
    device = "cuda"
    model = model.to(device)

    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    elif args.dataset_name == 'wice_eval_f':
        dataset = []
        with open('hall_eval_f.jsonl', 'r') as json_file:
            for line in json_file:
                data = json.loads(line) 
                dataset.append(data)
        formatter = tokenized_tqa_wice
    elif args.dataset_name == 'med':
        dataset = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")['train']
        formatter = tokenized_tqa_med


    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f"/net/scratch2/steeringwheel/dwlyu/features/{args.model_name}_{args.dataset_name}_categories.pkl", 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    collectors = []
    pv_config = []
    for layer in range(model.language_model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"language_model.model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations.copy())

    chunk_size = 100
    print("Saving labels")
    np.save(f"/net/scratch2/steeringwheel/dwlyu/features/{args.model_name}_{args.dataset_name}_labels.npy", labels)
    print("Saving layer wise activations in chunks")
    chunk_size = 100
    for i in range(0, len(all_layer_wise_activations), chunk_size):
        chunk = all_layer_wise_activations[i:i+chunk_size]
        np.save(f"/net/scratch2/steeringwheel/dwlyu/features/{args.model_name}_{args.dataset_name}_layer_wise_{i // chunk_size}.npy", chunk)
    print("Saving head wise activations in chunks")
    for i in range(0, len(all_head_wise_activations), chunk_size):
        chunk = all_head_wise_activations[i:i+chunk_size]
        np.save(f"/net/scratch2/steeringwheel/dwlyu/features/{args.model_name}_{args.dataset_name}_head_wise_{i // chunk_size}.npy", chunk)


if __name__ == '__main__':
        main()
