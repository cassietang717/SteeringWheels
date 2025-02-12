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
from utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q, tokenized_hallubench, tokenized_hallubench_qa
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
    
    tokenizer = AutoTokenizer.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf",)
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model = model_llava.language_model
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
    elif args.dataset_name == 'halu_qa':
        dataset = []
        k = 0
        with open('qa_data.json', 'r') as json_file:
            for line in json_file:
                data = json.loads(line) 
                dataset.append(data)
                k += 1
                if k >= args.dataset_length:
                    break
        formatter = tokenized_hallubench_qa
    elif args.dataset_name == 'summary':
        dataset = []
        k = 0
        with open('summarization_data.json', 'r') as json_file:
            for line in json_file:
                data = json.loads(line) 
                dataset.append(data)
                k += 1
                if k >= args.dataset_length:
                    break
        formatter = tokenized_hallubench


    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'../features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
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
    if args.save == 1:
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

    layer_num = 32
    fig_head, axes_head = plt.subplots(8, 4, figsize=(20, 40))
    axes_head = axes_head.flatten()
    all_head_wise_activations = np.asarray(all_head_wise_activations)
    all_layer_wise_activations = np.asarray(all_layer_wise_activations)

    for layer_id in range(layer_num):
        yes_features_head = []
        no_features_head = []

        for idx in range(len(labels)):
            if labels[idx] == 1:
                yes_features_head.append(all_head_wise_activations[idx, layer_id])
            else:
                no_features_head.append(all_head_wise_activations[idx, layer_id])

        # Convert lists to NumPy arrays
        yes_features_head = np.array(yes_features_head)
        no_features_head = np.array(no_features_head)
        
        # Concatenate and apply PCA
        combined_features_head = np.concatenate((yes_features_head, no_features_head), axis=0)
        pca_head = PCA(n_components=2)
        pca_features_head = pca_head.fit_transform(combined_features_head.reshape(combined_features_head.shape[0], -1))

        # Plot in the corresponding subplot
        ax = axes_head[layer_id]
        ax.scatter(pca_features_head[:yes_features_head.shape[0], 0], pca_features_head[:yes_features_head.shape[0], 1], 
                label='Yes', alpha=0.7, color='blue', s=10)
        ax.scatter(pca_features_head[yes_features_head.shape[0]:, 0], pca_features_head[yes_features_head.shape[0]:, 1], 
                label='No', alpha=0.7, color='orange', s=10)
        ax.set_title(f'Head-wise Layer {layer_id}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{args.dataset_name}_attention_output.png')

    fig_layer, axes_layer = plt.subplots(8, 4, figsize=(20, 40))
    axes_layer = axes_layer.flatten()

    for layer_id in range(layer_num):
        yes_features_layer = []
        no_features_layer = []

        # Separate layer-wise activations based on labels
        for idx in range(len(labels)):
            if labels[idx] == 1:
                yes_features_layer.append(all_layer_wise_activations[idx, layer_id])
            else:
                no_features_layer.append(all_layer_wise_activations[idx, layer_id])

        # Convert lists to NumPy arrays
        yes_features_layer = np.array(yes_features_layer)
        no_features_layer = np.array(no_features_layer)
        
        # Concatenate and apply PCA
        combined_features_layer = np.concatenate((yes_features_layer, no_features_layer), axis=0)
        pca_layer = PCA(n_components=2)
        pca_features_layer = pca_layer.fit_transform(combined_features_layer.reshape(combined_features_layer.shape[0], -1))

        # Plot in the corresponding subplot
        ax = axes_layer[layer_id]
        ax.scatter(pca_features_layer[:yes_features_layer.shape[0], 0], pca_features_layer[:yes_features_layer.shape[0], 1], 
                label='Yes', alpha=0.7, color='blue')
        ax.scatter(pca_features_layer[yes_features_layer.shape[0]:, 0], pca_features_layer[yes_features_layer.shape[0]:, 1], 
                label='No', alpha=0.7, color='orange')
        ax.set_title(f'Layer-wise Layer {layer_id}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{args.dataset_name}_hidden_state_output.png')




if __name__ == '__main__':
        main()
