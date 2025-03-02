import os
from tqdm import tqdm
import numpy as np
import pickle
import sys

import numpy as np
import pickle
import argparse
import json

import pyvene as pv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from interveners import wrapper, Collector, ITI_Intervener
from utils import get_prompt_pairs, get_activations_pyvene, save_activations, plot_pca_comparison


def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "HaloQuest". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llava_7B')
    parser.add_argument('--dataset_name', type=str, default='HaloQuest')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save', type=int, default=1)
    args = parser.parse_args()

    # Vision Encoder → Projection Layer → Transformer Decoder
    # vision_tower → multi_modal_projector → language_model

    print("Loading models")
    processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model = model_llava.language_model
    device = "cuda"
    model = model.to(device)

    if args.dataset_name == "HaloQuest": 
        dataset = load_dataset("csv", data_files="../HaloQuest/output/HaloQuest_llama.csv")
        dataset = dataset.filter(lambda entry: entry["llama_hallucination_evaluation"] == "yes")
        formatter = get_prompt_pairs
    else:
        raise ValueError(f"Wrong Dataset Choice: {args.dataset_name}")
    
    
    prompt_pairs = formatter(dataset, processor)

    num_hidden_layers = model.config.num_hidden_layers

    collectors, pv_configs = [None] * num_hidden_layers, [None] * num_hidden_layers
    for i in range(num_hidden_layers):
        collector = Collector(head=-1)
        pv_config = {
            # (batch_size, seq_len, hidden_dim=num_heads * D_head)
            "component": f"model.layers[{i}].self_attn.o_proj.input",
            "intervention": wrapper(collector)
        }
        collectors[i] = collector
        pv_configs[i] = pv_config
    pv_model = pv.IntervenableModel(pv_configs, model)


    print("Getting activations")
    gt_layer_wise_activations, hallucinated_layer_wise_activations = [None] * len(prompt_pairs), [None] * len(prompt_pairs)
    gt_head_wise_activations, hallucinated_head_wise_activations = [None] * len(prompt_pairs), [None] * len(prompt_pairs)

    for i, prompt_pair in tqdm(enumerate(prompt_pairs), total=len(dataset["train"]), desc="Processing prompts"):
        gt_prompt, hallcuinated_prompt = prompt_pair

        gt_layer_wise_activation, gt_head_wise_activation = get_activations_pyvene(pv_model, collectors, gt_prompt, device)
        gt_layer_wise_activations[i] = gt_layer_wise_activation[:, -1, :].copy()
        gt_head_wise_activations[i] = gt_head_wise_activation.copy()

        hallcuinated_layer_wise_activation, hallcuinated_head_wise_activation = get_activations_pyvene(pv_model, collectors, hallcuinated_prompt, device)
        hallucinated_layer_wise_activations[i] = hallcuinated_layer_wise_activation[:, -1, :].copy()
        hallucinated_head_wise_activations[i] = hallcuinated_head_wise_activation.copy()

        # print(gt_head_wise_activation.shape) # (32, 4096)
        # print(gt_layer_wise_activation[:, -1, :].shape) # (33, 4096)
        # print(hallcuinated_head_wise_activation.shape) # (32, 4096)
        # print(hallcuinated_layer_wise_activation[:, -1, :].shape) # (33, 4096)
        # break

    gt_layer_wise_activations, gt_head_wise_activations = np.stack(gt_layer_wise_activations), np.stack(gt_head_wise_activations)
    hallucinated_layer_wise_activations, hallucinated_head_wise_activations = np.stack(hallucinated_layer_wise_activations), np.stack(hallucinated_head_wise_activations)

    if args.save == 1:
        save_activations(gt_layer_wise_activations, gt_head_wise_activations, "HaloQuest_gt")
        save_activations(hallucinated_layer_wise_activations, hallucinated_head_wise_activations, "HaloQuest_hallucinated")
    
    plot_pca_comparison(gt_layer_wise_activations, hallucinated_layer_wise_activations, "HaloQuest_layer_wise", layer_num=33)
    plot_pca_comparison(gt_head_wise_activations, hallucinated_head_wise_activations, "HaloQuest_head_wise", layer_num=32)


if __name__ == '__main__':
        main()