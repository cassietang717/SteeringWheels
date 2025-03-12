from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import argparse
from einops import rearrange

import pyvene as pv
import torch
from datasets import load_dataset, concatenate_datasets
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# from vision_activation.interveners import wrapper, ITI_Intervener
from vision_activation.utils import ignore_warnings, load_chunks, get_com_directions, get_top_heads, layer_head_to_flattened_idx
from vision_activation.utils import apply_interventions, llama_evaluate, eval_ce_kl_owt, plot_layer_head_PCA


gt_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_HaloQuest/HaloQuest_gt_layer_wise_*.npy"
gt_layer_wise_activations = load_chunks(gt_layer_wise_pattern) #(P, 33, 4096)
hallucinated_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_HaloQuest/HaloQuest_hallucinated_layer_wise_*.npy"
hallucinated_layer_wise_activations = load_chunks(hallucinated_layer_wise_pattern) #(P, 33, 4096)

gt_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_HaloQuest/HaloQuest_gt_head_wise_*.npy"
gt_head_wise_activations = load_chunks(gt_head_wise_pattern) #(P, 32, 4096)
hallucinated_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_HaloQuest/HaloQuest_hallucinated_head_wise_*.npy"
hallucinated_head_wise_activations = load_chunks(hallucinated_head_wise_pattern) #(P, 32, 4096)

gt_head_wise_activations = rearrange(gt_head_wise_activations, 'b l (h d) -> b l h d', h = 32) #(P, 33, 32, 128)
hallucinated_head_wise_activations = rearrange(hallucinated_head_wise_activations, 'b l (h d) -> b l h d', h = 32) #(P, 32, 32, 128)
all_head_wise_activations = np.concatenate([gt_head_wise_activations, hallucinated_head_wise_activations], axis=0)  #(2P, 33, 32, 128)

print("Successfully loaded all activation chunks")

with open("output/stats_seed3407_alpha12_top48", "r") as file:
    data = json.load(file)

top_head_idxs = data["top_head_idxs"][:4]

plot_layer_head_PCA(gt_head_wise_activations, hallucinated_head_wise_activations, top_head_idxs, 4, "figures/" + "top4", font_size=20)