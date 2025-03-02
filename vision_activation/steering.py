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
from utils import get_prompt_pairs, get_activations_pyvene, save_activations, plot_pca_comparison, load_chunks

def main():

    gt_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest_gt_layer_wise_*.npy"
    gt_layer_wise_activations = load_chunks(gt_layer_wise_pattern)
    hallucinated_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest_hallucinated_layer_wise_*.npy"
    hallucinated_layer_wise_activations = load_chunks(hallucinated_layer_wise_pattern)

    gt_head_wise_pattern = f"/net/scratch2/steeringwheel/dwlyu/activations/HaloQuest_gt_head_wise_*.npy"
    gt_head_wise_activations = load_chunks(gt_head_wise_pattern)
    hallucinated_head_wise_pattern = f"/net/scratch2/steeringwheel/dwlyu/activations/HaloQuest_hallucinated_head_wise_*.npy"
    hallucinated_head_wise_activations = load_chunks(hallucinated_head_wise_pattern)
    
    print("Successfully loaded all chunks!")