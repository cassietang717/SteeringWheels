from tqdm import tqdm
import numpy as np
from collections import defaultdict

import argparse
from einops import rearrange

import pyvene as pv
import torch
from datasets import load_dataset, concatenate_datasets
#from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from vision_activation.interveners import wrapper, ITI_Intervener
from vision_activation.utils import ignore_warnings, load_chunks, get_com_directions, get_top_heads, layer_head_to_flattened_idx
from vision_activation.utils import apply_interventions, llama_evaluate, eval_ce_kl_owt, plot_layer_head_PCA

print("what")