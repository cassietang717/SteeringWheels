import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
import math
from .pca import PCA

# import kornia
from transformers import set_seed

import random
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import List, Tuple
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import glob
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import json
import re
from datasets import load_dataset
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from evaluate import load

from sklearn.linear_model import LogisticRegression
from evaluate import load as load_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(image, processor):
    # if isinstance(image, Image.Image):
    #     image = image.convert("RGB")
    # elif isinstance(image, list) and isinstance(image[0], Image.Image):
    #     image = image[0].convert("RGB")

    answer = processor.image_processor(image, return_tensors="pt")

    # Check if the result is a dictionary and contains 'pixel_values' key
    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]

    # Convert numpy array to torch tensor if necessary
    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    
    # If it's already a tensor, return it directly
    elif isinstance(answer, torch.Tensor):
        return answer
    
    else:
        raise ValueError("Unexpected output format from image_processor.")
    
    return answer


def mask_patches(tensor, indices, patch_size=14):
    """
    Creates a new tensor where specified patches are set to the mean of the original tensor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    indices (list of int): Indices of the patches to modify
    patch_size (int): Size of one side of the square patch
    
    Returns:
    torch.Tensor: New tensor with modified patches
    """
    # Clone the original tensor to avoid modifying it
    new_tensor = tensor.clone()

    # Calculate the mean across the spatial dimensions
    mean_values = tensor.mean(dim=(1, 2), keepdim=True)
    
    # Number of patches along the width
    patches_per_row = tensor.shape[2] // patch_size
    total_patches = (tensor.shape[1] // patch_size) * (tensor.shape[2] // patch_size)

    for index in indices:
        # Calculate row and column position of the patch
        row = index // patches_per_row
        col = index % patches_per_row

        # Calculate the starting pixel positions
        start_x = col * patch_size
        start_y = row * patch_size

        # Replace the patch with the mean values
        new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = mean_values.expand(-1, patch_size, patch_size)#new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size].mean(dim=(1, 2), keepdim=True).expand(-1, patch_size, patch_size)# mean_values.expand(-1, patch_size, patch_size)

    return new_tensor

def format_prompt(image, question, answer, processor):
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"Q: {question} A:{answer}"},
        ],
    }]

    prompt = processor.apply_chat_template(conversation=conversation, add_generation_prompt=False)
    input = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    return input

def get_prompt_pairs_url(dataset, processor):
    all_prompt_pairs = [None] * len(dataset)

    for i, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Tokenizing prompts"):
        question = entry["question"]
        gt_answer = entry["gt_answer"]
        hallucinated_answer = entry["llava_model_answer"]
        image_url = entry["image_url"]

        try:
            response = requests.get(image_url)
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)
        except Exception as e:
            print(f"Error processing image URL {image_url}: {e}")
            continue

        gt_tokenized = format_prompt(image, question, gt_answer, processor)
        hallucinated_tokenized = format_prompt(image, question, hallucinated_answer, processor)

        all_prompt_pairs[i] = (hallucinated_tokenized, gt_tokenized) # positive and negative pairs

    return all_prompt_pairs

def get_prompt_pairs_nourl(args, dataset, processor):
    all_prompt_pairs = [None] * len(dataset)

    for i, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Tokenizing prompts"):
        question = entry["question"]
        gt_answer = entry["value"]
        hallucinated_answer = entry["h_value"]
        image_name = entry["image"]
        image_path = os.path.join(args.data_file, 'train2014', image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        gt_tokenized = format_prompt(image, question, gt_answer, processor)
        hallucinated_tokenized = format_prompt(image, question, hallucinated_answer, processor)

        all_prompt_pairs[i] = (hallucinated_tokenized, gt_tokenized) # positive and negative pairs

    return all_prompt_pairs


# def get_demos(args, processor, patch_size = 14, file_path = '/home/cassietang/steeringwheel/vision_tower/experiment/data/hallucination_vti_demos.jsonl'): 
#     # Initialize a list to store the JSON objects
#     data = []

#     # Open the file and read line by line
#     with open(file_path, 'r') as file:
#         for line in file:
#             # Each line is a complete JSON object
#             json_object = json.loads(line.strip())
#             data.append(json_object)
#     data_demos = random.sample(data, args.num_demos)

#     inputs_images = []
#     for i in range(len(data_demos)):
#         question = data_demos[i]['question']
#         image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
#         image_raw = Image.open(image_path).convert("RGB")
#         image_tensor = process_image(processor, image_raw)
#         image_tensor_cd_all_trials = []

#         for t in range(args.num_trials):
#             token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
#             mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
#             image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
#             image_tensor_cd_all_trials.append(image_tensor_cd)

#         inputs_images.append([image_tensor_cd_all_trials, image_tensor])

#     input_ids = get_prompt_pairs(data_demos, processor)
    
#     return inputs_images, input_ids

def get_demos_coco(args, processor, patch_size=14, file_path='/home/weiyitian/Winter 2025/steeringwheel/vision_tower/experiment/data/hallucination_vti_demos.jsonl'): 
    # Step 1: Load CSV
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)

    inputs_images = []
    for i in tqdm(range(len(data_demos)), desc="Getting data demos"):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")

        image_tensor = process_image(image_raw, processor)
        # Keep all 5 views (don't select only the first tile)
        image_tensor_cd_all_trials = []
        for t in range(args.num_trials):
            masked_views = []
            for view in image_tensor:
                token_numbers = view.shape[-1] * view.shape[-2] / patch_size**2
                mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
                masked_view = mask_patches(view, mask_index, patch_size=patch_size)
                masked_views.append(masked_view)
            image_tensor_cd_all_trials.append(torch.stack(masked_views))

        # Append both the masked trials and the reference image (all 5 views)
        inputs_images.append([image_tensor_cd_all_trials, image_tensor])
    # Step 3: Tokenize prompts
    input_ids = get_prompt_pairs_nourl(args, data_demos, processor)

    return inputs_images, input_ids

def get_demos_Halo(args, processor, patch_size=14, file_path='/home/weiyitian/Winter 2025/steeringwheel/HaloQuest/output/HaloQuest_llama.csv'): 
    # Step 1: Load CSV
    df = pd.read_csv(file_path)
    df = df[df["llama_hallucination_evaluation"] == "yes"]

    # Step 2: Randomly sample demos
    data_demos = df.sample(n=args.num_demos, random_state=42).to_dict(orient="records")

    inputs_images = []

    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_url = data_demos[i]['image_url']

        try:
            response = requests.get(image_url)
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)
            image.load()
            image_raw = image.convert("RGB")
        except Exception as e:
            print(f"Error loading image from {image_url}: {e}")
            continue

        image_tensor = process_image(image_raw, processor)
        # Keep all 5 views (don't select only the first tile)
        image_tensor_cd_all_trials = []
        for t in range(args.num_trials):
            masked_views = []
            for view in image_tensor:
                token_numbers = view.shape[-1] * view.shape[-2] / patch_size**2
                mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
                masked_view = mask_patches(view, mask_index, patch_size=patch_size)
                masked_views.append(masked_view)
            image_tensor_cd_all_trials.append(torch.stack(masked_views))

        # Append both the masked trials and the reference image (all 5 views)
        inputs_images.append([image_tensor_cd_all_trials, image_tensor])

    # Step 3: Tokenize prompts
    input_ids = get_prompt_pairs_url(data_demos, processor)
    
    return inputs_images, input_ids

# def get_hiddenstates(model, inputs, image_tensor):
#         h_all = []
#         with torch.no_grad():
#             for example_id in range(len(inputs)):
#                 embeddings_for_all_styles= []
#                 for style_id in range(len(inputs[example_id])):
#                     if image_tensor is None:
#                         h = model(
#                                 **inputs[example_id][style_id],
#                                 output_hidden_states=True,
#                                 return_dict=True).hidden_states
#                     else:
#                         h = model(
#                                 inputs[example_id][style_id],
#                                 images=image_tensor[example_id][-1].unsqueeze(0).half(),
#                                 use_cache=False,
#                                 output_hidden_states=True,
#                                 return_dict=True).hidden_states

#                     embedding_token = []
#                     for layer in range(len(h)):
#                         embedding_token.append(h[layer][:,-1].detach().cpu())
                    
#                     embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
#                     embeddings_for_all_styles.append(embedding_token)
#                 h_all.append(tuple(embeddings_for_all_styles))
#         return h_all

def get_hiddenstates(model, inputs, image_tensor):
    h_all = []
    with torch.no_grad():
        for example_id in range(len(inputs)):
            embeddings_for_all_styles = []
            for style_id in range(len(inputs[example_id])):
                # Fix: Convert BatchFeature to dict
                input_dict = inputs[example_id][style_id].to(device="cuda:0")  # move to GPU if needed
                input_dict = {k: v for k, v in input_dict.items()}

                if image_tensor is not None:
                    h = model(
                            **input_dict,
                            images=image_tensor[example_id][-1].unsqueeze(0).half(),
                            use_cache=False,
                            output_hidden_states=True,
                            return_dict=True).hidden_states
                else:
                    h = model(
                            **input_dict,
                            output_hidden_states=True,
                            return_dict=True).hidden_states

                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,-1].detach().cpu())
                
                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
    return h_all


def obtain_textual_vti(model, inputs, image_tensor, rank=1):
    # list of (neg_embedding: [num_layer, hidden_dim], pos_embedding: [num_layer, hidden_dim])
    hidden_states = get_hiddenstates(model, inputs, image_tensor)
    hidden_states_all = [] # [N, num_layer x hidden_dim]
    # N
    num_demonstration = len(hidden_states)
    neg_all = [] # [N, num_layer x hidden_dim]
    pos_all = [] # [N, num_layer x hidden_dim]

    for demonstration_id in range(num_demonstration):
        # [num_layer x hidden_dim]
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))

    # [N, num_layer x hidden_dim]
    fit_data = torch.stack(hidden_states_all)
    # [1, rank, num_layer x hidden_dim]
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    # [N, rank]
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data) 

    # pca.mean_ == X.mean(dim=0)
    # ([1, rank, num_layer x hidden_dim] + [num_layer x hidden_dim]) => num_layer x hidden_dim => [num_layer, hidden_dim]
    direction = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    # [N, num_layer x hidden_dim] => [num_layer, hidden_dim]
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction


def average_tuples(tuples: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # Check that the input list is not empty
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    # Check that all tuples have the same length
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the indices of the tuples
    for i in range(n):
        # Stack the tensors at the current index and compute the average
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)

    # Convert the list of averaged tensors to a tuple
    averaged_tuple = tuple(averaged_tensors)

    return averaged_tuple

def get_visual_hiddenstates(model, image_tensor):
    # num_tokens: (image_H * image_W) / (patch_H * patch_W)
    # num_trials: perturbed versions
    h_all = [] # N
    with torch.no_grad():
        try:
            vision_model = model.vision_tower.vision_model
        except:
            vision_model = model.vision_model

        # N  
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            
            # image_tensor[example_id] = [[img_trial_1, ..., img_trial_5], original image]
            for style_id in range(len(image_tensor[example_id])):
                # style_id = 0: perturbed image
                if isinstance(image_tensor[example_id][style_id], list):
                    h = []
                    # each image_tensor: [3, H, W]
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        # list of layer_num × [1, num_tokens, hidden_dim]
                        h_ = vision_model(
                            # [5, 3, H, W]
                            image_tensor_.half().cuda(),
                            output_hidden_states=True,
                            return_dict=True).hidden_states
                        
                        # convert each [5, 577, 1024] → [1, 577, 1024] # [1, num_tokens, hidden_dim]
                        h_ = [layer.mean(dim=0, keepdim=True) for layer in h_]
                        h.append(h_)
                    # averaged list of layer_num × [1, num_tokens, hidden_dim]
                    h = average_tuples(h)

                # style_id = 1: original image
                else:
                    # layer_num × [1, num_tokens, hidden_dim]
                    h = vision_model(
                        image_tensor[example_id][style_id].cuda(),
                        output_hidden_states=True,
                        return_dict=True).hidden_states
                    h = [layer.mean(dim=0, keepdim=True) for layer in h]
                
                embedding_token = []
                for layer in range(len(h)):
                    # [1, num_tokens, hidden_dim]
                    embedding_token.append(h[layer].detach().cpu())
                # [layer_num, num_tokens, hidden_dim]
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            # list of N (masked_avg_embedding, original_embedding), each of [layer_num, num_tokens, hidden_dim]
            h_all.append(tuple(embeddings_for_all_styles))

    del h, embedding_token

    return h_all

def obtain_visual_vti(model, image_tensor, rank=1):
    # list of N (masked_avg_embedding, original_embedding)
    hidden_states = get_visual_hiddenstates(model, image_tensor)
    # [layer_num, num_tokens, hidden_dim]
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    # N
    num_demonstration = len(hidden_states)

    hidden_states_all = [] # N
    for demonstration_id in range(num_demonstration):
        # average masked - original
        # [num_tokens, layer_num x hidden_dim]
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    # [num_tokens, N, layer_num x hidden_dim]
    fit_data = torch.stack(hidden_states_all,dim=1)[:] # n_token (no CLS token) x n_demos x D
    # PCA per token
    print("Fitting PCA")
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    # [num_tokens, rank, layer_num x hidden_dim] => [num_tokens, layer_num x hidden_dim] => [layer_num, num_tokens, hidden_dim]
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    # [num_tokens, layer_num x hidden_dim] => [layer_num, num_tokens, hidden_dim]
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction
