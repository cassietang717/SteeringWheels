
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

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(image, processor):
    # if isinstance(image, Image.Image):
    #     image = image.convert("RGB")
    # elif isinstance(image, list) and isinstance(image[0], Image.Image):
    #     image = image[0].convert("RGB")
    # processor.image_processor.crop_size={"height": 336, "width": 336}
    # processor.image_processor.size= {"shortest_edge": 336}
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

    prompt = processor.apply_chat_template(conversation=conversation, add_generation_prompt=True)
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

def get_demos_coco(args, image_processor, processor, patch_size=14, file_path='/home/cassietang/steeringwheel/vision_tower/experiment/data/hallucination_vti_demos.jsonl'): 
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
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_raw,image_processor)
        # image_tensor = image_tensor[0] # （5， 3, 336, 336）-> (3, 336, 336)

        image_tensor_cd_all_trials = []
        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1] * image_tensor.shape[-2] / patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
            image_tensor_cd_all_trials.append(image_tensor_cd)

        inputs_images.append([image_tensor_cd_all_trials, image_tensor])

    # Step 3: Tokenize prompts
    input_ids = get_prompt_pairs_nourl(args, data_demos, processor)
    
    return inputs_images, input_ids

def get_demos_Halo(args, image_processor, processor, patch_size=14, file_path='/home/cassietang/steeringwheel/HaloQuest/output/HaloQuest_llama.csv'): 
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

        image_tensor = process_image(image_raw,image_processor)
        # image_tensor = image_tensor[0] # （5， 3, 224, 224）-> (3, 224, 224)
        image_tensor_cd_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1] * image_tensor.shape[-2] / patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
            image_tensor_cd_all_trials.append(image_tensor_cd)

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
    hidden_states = get_hiddenstates(model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data) 

    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
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
    h_all = []
    with torch.no_grad():
        try:
            vision_model = model.vision_tower.vision_model
        except:
            vision_model = model.vision_model
            
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                if isinstance(image_tensor[example_id][style_id], list):
                    h = []
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        h_ = vision_model(
                            image_tensor_.unsqueeze(0).half().cuda(),
                            output_hidden_states=True,
                            return_dict=True).hidden_states
                        h.append(h_)
                    h = average_tuples(h)
                else:
                    h = vision_model(
                        image_tensor[example_id][style_id].unsqueeze(0).cuda(),
                        output_hidden_states=True,
                        return_dict=True).hidden_states
                
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))

    del h, embedding_token

    return h_all

def obtain_visual_vti(model, image_tensor, rank=1):

    hidden_states = get_visual_hiddenstates(model, image_tensor)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)

    
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:] # n_token (no CLS token) x n_demos x D
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    # direction = pca.components_.mean(dim=0).view(n_layers, n_tokens, -1)
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction
