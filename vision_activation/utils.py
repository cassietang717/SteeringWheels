import os
from tqdm import tqdm
import numpy as np
import math
import glob
from functools import partial
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from baukit import Trace, TraceDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


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


def get_prompt_pairs(dataset, processor):
    all_prompt_pairs = [None] * len(dataset["train"])

    for i, entry in tqdm(enumerate(dataset["train"]), total=len(dataset["train"]), desc="Tokenizing prompts"):
        question = entry["question"]
        gt_answer = entry["gt_answer"]
        hallucinated_answer = entry["llava_model_answer"]
        image_url = entry["image_url"]

        response = requests.get(image_url)
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)

        gt_tokenized = format_prompt(image, question, gt_answer, processor)
        hallucinated_tokenized = format_prompt(image, question, hallucinated_answer, processor)

        all_prompt_pairs[i] = (gt_tokenized, hallucinated_tokenized)

    return all_prompt_pairs


def get_activations_pyvene(pv_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = pv_model({"input_ids": prompt["input_ids"], 
                           #"pixel_values": prompt["pixel_values"], 
                           "output_hidden_states": True})[1]
    

    # Input → Self-Attention → o_proj → Residual Connection → LayerNorm → [hidden_states] → MLP → Residual Connection → LayerNorm → Next Layer
    # Collector hook: Input → Self-Attention → Collector (captures o_proj.input) → o_proj → Residual Connection -> ...

    # output of LayerNorm, input to MLP
    hidden_states = output.hidden_states #(B, S, H)
    hidden_states = torch.stack(hidden_states, dim=0).squeeze() #(L, B, S, H) or (L, S, H) if B = 1
    hidden_states = hidden_states.detach().cpu().numpy()

    # o_proj.input
    head_wise_hidden_states = [None] * len(collectors)
    for i, collector in enumerate(collectors):
        if collector.collect_state:
            states = torch.stack(collector.states, axis=0)
            head_wise_hidden_states[i] = states
        collector.reset()
    head_wise_hidden_states = torch.stack(head_wise_hidden_states, axis=0).squeeze().cpu().numpy()
    
    return hidden_states, head_wise_hidden_states


def save_activations(layer_wise_activations, head_wise_activations, name, chunk_size=100):
    print(f"Saving {name} layer wise activations in chunks")
    for i in range(0, len(layer_wise_activations), chunk_size):
        chunk = layer_wise_activations[i: i+chunk_size]
        np.save(f"/net/scratch2/steeringwheel/weiyitian/activations/{name}_layer_wise_{i // chunk_size}.npy", chunk)
    print(f"Saving {name} head wise activations in chunks")
    for i in range(0, len(head_wise_activations), chunk_size):
        chunk = head_wise_activations[i:i+chunk_size]
        np.save(f"/net/scratch2/steeringwheel/weiyitian/activations/{name}_head_wise_{i // chunk_size}.npy", chunk)


def plot_pca_comparison(gt_activations, hallucinated_activations, name, layer_num=32):
    cols = 4
    rows = math.ceil(layer_num / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    gt_activations = np.asarray(gt_activations)
    hallucinated_activations = np.asarray(hallucinated_activations)

    for layer_id in range(layer_num):
        gt_features = gt_activations[:, layer_id, :] #(prompt_num x hidden_dim)
        hallucinated_features = hallucinated_activations[:, layer_id, :] #(prompt_num x hidden_dim)

        combined_features = np.concatenate((gt_features, hallucinated_features), axis=0) # (2 * prompt_num x hidden_dim)
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(combined_features)

        pca_gt = pca_features[:gt_features.shape[0]]
        pca_hallu = pca_features[gt_features.shape[0]:]

        concat_x_values = np.concatenate([pca_gt[:, 0], pca_hallu[:, 0]])
        lower_bound = np.percentile(concat_x_values, 15)
        upper_bound = np.percentile(concat_x_values, 85)

        ax = axes[layer_id]
        ax.scatter(pca_gt[:, 0], pca_gt[:, 1], label='Ground Truth', color='blue', s=10)
        ax.scatter(pca_hallu[:, 0], pca_hallu[:, 1], label='Hallucinated', 
                   color='red', alpha=0.5, edgecolor='black', s=10)

        ax.set_title(f'Layer {layer_id}')
        ax.set_xlim(lower_bound, upper_bound)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()

    for i in range(layer_num, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(f'figures/{name}_GT_vs_Hallucinated_PCA.png')


def load_chunks(file_pattern):
    chunk_files = sorted(glob.glob(file_pattern), key=os.path.getmtime)
    chunks = [np.load(chunk_file) for chunk_file in chunk_files]
    return np.concatenate(chunks, axis=0)


def get_com_directions(num_layers, num_heads, train_idx, gt_head_wise_activations, hallucinated_head_wise_activations):
    com_directions = []
    for layer in tqdm(range(num_layers), desc="Getting com directions from layers"):
        for head in range(num_heads):
            gt_activations = gt_head_wise_activations[train_idx, layer, head, :] #(T, 128)
            hallucinated_activations = hallucinated_head_wise_activations[train_idx, layer, head, :] #(T, 128)

            gt_mass_mean = np.mean(gt_activations, axis=0)
            hallucinated_mass_mean = np.mean(hallucinated_activations, axis=0)

            com_directions.append(gt_mass_mean - hallucinated_mass_mean)
    
    com_directions = np.array(com_directions)
    return com_directions


def train_probes(seed, train_set_idxs, val_set_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads):
    X_train = np.concatenate([gt_head_wise_activations[train_set_idxs], hallucinated_head_wise_activations[train_set_idxs]], axis=0) # (2T, 32, 32, 128)
    y_train = np.concatenate([np.ones(len(train_set_idxs)), np.zeros(len(train_set_idxs))], axis=0)

    X_val = np.concatenate([gt_head_wise_activations[val_set_idxs], hallucinated_head_wise_activations[val_set_idxs]], axis=0) # (2V, 32, 32, 128)
    y_val = np.concatenate([np.ones(len(val_set_idxs)), np.zeros(len(val_set_idxs))], axis=0)

    all_head_accs = []
    for layer in tqdm(range(num_layers), desc="training probes on layers"): 
        for head in range(num_heads): 
            X_train_head = X_train[:, layer, head, :] # (2T, 128)
            X_val_head = X_val[:, layer, head, :] # (2V, 128)

            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train_head, y_train)

            y_val_pred = clf.predict(X_val_head)
            acc = accuracy_score(y_val, y_val_pred)
            all_head_accs.append(acc)

    all_head_accs = np.array(all_head_accs)
    return all_head_accs


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def get_top_heads(train_idxs, val_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads, seed, num_to_intervene, use_random_dir=False):
    all_head_accs_np = train_probes(seed, train_idxs, val_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_acc_idxs = np.argsort(all_head_accs_np.flatten())[::-1][:num_to_intervene]
    top_head_idxs = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_acc_idxs]

    if use_random_dir: 
        random_idxs = np.random.choice(num_heads * num_layers, num_heads * num_layers, replace=False)
        top_head_idxs = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_head_idxs


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


