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

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


def apply_llama_prompt(hallucination_type, question, gt_answer, model_answer):
    template = f'''You are an evaluator tasked with determining whether LLava’s answer is hallucinating. You do not have access to the image, only the following textual information. Your goal is to check if the **direct answer** to the question contradicts the ground truth. Additional details, even if extraneous, should be ignored as long as the direct answer is correct.

    ### Instructions:
    1. **Direct Answer Focus:** Identify the specific part of LLava’s response that directly answers the question.
    2. **Ground Truth Comparison:** Compare the direct answer with the provided ground truth.
    3. **Contradiction Check:** 
    - If the direct answer contradicts the ground truth, mark it as a hallucination.
    - If the direct answer is in agreement with the ground truth—even if extra details are present—mark it as not hallucinating.
    4. **Extra Details:** Do not consider additional information or context that does not alter the core answer.

    ---
    ### **Input Details**
    #### **Possible hallucination type:**
    {hallucination_type}

    #### **User Question:**
    {question}

    #### **Standard Human-Generated Answer:**
    {gt_answer}

    #### **Llava Response to Evaluate:**
    {model_answer}

    ---
    ### **Evaluation Criteria**
    1. **Does the response contain hallucination?** (Yes or No)
    - If **Yes**, identify the incorrect information and explain why it is not supported by the provided details.
    - If **No**, confirm that the response aligns with the given information.
    
    2. **Provide a confidence score** on a scale of **0 to 1**, where:
    - **0.0** = No confidence (high uncertainty)
    - **0.5** = Somewhat confident
    - **1.0** = Fully confident (high certainty in the evaluation)

    3. **Rate the response using the following scale:**
    - **0** → Very informative, no hallucination
    - **1** → Informative, no hallucination
    - **2** → Somewhat informative, no hallucination
    - **3** → Not informative, no hallucination
    - **4** → Very informative, with hallucination
    - **5** → Somewhat informative, with hallucination
    - **6** → Not informative, with hallucination

    ---
    ### **Expected Output Format (JSON)**
    **IMPORTANT:** Your entire response must be a single, valid JSON object with **no additional text or formatting whatsoever**. Do not include markdown, code fences, or any other commentary. 
    All double quotes inside string values must be properly escaped with a backslash (\\) (for example, use `\\\"` for quotes inside strings). This is especially important in the "explanation" field. The JSON object must exactly adhere to the following format:

    ```json
    {{
    "explanation": "Your explanation here",
    "hallucination": "yes" | "no",
    "rating": rating_value
    }}
    '''
    return template


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

        try:
            response = requests.get(image_url)
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)
        except Exception as e:
            print(f"Error processing image URL {image_url}: {e}")
            continue

        gt_tokenized = format_prompt(image, question, gt_answer, processor)
        hallucinated_tokenized = format_prompt(image, question, hallucinated_answer, processor)

        all_prompt_pairs[i] = (gt_tokenized, hallucinated_tokenized)

    return all_prompt_pairs


def get_activations_pyvene(pv_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = pv_model({"input_ids": prompt["input_ids"], 
                           "pixel_values": prompt["pixel_values"], 
                           "image_sizes": prompt["image_sizes"], 
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


def save_activations(gt_activations, hallucinated_activations, layer_type, chunk_size=100):
    print(f"Saving ground truth {layer_type} wise activations in chunks")
    for i in range(0, len(gt_activations), chunk_size):
        chunk = gt_activations[i: i+chunk_size]
        np.save(f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest/HaloQuest_gt_{layer_type}_wise_{i // chunk_size}.npy", chunk)
    print(f"Saving hallucinated {layer_type} wise activations in chunks")
    for i in range(0, len(hallucinated_activations), chunk_size):
        chunk = hallucinated_activations[i:i+chunk_size]
        np.save(f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest/HaloQuest_hallucinated_{layer_type}_wise_{i // chunk_size}.npy", chunk)


def plot_layer_pca_comparison(gt_activations, hallucinated_activations, name, layer_num=32):
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
        ax.scatter(pca_gt[:, 0], pca_gt[:, 1], label='Ground Truth', color='#f9bebb', s=10)
        ax.scatter(pca_hallu[:, 0], pca_hallu[:, 1], label='Hallucinated', 
                   color='#84c3b7', alpha=0.3, s=10)

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
    for layer in tqdm(range(num_layers), desc="training probs on layers"): 
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


def apply_interventions(dataset, test_idx, intervened_model, processor, file_name):
    results = []
    test_set = dataset["train"].select(test_idx)

    with torch.no_grad():
        for entry in tqdm(test_set, desc="Intervening prompts"):
            question = entry["question"]
            gt_answer = entry["gt_answer"]
            image_url = entry["image_url"]
            init_answer = entry["llava_model_answer"]
            hallucination_type = entry["hallucination_type"]

            conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
                ],
            }]

            response = requests.get(image_url)
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)

            prompt = processor.apply_chat_template(conversation=conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
            #_, output = intervened_model.generate({'input_ids': inputs["input_ids"]})
            _, output = intervened_model.generate({"input_ids": inputs["input_ids"],
                                                "attention_mask": inputs["attention_mask"],
                                               "pixel_values": inputs["pixel_values"], 
                                               "image_sizes": inputs["image_sizes"]}, 
                                               max_new_tokens=100)
            
            model_output = processor.decode(output[0], skip_special_tokens=True)
            model_answer = model_output.split("ASSISTANT:")[-1].strip()

            result_entry = {
                "before_steering": init_answer,
                "after_steering": model_answer,
                "question": question,
                "gt_answer": gt_answer,
                "image_url": image_url,
                "hallucination_type": hallucination_type
            }
            results.append(result_entry)

            print(f"Image url: {image_url}")
            print(f"Processed question: {question}")
            print(f"Ground truth: {gt_answer}")
            print(f"Before steering answer: {init_answer}")
            print(f"After steering answer: {model_answer}")
            print("=" * 50)
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_name, index=False)
    return results_df


def llama_evaluate(df, file_name):
    print("Loading Llama")
    tokenizer = AutoTokenizer.from_pretrained("/net/scratch2/steeringwheel/Llama-3.1-8B-Instruct")
    model = LlamaForCausalLM.from_pretrained("/net/scratch2/steeringwheel/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto")
    model.to("cuda:0")
    model = torch.compile(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for _, entry in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating entries"):
        image_url = entry["image_url"]
        question = entry["question"]
        gt_answer = entry["gt_answer"]
        init_answer = entry["before_steering"]
        model_answer = entry["after_steering"]
        hallucination_type = entry["hallucination_type"]

        filled_prompt = apply_llama_prompt(hallucination_type, question, gt_answer, model_answer)
        conversation = [
        {
            "role": "user",
            "content": filled_prompt,
        }]

        tokenized = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(tokenized.to("cuda:0"), max_new_tokens=256)

        input_len = tokenized.shape[-1]
        decoded_model_output = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

        match = re.search(r"({.*?})", decoded_model_output, re.DOTALL)
        model_output = match.group(1)
        
        try:
            parsed_output = json.loads(model_output)
        except Exception as e:
            print("Wrong output format from Llama")
            continue

        explanation = parsed_output["explanation"]
        hallucination = parsed_output["hallucination"]
        rating = parsed_output["rating"]

        result_entry = {
                "before_steering": init_answer,
                "after_steering": model_answer,
                "llama_hallucination_evaluation": hallucination,
                "llama_hallucination_analysis": explanation,
                "question": question,
                "gt_answer": gt_answer,
                "image_url": image_url,
                "hallucination_type": hallucination_type,
                "llama_hallucination_rating": rating
            }
        results.append(result_entry)

        print(f"Before steering answer: {init_answer}")
        print(f"After steering answer: {model_answer}")
        print(f"Llama evaluation: {hallucination}")
        print(f"Llama explanation: {explanation}")
        print(f"Image url: {image_url}")
        print(f"Processed question: {question}")
        print(f"Ground truth: {gt_answer}")
        print("=" * 50)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_name, index=False)
    return results_df


def get_ce_loss_owt(orig_model, intervened_model, processor, device='cuda', num_samples=100):
    dataset = load_dataset("stas/openwebtext-10k")['train']
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

    owt = dataset.map(lambda x: {'input_ids': torch.tensor(processor(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])
    
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()
    orig_losses, intervened_losses = [None] * len(rand_idxs), [None] * len(rand_idxs)
    with torch.no_grad(): 
        for i in tqdm(rand_idxs, desc="Computing CE loss on OWT"):
            input_ids = owt[i]['input_ids'][:, :128].to(device)

            orig_output = orig_model(input_ids=input_ids, labels=input_ids)
            orig_loss = orig_output.loss
            orig_losses[i] = orig_loss.item()

            intervened_output = intervened_model({'input_ids': input_ids, 'labels': input_ids})
            intervened_loss = intervened_output[1].loss
            intervened_losses[i] = intervened_loss.item()

    orig_ce, intervened_ce = np.mean(orig_losses), np.mean(intervened_losses)
    print(f"CE loss on OWT before steering: {orig_ce}")
    print(f"CE loss on OWT after steering: {intervened_ce}")

    return orig_ce, intervened_ce


def calculate_kl_divergence(probs1_tensor, probs2_tensor):
    return (probs1_tensor * (probs1_tensor / probs2_tensor).log()).sum()


def get_kl_divergence_owt(orig_model, intervened_model, processor, device='cuda', num_samples=100):
    dataset = load_dataset("stas/openwebtext-10k")['train']
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

    # len(owt) = num_samples
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(processor(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])
    
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()
    kl_divgs = [None] * len(rand_idxs)
    epsilon = 1e-10

    with torch.no_grad(): 
        for i in tqdm(rand_idxs, desc="Computing KL divergence on OWT"):
            input_ids = owt[i]['input_ids'][:, :128].to(device)

            orig_output = orig_model(input_ids=input_ids)
            orig_logits = orig_output.logits.cpu().type(torch.float32)
            orig_probs = F.softmax(orig_logits, dim=-1)

            intervened_output = intervened_model({'input_ids': input_ids})[1]
            intervened_logits = intervened_output.logits.cpu().type(torch.float32)
            intervened_probs = F.softmax(intervened_logits, dim=-1)

            orig_probs = orig_probs.clamp(min=epsilon)
            intervened_probs = intervened_probs.clamp(min=epsilon)
            kl_div = calculate_kl_divergence(orig_probs, intervened_probs) / (input_ids.shape[-1] * input_ids.shape[-2])
            kl_divgs[i] = kl_div.item()

    kl = np.mean(kl_divgs)
    print(f"KL-divergence between original model and the steered model: {kl}")
    return kl


def get_hallucination_num_after_steering(llama_result_file):
    dataset = load_dataset("csv", data_files=llama_result_file)["train"]

    total_entries = len(dataset)
    hallucination_count = sum(1 for entry in dataset if entry["llama_hallucination_evaluation"] == "yes")
    hallucination_proportion = hallucination_count / total_entries
    
    print(f"Total entries: {total_entries}")
    print(f"Hallucination count: {hallucination_count}")
    print(f"Proportion: {hallucination_proportion:.2%}")

    return total_entries, hallucination_count, hallucination_proportion


def eval_ce_kl_owt(orig_model, intervened_model, processor, top_heads_by_layer, llama_result, file_name, device='cuda', num_samples=100):
    orig_ce, intervened_ce = get_ce_loss_owt(orig_model, intervened_model, processor, device, num_samples)
    kl = get_kl_divergence_owt(orig_model, intervened_model, processor, device, num_samples)
    total_entries, hallucination_count, hallucination_proportion = get_hallucination_num_after_steering(llama_result)

    results = {
        "original_ce_loss": orig_ce,
        "intervened_ce_loss": intervened_ce,
        "kl_divergence": kl,
        "total_entries": total_entries,
        "hallucination_entries_after_steering": hallucination_count,
        "hallucination_proportion_after_steering": hallucination_proportion,
        "top_heads_by_layer": top_heads_by_layer
    }


    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {file_name}")


def plot_layer_head_PCA(gt_head_wise_activations, hallucinated_head_wise_activations, top_heads_by_layer, top_num_heads, file_name):
    cols = 4
    rows = math.ceil(top_num_heads / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    gt_head_wise_activations = np.asarray(gt_head_wise_activations)
    hallucinated_head_wise_activations = np.asarray(hallucinated_head_wise_activations)
    
    ax_ind = 0
    for layer, heads in top_heads_by_layer.items():
        for head in heads:
            gt_features = gt_head_wise_activations[:, layer, head, :] #(prompt_num x hidden_dim / head_num)
            hallucinated_features = hallucinated_head_wise_activations[:, layer, head, :] #(prompt_num x hidden_dim / head_num)

            combined_features = np.concatenate((gt_features, hallucinated_features), axis=0) # (2 * prompt_num x hidden_dim / head_num)
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(combined_features)

            pca_gt = pca_features[:gt_features.shape[0]]
            pca_hallu = pca_features[gt_features.shape[0]:]

            ax = axes[ax_ind]
            ax.scatter(pca_gt[:, 0], pca_gt[:, 1], label='Ground Truth', color='#f9bebb', s=10)
            ax.scatter(pca_hallu[:, 0], pca_hallu[:, 1], label='Hallucinated', 
                    color='#84c3b7', alpha=0.3, s=10)

            # concat_x_values = np.concatenate([pca_gt[:, 0], pca_hallu[:, 0]])
            # lower_bound = np.percentile(concat_x_values, 15)
            # upper_bound = np.percentile(concat_x_values, 85)
            # ax.set_xlim(lower_bound, upper_bound)

            ax.set_title(f'Layer {layer} Head {head}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend()

            ax_ind += 1
    
    for ax in axes[top_num_heads:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(f'{file_name}_layer_head_PCA.png')


def ignore_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*The use of `x.T` on tensors of dimension other than 2.*",
        category=UserWarning
    )
