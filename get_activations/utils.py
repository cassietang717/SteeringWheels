# Utils to work with pyvene

import os
import sys
sys.path.insert(0, "TruthfulQA")
from t5.evaluation import metrics
from evaluate import load as load_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
# import llama
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import warnings
from einops import rearrange
# from transformers import LlavaNextProcessor
from baukit import Trace, TraceDict
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
from functools import partial
import json
import re

ENGINE_MAP = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    'llava_1.6': "/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf",
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
}

from truthfulqa.utilities import (
    format_prompt,
    format_prompt_with_answer_strings,
    split_multi_answer,
    format_best,
    find_start,
)
from truthfulqa.presets import preset_map, COMPARE_PRIMER
from truthfulqa.models import find_subsequence, set_columns, MC_calcs
from truthfulqa.evaluate import format_frame, data_to_dict


def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def format_truthfulqa_end_q(question, choice, rand_question): 
    return f"Q: {question} A: {choice} Q: {rand_question}"


def tokenized_tqa(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0: 
                print(prompt)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(label)
    
    return all_prompts, all_labels

def tokenized_tqa_gen_end_q(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset[rand_idx]['question']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories

def tokenized_tqa_gen(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)): 
        question = dataset[i]['question']
        category = dataset[i]['category']

        for j in range(len(dataset[i]['correct_answers'])): 
            answer = dataset[i]['correct_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)
        
        for j in range(len(dataset[i]['incorrect_answers'])):
            answer = dataset[i]['incorrect_answers'][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)
        
    return all_prompts, all_labels, all_categories


def tokenized_tqa_wice(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        claim = dataset[i]['claim']
        evidence = dataset[i]['evidence']
        gt_label = dataset[i]['label']
        # labels = ['reason','alt']
        # supportive = ["Not Support.", "Support."]
        true_false = [1,0]
        for label in true_false: 
            if label == 1:
                sup_label = dataset[i]['ground_truth']
            # elif gt_label == "not_supported" and label == 1:
            #     sup_label = "The evidence does not support the claim."
            else:
                sup_label = dataset[i]['reason']

            conversation = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that supports claims based on evidence."
                },
                {
                    "role": "user",
                    "content": [
                        # {"type": "image", "data": image_data},  # Uncomment this if an image is included
                        {"type": "text", "text": f"Claim: {claim}"},
                        {"type": "text", "text": f"Evidence: {evidence}"},
                        {"type": "text", "text": f"{sup_label}"}
                    ],
                }
            ]
            prompt = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True)
            input = tokenizer(images=None, text=prompt, return_tensors="pt").to("cuda:0")

            all_prompts.append(input.input_ids)
            all_labels.append(label)
    
    return all_prompts, all_labels

def tokenized_tqa_med(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    indices = random.sample(range(len(dataset)), 2500)
    for i in indices:
        knowledge = dataset[i]['Knowledge']
        question = dataset[i]['Question']
        true_false = [1,0]
        labels = ["Ground Truth", "Hallucinated Answer"]
        for sample_label,label in zip(labels,true_false): 
            print(sample_label)
            summary = dataset[i][sample_label]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        # {"type": "image", "data": image_data},  # Uncomment this if an image is included
                        {"type": "text", "text": f"Knowledge: {knowledge}"},
                        {"type": "text", "text": f"Question: {question}"},
                        {"type": "text", "text": f"Answer: {summary}"}
                    ],
                }
            ]
            prompt = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True)
            input = tokenizer(images=None, text=prompt, return_tensors="pt").to("cuda:0")

            all_prompts.append(input.input_ids)
            all_labels.append(label)
    
    return all_prompts, all_labels

def judgement_accuracy(frame):
    improvement = frame["hall_label"] != frame["answer_after_steering"]
    return sum(improvement)/len(improvement)

def run_bleurt(frame):
    bleurt = load_metric("bleurt", cache_dir=None)
    for calc in ['max', 'diff', 'acc']:
        col_name = '{0} BLEURT {1}'.format('llava', calc)
        if col_name not in frame.columns:
            frame[col_name] = np.nan
    results = {} 
    for idx in tqdm(frame.index,desc='run bleurt'):
        scores_true = bleurt.compute(
                predictions=[frame.loc[idx, 'answer_after_steering'] + frame.loc[idx, 'reason_after_steering']], 
                references=[frame.loc[idx, 'gt_label'] + frame.loc[idx, 'gt_answer']]
            )['scores']
        scores_false = bleurt.compute(
                predictions=[frame.loc[idx, 'answer_after_steering'] + frame.loc[idx, 'reason_after_steering']], 
                references=[frame.loc[idx, 'hall_label'] + frame.loc[idx, 'hall_answer']]
            )['scores']       
        for calc in ['max', 'diff', 'acc']:
            col_name = '{0} BLEURT {1}'.format('llava', calc)
            if calc == 'max':
                frame.loc[idx, col_name] = max(scores_true)
            elif calc == 'diff':
                frame.loc[idx, col_name] = max(scores_true) - max(scores_false)
            elif calc == 'acc':
                frame.loc[idx, col_name] = int(max(scores_true) > max(scores_false))
            print(frame.loc[idx, col_name])
    for calc in ['max', 'diff', 'acc']:
        col_name = '{0} BLEURT {1}'.format('llava', calc)
        results[col_name] = sum(frame[col_name])/len(frame[col_name])
        print(f'Average {col_name} {results[col_name]}')
            
    return frame, results


def get_com_directions_img(num_layers, num_heads, train_idx, gt_head_wise_activations, hallucinated_head_wise_activations):
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

def train_probes_img(seed, train_set_idxs, val_set_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads):
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


def get_top_heads_img(train_idxs, val_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads, seed, num_to_intervene, use_random_dir=False):
    all_head_accs_np = train_probes_img(seed, train_idxs, val_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_acc_idxs = np.argsort(all_head_accs_np.flatten())[::-1][:num_to_intervene]
    top_head_idxs = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_acc_idxs]

    if use_random_dir: 
        random_idxs = np.random.choice(num_heads * num_layers, num_heads * num_layers, replace=False)
        top_head_idxs = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_head_idxs

def get_llama_activations_bau(model, prompt, device): 
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS) as ret:
        # with TraceDict(model, HEADS+MLPS, retain_input=True) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
    hidden_states = hidden_states.detach().cpu().numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

def get_llama_logits(model, prompt, device): 

    model.eval()
    with torch.no_grad(): 
        prompt = prompt.to(device)
        logits = model(prompt).logits
        logits = logits.detach().cpu()
        return logits

def save_probes(probes, path): 
    """takes in a list of sklearn lr probes and saves them to path"""
    with open(path, 'wb') as f: 
        pickle.dump(probes, f)

def load_probes(path): 
    """loads a list of sklearn lr probes from path"""
    with open(path, 'rb') as f: 
        probes = pickle.load(f)
    return probes

# -- TruthfulQA helper functions -- # 

def tqa_run_answers(frames, engine, tag, preset, model=None, tokenizer=None, verbose=True, device=None, cache_dir=None, interventions={}, intervention_fn=None, instruction_prompt="default", many_shot_prefix=None):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""
    prefix = '''
                Your task is to evaluate if a claim is supported by a provided evidence.

                Choose your conclusion from the options **[supported, not_supported]**, and present your reason in **one sentence**.

                ### **Input:**

                #### Claim:
                {claim}

                #### Evidence:
                {evidence}

                
                ### **Expected Output Format (JSON)**
                Your response **must** be valid **JSON** output only. Do not include explanations, preamble, or any extra text.
                ```json
                {{
                    "answer": "Choose from [supported, not_supported]",
                    "reason": "Clearly state in one sentence why the evidence supports or does not support the claim."
                }}
                ```
            '''
    sequences = []
    with torch.no_grad():
        for idx in tqdm(frames.index, desc="tqa_run_answers"): 
            frame = frames.iloc[idx]
            prompt_item = prefix.format(claim=frame['claim'], evidence=frame['evidence'])
            chat=[{
                    "role": "user",
                    "content": [{"type": "text", "text": f"{prompt_item}"}]
                },]      
            prompt = tokenizer.apply_chat_template(conversation=chat, add_generation_prompt=True)
            inputs = tokenizer(text=prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to('cuda')
            _, output = model.generate(
                {"input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]},
                use_cache=True,
                max_new_tokens=200,
                do_sample=True,  # Control randomness
                # num_beams=4,
                # # temperature=override_temperature if override_temperature is not None else 1,
                temperature=1e-5,
                # # top_p=0.9,
                num_return_sequences=1,  # ✅ Return multiple responses
                stopping_criteria=None,
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
            match = re.search(r"({.*?})", response, re.DOTALL)
            if match:
                try:
                    model_output = match.group(1)
                    response_json = json.loads(model_output)
                    reason = response_json.get("reason", "N/A")
                    answer = response_json.get("answer", "N/A")
                except json.JSONDecodeError:
                    print(f"⚠️ Warning: JSON decoding failed for claim '{frame['claim']}'. Skipping...")
                    continue
            else:
                print(f"⚠️ Warning: No valid JSON found in response for claim '{frame['claim']}'. Skipping...")
                continue

            result_entry = {
                "claim": frame['claim'],
                "evidence": frame['evidence'],
                "gt_label": frame['label'],
                "gt_answer": frame['ground_truth'],
                "hall_label": frame['answer'],
                "hall_answer": frame['reason'],
                "answer_after_steering": answer,
                "reason_after_steering": reason
            }

            sequences.append(result_entry)
            print(f"Answer before steering: {frame['answer']}, {frame['reason']}")
            print(f"Answer after steering: {answer}, {reason}")

            # --- intervention code --- #

    if device:
        torch.cuda.empty_cache()

    return pd.DataFrame(sequences)

def run_ce_loss(model_key, model=None, tokenizer=None, device='cuda', interventions={}, intervention_fn=None, num_samples=100): 

    # load owt text
    # note this is tokenized with llama tokenizer
    dataset = load_dataset("stas/openwebtext-10k")['train']
    dataset = dataset.filter(lambda x: x['text'].startswith("http") is False)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

    # tokenize
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])
    
    # # define intervention
    # def id(head_output, layer_name):
    #     return head_output
    
    # if interventions == {}:
    #     layers_to_intervene = []
    #     intervention_fn = id
    # else: 
    #     layers_to_intervene = list(interventions.keys())
    #     intervention_fn = partial(intervention_fn, start_edit_location=0)

    losses = []
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()
    with torch.no_grad(): 
        for i in tqdm(rand_idxs, desc="run_ce_loss"):

            input_ids = owt[i]['input_ids'][:, :128].to(device)
            
            # with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
            _, loss = model({'input_ids': input_ids, 'labels': input_ids})
            loss = loss.loss
            
            losses.append(loss.item())
    
    return np.mean(losses)

def run_kl_wrt_orig(model_key, model=None, tokenizer=None, device='cuda', interventions={}, intervention_fn=None, num_samples=100, separate_kl_device=None, orig_model=None): 
    # note this is tokenized with llama tokenizer
    dataset = load_dataset("stas/openwebtext-10k")['train']
    dataset = dataset.filter(lambda x: x['text'].startswith("http") is False)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

    # tokenize
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])
    
    # # define intervention
    # def id(head_output, layer_name):
    #     return head_output
    
    # if interventions == {}:
    #     layers_to_intervene = []
    #     intervention_fn = id
    # else: 
    #     layers_to_intervene = list(interventions.keys())
    #     intervention_fn = partial(intervention_fn, start_edit_location=0)

    kl_divs = []
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()

    if separate_kl_device is not None: 
        # orig_model = AutoModelForCausalLM.from_pretrained(ENGINE_MAP[model_key], torch_dtype=torch.float16, low_cpu_mem_usage=True)
        orig_model.to('cuda')

    with torch.no_grad(): 
        epsilon = 1e-10  # Small value to avoid division by zero
        for i in tqdm(rand_idxs, desc="run_kl_wrt_orig"):
            input_ids = owt[i]['input_ids'][:, :128].to(device)
            if separate_kl_device is not None: 
                orig_logits = orig_model(input_ids.to('cuda'))
                orig_logits = orig_logits.logits.cpu().type(torch.float32)
            else: 
                _, orig_logits = model({'input_ids': input_ids})
                orig_logits = orig_logits.logits.cpu().type(torch.float32)
                
            orig_probs = F.softmax(orig_logits, dim=-1)

            # with TraceDict(model, layers_to_intervene, edit_output=intervention_fn) as ret:
            _, logits = model({'input_ids': input_ids})
            logits = logits.logits.cpu().type(torch.float32)
            probs  = F.softmax(logits, dim=-1)

            # Add epsilon to avoid division by zero
            probs = probs.clamp(min=epsilon)
            orig_probs = orig_probs.clamp(min=epsilon)            
            kl_div = (orig_probs * (orig_probs / probs).log()).sum() / (input_ids.shape[-1] * input_ids.shape[-2])
            kl_divs.append(kl_div.item())

    return np.mean(kl_divs)

def alt_tqa_evaluate(models, metric_names, input_path, output_path, summary_path, device='cpu', verbose=False, preset='qa', interventions={}, intervention_fn=None, cache_dir=None, separate_kl_device=None, orig_model=None, instruction_prompt="default", many_shot_prefix=None, judge_name=None, info_name=None, tokenizer = None): 
    """
    Inputs:
    models: a dictionary of the form {model_name: model} where model is a HF transformer # TODO: doesn't work with models other than llama right now
    metric_names: a list of metric names to evaluate (ex: ['mc', 'judge', 'info', 'bleu'])
    input_path: where to draw TruthfulQA questions from
    output_path: where to store model outputs and full metric outputs
    summary_path: where to store metric summaries
    interventions: a dictionary of the form {layer_name: [(head, direction, projected_mean, projected_std)]}
    intervention_fn: a function that takes in a head output and a layer name and returns the intervened output

    Outputs a pd dataframe with summary values
    """
    dataset = pd.read_csv(input_path)

    # print("ASSUMES OPENAI_API_KEY ENVIRONMENT VARIABLE IS SET")
    # import os
    # openai.api_key = os.environ.get('OPENAI_API_KEY')
    
    for mdl in models.keys(): 
        # llama
        if 'llama' in mdl or 'alpaca' in mdl or 'vicuna' or 'llava' in mdl:
            assert models[mdl] is not None, 'must provide llama model'
            llama_model = models[mdl]
            llama_tokenizer = tokenizer
            if 'judge' in metric_names or 'info' in metric_names:
                questions = tqa_run_answers(dataset, ENGINE_MAP[mdl], mdl, preset, model=llama_model, tokenizer=llama_tokenizer,
                                device=device, cache_dir=cache_dir, verbose=verbose,
                                interventions=interventions, intervention_fn=intervention_fn, instruction_prompt=instruction_prompt, many_shot_prefix=many_shot_prefix)

            questions.to_csv(output_path, index=False)
    frames, results = run_bleurt(questions)
    results['acc'] = judgement_accuracy(frames)
    # calculate cross entropy loss on owt and kl wrt to original unedited on owt
    results['CE Loss'] = np.nan
    results['KL wrt Orig'] = np.nan

    for model_key in models.keys(): 
        # if model_key not in questions.columns:
        #     warnings.warn("Answers missing for {0}!".format(model_key), stacklevel=2)
        #     continue
        if 'llama' in model_key or 'alpaca' in model_key or 'vicuna' or 'llava' in model_key:
            ce_loss = run_ce_loss(model_key, model=llama_model, tokenizer=llama_tokenizer, device=device, interventions=interventions, intervention_fn=intervention_fn)
            kl_wrt_orig = run_kl_wrt_orig(model_key, model=llama_model, tokenizer=llama_tokenizer, device=device, interventions=interventions, intervention_fn=intervention_fn, separate_kl_device=separate_kl_device, orig_model=orig_model)

        results['CE Loss'] = ce_loss
        results['KL wrt Orig'] = kl_wrt_orig
    results = pd.DataFrame([results])
    results.to_csv(summary_path, index=False)
    
    return results

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def train_probes(seed, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads):
    
    all_head_accs = []
    probes = []

    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis = 0)
    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis = 0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)

    for layer in tqdm(range(num_layers), desc="train_probes"): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
    
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np

def get_top_heads(train_idxs, val_idxs, separated_activations, separated_labels, num_layers, num_heads, seed, num_to_intervene, use_random_dir=False):

    probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, separated_activations, separated_labels, num_layers=num_layers, num_heads=num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_heads, all_head_accs_np.reshape(num_heads*num_layers)

def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 

    interventions = {}
    for layer, head in top_heads: 
        interventions[f"language_model.model.layers.{layer}.self_attn.head_out"] = []

    for layer, head in top_heads:
        if use_center_of_mass: 
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(128,))
        else: 
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:] # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"language_model.model.layers.{layer}.self_attn.head_out"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads: 
        interventions[f"language_model.model.layers.{layer}.self_attn.head_out"] = sorted(interventions[f"language_model.model.layers.{layer}.self_attn.head_out"], key = lambda x: x[0])
    return interventions

def get_separated_activations(labels, head_wise_activations): 

    # separate activations by question
    dataset = []
    with open('/home/dwlyu/steeringwheel/hall_eval1.jsonl', 'r') as json_file:
            for line in json_file:
                data = json.loads(line) 
                dataset.append(data)
    df = dataset
    actual_labels = []
    for i in range(len(dataset)):
        actual_labels.append(2)

    idxs_to_split_at = np.cumsum([x for x in actual_labels]) 
    print(idxs_to_split_at)       

    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    assert len(separated_labels) == len(actual_labels)
    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    return separated_head_wise_activations[:-1], separated_labels, idxs_to_split_at

def get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels): 

    com_directions = []

    for layer in tqdm(range(num_layers), desc="get_com_directions"): 
        for head in range(num_heads): 
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            usable_head_wise_activations = np.concatenate([separated_head_wise_activations[i][:,layer,head,:] for i in usable_idxs], axis=0)
            usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions) #num_layer x num_heads x hidden_state

    return com_directions
