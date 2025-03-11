import torch
from einops import rearrange
import numpy as np
# import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
# from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import glob
import os
import sys
sys.path.append('../')
import json
# Specific pyvene imports
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, get_top_heads_img, get_com_directions_img
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llava_1.6', help='prefix to model name')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix to model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=5, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=True)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--use_image', type=int, default=0)
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = []
    with open('/home/dwlyu/steeringwheel/hall_eval1.jsonl', 'r') as json_file:
            for line in json_file:
                data = json.loads(line) 
                dataset.append(data)
    df = dataset
    # get two folds using numpy

    ################### Llava model initialization
    # create model
    tokenizer = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model_llava
    device = "cuda"
    model = model.to(device)
    if tokenizer.tokenizer.pad_token is None:
            tokenizer.tokenizer.pad_token = tokenizer.eos_token
    model.language_model.generation_config.pad_token_id = tokenizer.tokenizer.pad_token_id

    # define number of layers and heads
    num_layers = model.language_model.config.num_hidden_layers
    num_heads = model.language_model.config.num_attention_heads
    hidden_size = model.language_model.config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_heads = model.language_model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    #################### Load Activation Packages

    def load_chunks(file_pattern):
        chunk_files = sorted(glob.glob(file_pattern), key=os.path.getmtime)
        print(chunk_files)
        chunks = [np.load(chunk_file) for chunk_file in chunk_files]
        return np.concatenate(chunks, axis=0)
    
    if args.use_image:
        args.dataset_name_img = "HaloQuest"

        gt_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_{args.dataset_name_img}/{args.dataset_name_img}_gt_layer_wise_*.npy"
        gt_layer_wise_activations = load_chunks(gt_layer_wise_pattern) #(P, 33, 4096)
        hallucinated_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_{args.dataset_name_img}/{args.dataset_name_img}_hallucinated_layer_wise_*.npy"
        hallucinated_layer_wise_activations = load_chunks(hallucinated_layer_wise_pattern) #(P, 33, 4096)

        gt_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_{args.dataset_name_img}/{args.dataset_name_img}_gt_head_wise_*.npy"
        gt_head_wise_activations = load_chunks(gt_head_wise_pattern) #(P, 32, 4096)
        hallucinated_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/VC_{args.dataset_name_img}/{args.dataset_name_img}_hallucinated_head_wise_*.npy"
        hallucinated_head_wise_activations = load_chunks(hallucinated_head_wise_pattern) #(P, 32, 4096)

        gt_head_wise_activations = rearrange(gt_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads) #(P, 33, 32, 128)
        hallucinated_head_wise_activations = rearrange(hallucinated_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads) #(P, 32, 32, 128)
        all_head_wise_activations = np.concatenate([gt_head_wise_activations, hallucinated_head_wise_activations], axis=0) 

    else:
         
        layer_wise_pattern = f"/net/scratch2/steeringwheel/dwlyu/features/llava_7B_wice_eval1_layer_wise_*.npy"
        all_layer_wise_activations = load_chunks(layer_wise_pattern)

        # Load head-wise activations
        head_wise_pattern = f"/net/scratch2/steeringwheel/dwlyu/features/llava_7B_wice_eval1_head_wise_*.npy"
        all_head_wise_activations = load_chunks(head_wise_pattern)
        # load activations 

        labels = np.load(f"/net/scratch2/steeringwheel/dwlyu/features/llava_7B_wice_eval1_labels.npy")
        head_wise_activations = rearrange(all_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

        # tuning dataset: no labels used, just to get std of activations along the direction
        activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
        tuning_activations = rearrange(all_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        tuning_labels = np.load(f"/net/scratch2/steeringwheel/dwlyu/features/llava_7B_wice_eval1_labels.npy")
        separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
        assert len(separated_head_wise_activations) == len(dataset)


    #################### TODO: resize the flattened activation states to match each question for multiple choice

    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):
        test_size = int(0.1 * len(dataset))
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        test_idxs = indices[:test_size]
        train_idxs = indices[test_size:]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        pd.DataFrame([dataset[idx] for idx in train_set_idxs]).to_csv(f'wice_eval1_fold_{i}_train_seed_{args.seed}.csv', index=False)
        pd.DataFrame([dataset[idx] for idx in val_set_idxs]).to_csv(f'wice_eval1_fold_{i}_val_seed_{args.seed}.csv', index=False)
        pd.DataFrame([dataset[idx] for idx in test_idxs]).to_csv(f'wice_eval1_fold_{i}_test_seed_{args.seed}.csv', index=False)

        # get directions
        ################## Get Steering Directions
        if args.use_image:
            com_directions = get_com_directions_img(num_layers, num_heads, train_idxs, gt_head_wise_activations, hallucinated_head_wise_activations)
        # get top k impactful heads
            top_heads = get_top_heads_img(train_set_idxs, val_set_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
        else:
            if args.use_center_of_mass:
                com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels) # Shape 32x4096
            else:
                com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels) # Shape 32x4096
                com_directions = np.random.rand(*com_directions.shape)

            top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir) #Train logistic regression in trainset and return heads with top accuracy on Validation set
            np.save("/home/dwlyu/honest_llama/honest_llama/get_activations/wice_eval1_heads.npy",top_heads)
            np.save("/home/dwlyu/honest_llama/honest_llama/get_activations/wice_eval1_acc.npy",probes)

        print("Heads intervened: ", sorted(top_heads))
        ################# Add Steering Vector
        interveners = []
        pv_config = []
        top_heads_by_layer = {}
        for layer, head, in top_heads:
            if layer not in top_heads_by_layer:
                top_heads_by_layer[layer] = []
            top_heads_by_layer[layer].append(head)
        for layer, heads in top_heads_by_layer.items():
            direction = torch.zeros(head_dim * num_heads).to("cpu")
            for head in heads:
                dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer, head, num_heads)], dtype=torch.float32).to("cpu")
                dir = dir / torch.norm(dir)
                if args.use_image == 1:
                    activations = torch.tensor(all_head_wise_activations[:, layer, head, :], dtype=torch.float32).to("cpu")
                else:
                    activations = torch.tensor(tuning_activations[:,layer,head,:], dtype=torch.float32).to("cpu")
                proj_vals = activations @ dir.T
                proj_val_std = torch.std(proj_vals)
                direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
            print(f'Intervener Strength is {args.alpha}')
            intervener = ITI_Intervener(direction, args.alpha) #head=-1 to collect all head activations, multiplier doens't matter
            interveners.append(intervener)
            pv_config.append({
                "component": f"language_model.model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        intervened_model = pv.IntervenableModel(pv_config, model)

        filename = f'{args.model_prefix}{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'

        if args.use_center_of_mass:
            filename += '_com'
        if args.use_random_dir:
            filename += '_random'

        os.makedirs('results_dump/answer_dump', exist_ok=True)
        os.makedirs('results_dump/summary_dump', exist_ok=True)
                                
        curr_fold_results = alt_tqa_evaluate(
            models={args.model_name: intervened_model},
            metric_names=['judge', 'info', 'mc','bleurt'],
            input_path=f'wice_eval1_fold_{i}_test_seed_{args.seed}.csv',
            output_path=f'results_dump/answer_dump/{filename}_image_rand.csv',
            summary_path=f'results_dump/summary_dump/{filename}_image_rand.csv',
            device="cuda", 
            interventions=None, 
            intervention_fn=None, 
            instruction_prompt=args.instruction_prompt,
            judge_name=args.judge_name, 
            info_name=args.info_name,
            separate_kl_device='cuda',
            orig_model=model,
            tokenizer= tokenizer,
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)
    print(results)
    print(final)

    # print(f'alpha: {args.alpha}, heads: {args.num_heads}, True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()
