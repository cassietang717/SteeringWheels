from tqdm import tqdm
import numpy as np
from collections import defaultdict

import argparse
from einops import rearrange

import pyvene as pv
import torch
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from interveners import wrapper, ITI_Intervener
from utils import get_prompt_pairs, load_chunks, get_com_directions, get_top_heads, layer_head_to_flattened_idx, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llava_7B')
    parser.add_argument('--dataset_name', type=str, default='HaloQuest')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_heads', type=int, default=32, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=20, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)
    args = parser.parse_args()
    
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataset
    if args.dataset_name == "HaloQuest": 
        dataset = load_dataset("csv", data_files="../HaloQuest/output/HaloQuest_llama.csv")
        dataset = dataset.filter(lambda entry: entry["llama_hallucination_evaluation"] == "yes")
        formatter = get_prompt_pairs
    else:
        raise ValueError(f"Wrong Dataset Choice: {args.dataset_name}")
    
    fold_idxs = np.array_split(np.arange(len(dataset["train"])), args.num_fold)

    # load models
    print("Loading models")
    processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model = model_llava.language_model
    device = f"cuda:{args.device}"
    model = model.to(device)

    if processor.pad_token is None:
        processor.pad_token = processor.eos_token
    model.generation_config.pad_token_id = processor.pad_token_id

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_heads = model.config.num_key_value_heads # unique key-value heads

    # load activations
    gt_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/{args.dataset_name}_gt_layer_wise_*.npy"
    gt_layer_wise_activations = load_chunks(gt_layer_wise_pattern) #(P, 33, 4096)
    hallucinated_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/{args.dataset_name}_hallucinated_layer_wise_*.npy"
    hallucinated_layer_wise_activations = load_chunks(hallucinated_layer_wise_pattern) #(P, 33, 4096)

    gt_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/{args.dataset_name}_gt_head_wise_*.npy"
    gt_head_wise_activations = load_chunks(gt_head_wise_pattern) #(P, 32, 4096)
    hallucinated_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/{args.dataset_name}_hallucinated_head_wise_*.npy"
    hallucinated_head_wise_activations = load_chunks(hallucinated_head_wise_pattern) #(P, 32, 4096)

    gt_head_wise_activations = rearrange(gt_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads) #(P, 33, 32, 128)
    hallucinated_head_wise_activations = rearrange(hallucinated_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads) #(P, 32, 32, 128)
    all_head_wise_activations = torch.cat([gt_head_wise_activations, hallucinated_head_wise_activations], dim=0)  #(2P, 33, 32, 128)

    print("Successfully loaded all activation chunks")


    # k-fold validation
    results = [None]* args.num_fold
    for i in range(args.num_fold):
        print(f"Running fold {i}")
        fold_idxs_copy = fold_idxs.copy()
        test_idx = fold_idxs_copy.pop(i)
        train_idxs = np.concatenate(fold_idxs_copy)

        # separate into train and val sets
        shuffled_train_idx = np.random.permutation(train_idxs)
        split_idx = int(len(train_idxs) * args.val_ratio)
        train_set_idxs, val_set_idxs = np.split(shuffled_train_idx, [split_idx])

        # get mean steering direction
        #(L * H, 128)
        com_directions = get_com_directions(num_layers, num_heads, train_idxs, val_set_idxs, gt_head_wise_activations, hallucinated_head_wise_activations)
        # get top k impactful heads
        top_head_idxs = get_top_heads(train_set_idxs, val_set_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
        print("Heads to be intervened: ", sorted(top_head_idxs))
        print(f'Intervener Strength is {args.alpha}')

        top_heads_by_layer = defaultdict(list)
        for layer, head in top_head_idxs:
            top_heads_by_layer[layer].append(head)


        interveners = []
        pv_configs = []
        for layer, heads in tqdm(top_heads_by_layer.items()):
            direction = torch.zeros(head_dim * num_heads).to("cpu")
            for head in heads:
                dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer, head, num_heads)], dtype=torch.float32).to("cpu")
                dir = dir / torch.norm(dir)

                cur_head_activations = torch.tensor(all_head_wise_activations[:, layer, head, :], dtype=torch.float32).to("cpu")
                proj_vals = cur_head_activations @ dir.T # projection of current head activations onto dir
                proj_val_std = torch.std(proj_vals) # how much activations naturally vary in dir

                direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
            
            intervener = ITI_Intervener(direction, args.alpha)
            interveners.append(intervener)
            pv_configs.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(intervener),
            })
        
        intervened_model = pv.IntervenableModel(pv_configs, model)

        filename = f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'
        if args.use_center_of_mass:
            filename += '_com'
        if args.use_random_dir:
            filename += '_random'
        
        curr_fold_results = evaluate(
            models={args.model_name: intervened_model},
            metric_names=['judge', 'info', 'mc','bleurt'],
            input_path=f'splits/fold_{i}_test_seed_{args.seed}.csv',
            output_path=f'results_dump/answer_dump/{filename}.csv',
            summary_path=f'results_dump/summary_dump/{filename}.csv',
            device="cuda", 
            interventions=None, 
            intervention_fn=None, 
            instruction_prompt=args.instruction_prompt,
            judge_name=args.judge_name, 
            info_name=args.info_name,
            separate_kl_device='cuda',
            orig_model=model
        )

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results[i] = curr_fold_results
    
    results = np.array(results)
    final_result = results.mean(axis=0)
    print("Final result")
    print(final_result)
    

if __name__ == "__main__":
    main()