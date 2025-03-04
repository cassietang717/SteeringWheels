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
from utils import ignore_warnings, load_chunks, get_com_directions, get_top_heads, layer_head_to_flattened_idx
from utils import apply_interventions, llama_evaluate, eval_ce_kl_owt, plot_layer_head_PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llava_7B')
    parser.add_argument('--dataset_name', type=str, default='HaloQuest')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--top_num_heads', type=int, default=32, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=0, help='alpha, intervention strength')
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=True)
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()
    
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataset
    if args.dataset_name == "HaloQuest": 
        dataset = load_dataset("csv", data_files="../HaloQuest/output/HaloQuest_llama.csv")
        dataset = dataset.filter(lambda entry: entry["llama_hallucination_evaluation"] == "yes")
        dataset["train"] = dataset["train"].select(range(25))
    else:
        raise ValueError(f"Wrong Dataset Choice: {args.dataset_name}")
    
    dataset_size = len(dataset["train"])

    ignore_warnings()

    # load models
    print("Loading models")
    processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    device = f"cuda:{args.device}"
    model_llava = model_llava.to(device)

    model = model_llava.language_model

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_heads = model.config.num_key_value_heads # unique key-value heads

    # load activations
    gt_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest/{args.dataset_name}_gt_layer_wise_*.npy"
    gt_layer_wise_activations = load_chunks(gt_layer_wise_pattern) #(P, 33, 4096)
    hallucinated_layer_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest/{args.dataset_name}_hallucinated_layer_wise_*.npy"
    hallucinated_layer_wise_activations = load_chunks(hallucinated_layer_wise_pattern) #(P, 33, 4096)

    gt_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest/{args.dataset_name}_gt_head_wise_*.npy"
    gt_head_wise_activations = load_chunks(gt_head_wise_pattern) #(P, 32, 4096)
    hallucinated_head_wise_pattern = f"/net/scratch2/steeringwheel/weiyitian/activations/HaloQuest/{args.dataset_name}_hallucinated_head_wise_*.npy"
    hallucinated_head_wise_activations = load_chunks(hallucinated_head_wise_pattern) #(P, 32, 4096)

    gt_head_wise_activations = rearrange(gt_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads) #(P, 33, 32, 128)
    hallucinated_head_wise_activations = rearrange(hallucinated_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads) #(P, 32, 32, 128)
    all_head_wise_activations = np.concatenate([gt_head_wise_activations, hallucinated_head_wise_activations], axis=0)  #(2P, 33, 32, 128)

    print("Successfully loaded all activation chunks")


    # k-fold validation
    for i in range(args.num_fold):
        print(f"Running fold {i}")
        test_size = int(0.2 * dataset_size)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        test_idx = indices[:test_size]
        train_idxs = indices[test_size:]

        # separate into train and val sets
        shuffled_train_idx = np.random.permutation(train_idxs)
        split_idx = int(len(train_idxs) * args.val_ratio)
        train_set_idxs, val_set_idxs = np.split(shuffled_train_idx, [split_idx])

        # get mean steering direction
        #(L * H, 128)
        com_directions = get_com_directions(num_layers, num_heads, train_idxs, gt_head_wise_activations, hallucinated_head_wise_activations)
        # get top k impactful heads
        top_head_idxs = get_top_heads(train_set_idxs, val_set_idxs, gt_head_wise_activations, hallucinated_head_wise_activations, num_layers, num_heads, args.seed, args.top_num_heads, args.use_random_dir)
        print("Heads to be intervened: ", sorted(top_head_idxs))
        print(f'Intervener Strength is {args.alpha}')

        top_heads_by_layer = defaultdict(list)
        for layer, head in top_head_idxs:
            top_heads_by_layer[layer].append(head)


        interveners = []
        pv_configs = []
        for layer, heads in top_heads_by_layer.items():
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

        file_name = f'{args.model_name}_seed_{args.seed}_top_{args.top_num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'
        if args.use_center_of_mass:
            file_name += '_com'
        if args.use_random_dir:
            file_name += '_random'
        
        #model_llava.language_model = intervened_model
        # apply_interventions(dataset, test_idx, model_llava, processor, "output/" + file_name)
        cur_fold_result_df = apply_interventions(dataset, test_idx, intervened_model, processor, "output/" + file_name)
        cur_fold_llama_evaluation = llama_evaluate(cur_fold_result_df,  "output/eval_" + file_name)
        
        eval_ce_kl_owt(model, intervened_model, processor, top_heads_by_layer, "output/eval_" + file_name, file_name, device='cuda', num_samples=100)
        plot_layer_head_PCA(gt_head_wise_activations, hallucinated_head_wise_activations, top_heads_by_layer, args.top_num_heads, "figures/" + file_name)


if __name__ == "__main__":
    main()