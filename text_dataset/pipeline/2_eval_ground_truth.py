from ..bot.llamma import LLaMABot
import json
import os
from tqdm import tqdm
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument(
    "--dataset_path",
    type=str,
    help="Select the dataset path",
    default="dataset_path1",
)
ap.add_argument(
    "--prompt",
    type=str,
    help="Select the prompt",
    default="prompt_support",
)
ap.add_argument(
    "--save_file",
    type=str,
    help="Select the save file path",
    default="save_file1",
)
args = ap.parse_args()

dataset_path = args.dataset_path
save_file = args.save_file
prompt = args.prompt

bot = LLaMABot()
print("✅ Model and tokenizer loaded successfully!")

save_filepath = "./text_dataset/output/eval"
os.makedirs(save_filepath, exist_ok=True)
if save_file == "save_file1":
    save_file = os.path.join(save_filepath, "hallsup_eval.jsonl")
else:
    save_file = os.path.join(save_filepath, "hallnotsup_eval.jsonl")

if dataset_path == "dataset_path1":
    dataset_path = "./text_dataset/output/ground_truth/hallsup_output.jsonl"
else:
    dataset_path = "./text_dataset/output/ground_truth/hallnotsup_output.jsonl"
with open(dataset_path, 'r') as file:
    dataset = [json.loads(line) for line in file]

results = []
for entry in tqdm(dataset, desc="Processing entries"):
    claim = entry["claim"]
    support_sentence = entry["support_sentence"]
    evidence = entry["evidence"]
    label = entry["label"]
    ground_truth =  entry["ground_truth"]
    answer = entry["answer"]
    reason = entry["reason"]

    if prompt == "prompt_support":
        prompt = f"""
    Your task is to determine if the ground truth can indicate that the evidence **supports** the claim.
    
    Answer strictly with either "Yes" or "No" without additional explanations.

    ### **Input:**

    #### Claim:
    {claim}

    #### Evidence:
    {support_sentence}

    #### Ground Truth:
    {ground_truth}

    #### Label:
    {label}

    ### **Expected Output:**
    Just respond with "Yes" or "No".
    """
    else:
        prompt= f"""
    Your task is to determine if the ground truth correctly indicates that the evidence does **not** support the claim.

    Respond **only** with "Yes" or "No". Do **not** provide any explanations.

    ### **Input:**

    #### Claim:
    {claim}

    #### Evidence:
    {evidence}

    #### Ground Truth:
    {ground_truth}

    #### Label:
    {label}

    ### **Expected Output:**
    Just respond with "Yes" or "No".
    """

    response = bot.ask(prompt)[0]
    print(response)

    result_entry = {
            "claim": claim,
            "support_sentence": support_sentence,
            "evidence": evidence,
            "label": label,
            "ground_truth": ground_truth,
            "answer": answer,
            "reason": reason,
            "tag": response
        }
    results.append(result_entry)

    with open(save_file, 'a') as task_file:
        task_file.write(json.dumps(result_entry) + "\n")
    
print(save_file)