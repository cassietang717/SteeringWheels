from bot.llamma import LLaMABot
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

save_filepath = "./output/eval"
os.makedirs(save_filepath, exist_ok=True)
save_file1 = os.path.join(save_filepath, "hallsup_eval.jsonl")
print(save_file1)
save_file2 = os.path.join(save_filepath, "hallnotsup_eval.jsonl")
print(save_file2)

dataset_path1 = "./output/ground_truth/hallsup_output.jsonl"
dataset_path2 = "./output/ground_truth/hallnotsup_output.jsonl"
with open(dataset_path1, 'r') as file:
    dataset1 = [json.loads(line) for line in file]

with open(dataset_path2, 'r') as file:
    dataset2 = [json.loads(line) for line in file]

results = []
for entry in tqdm(dataset_path, desc="Processing entries"):
    claim = entry["claim"]
    support_sentence = entry["support_sentence"]
    evidence = entry["evidence"]
    label = entry["label"]
    ground_truth =  entry["ground_truth"]
    answer = entry["answer"]
    reason = entry["reason"]

    prompt_support = f"""
    Your task is to evaluate whether the ground truth aligns well with the evidence in supporting the claim.
    
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

    prompt_not_support = f"""
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
    