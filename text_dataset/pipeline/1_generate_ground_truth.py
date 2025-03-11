from bot.llava import LLaVABot
import json
from tqdm import tqdm
import os
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

bot = LLaVABot()
print("✅ Model and tokenizer loaded successfully!")

save_filepath = "./output/ground_truth"
os.makedirs(save_filepath, exist_ok=True)
save_file1 = os.path.join(save_filepath, "hallsup_output.jsonl")
print(save_file1)
save_file2 = os.path.join(save_filepath, "hallnotsup_output.jsonl")
print(save_file2)

dataset_path1 = "./output/answers/hall_sup.jsonl"
dataset_path2 = "./output/answers/hall_notsup.jsonl"
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
    answer = entry["answer"]
    reason = entry["reason"]

    prompt_support = f"""
    Your task is to determine why a claim is supported by a provided evidence.

    Present your reason in **one sentence**.

    ### **Input:**

    #### Claim:
    {claim}

    #### Evidence:
    {support_sentence}

    #### Label:
    {label}

    ### **Expected Output:**
    Clearly state in one sentence why the evidence supports the claim.
    """

    prompt_not_support = f"""
    Your task is to determine why a claim is **not supported** by a provided evidence.

    Present your reason in **one sentence**.

    ### **Input:**

    #### Claim:
    {claim}

    #### Evidence:
    {evidence}

    #### Label:
    {label}

    ### **Expected Output:**
    Clearly state in one sentence why the evidence does not support the claim.
    """

    response = bot.ask(prompt)[0]
    print(response)

    result_entry = {
            "claim": claim,
            "support_sentence": support_sentence,
            "evidence": evidence,
            "label": label,
            "ground_truth": response,
            "answer": answer,
            "reason": reason
        }
    results.append(result_entry)

    with open(save_file, 'a') as task_file:
        task_file.write(json.dumps(result_entry) + "\n")