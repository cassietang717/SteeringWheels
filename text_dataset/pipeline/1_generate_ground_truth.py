# import text_dataset.bot.llava
from ..bot.llava import LLaVABot
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

save_filepath = "./text_dataset/output/ground_truth"
os.makedirs(save_filepath, exist_ok=True)
if save_file == "save_file1":
    save_file = os.path.join(save_filepath, "hallsup_output.jsonl")
else:
    save_file = os.path.join(save_filepath, "hallnotsup_output2.jsonl")

if dataset_path == "dataset_path1":
    dataset_path = "./text_dataset/output/answers/hall_sup.jsonl"
else:
    dataset_path = "./text_dataset/output/answers/hall_notsup.jsonl"

with open(dataset_path, 'r') as file:
    dataset = [json.loads(line) for line in file]

print(dataset_path)

results = []
for entry in tqdm(dataset, desc="Processing entries"):
    claim = entry["claim"]
    support_sentence = entry["support_sentence"]
    evidence = entry["evidence"]
    label = entry["label"]
    answer = entry["answer"]
    reason = entry["reason"]

    if prompt == "prompt_support":
        prompt = f"""
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
    else:
        prompt = f"""
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

print(save_file)