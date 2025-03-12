import concurrent.futures
import json
import pathlib
from argparse import ArgumentParser
from dataclasses import asdict
import random
import numpy as np
import os
from tqdm import tqdm
import re

from ..bot.llava import LLaVABot  # ✅ Import LLaVABot directly


ap = ArgumentParser()
ap.add_argument(
    "--glm-model",
    type=str,
    help="Select the generating model",
    default="/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf",
)
ap.add_argument(
    "--dataset_path",
    type=str,
    help="Prompt to generate answers for",
    default="/net/scratch2/steeringwheel/wice/data/entailment_retrieval/subclaim/train.jsonl",
)
ap.add_argument(
    "--test_description_dir",
    default="./text_dataset/output/answers",
    help="Place to store descriptions",
)
ap.add_argument(
    "--force",
    action="store_true",
    help="Overwrite existing descriptions",
)
args = ap.parse_args()

model_path = args.glm_model
description_dir = pathlib.Path(args.test_description_dir)

bot = LLaVABot(model_path=model_path)

dataset_path = args.dataset_path
with open(dataset_path, 'r') as file:
    datasets = [json.loads(line) for line in file]

save_filepath = args.test_description_dir
os.makedirs(save_filepath, exist_ok=True)
save_file = os.path.join(save_filepath, "hall_answers.jsonl")
print(save_file)


def generate_topic(dataset):
    results = []
    for entry in tqdm(dataset, desc="Processing entries"):
        claim = entry["claim"]
        if entry["supporting_sentences"] == [[]]:  
            supporting_sentence = None
        else:
            supporting_sentences = [[entry["evidence"][i] for i in indices] for indices in entry["supporting_sentences"]]
            supporting_sentence = " ".join([" ".join(sent) for sent in supporting_sentences])
        evidence = " ".join(entry["evidence"])
        label = entry["label"]
        prompt_if_support = f'''
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

        
        bot.set_deterministic(False)
        bot.set_num_answers(1)

        response = bot.ask(prompt_if_support)
        match = re.search(r"({.*?})", response[0], re.DOTALL)
        if match:
            try:
                model_output = match.group(1)
                response_json = json.loads(model_output)
                reason = response_json.get("reason", "N/A")
                answer = response_json.get("answer", "N/A")
            except json.JSONDecodeError:
                print(f"⚠️ Warning: JSON decoding failed for claim '{claim}'. Skipping...")
                continue
        else:
            print(f"⚠️ Warning: No valid JSON found in response for claim '{claim}'. Skipping...")
            continue

        result_entry = {
            "claim": claim,
            "support_sentence": supporting_sentence,
            "evidence": evidence,
            "label": label,
            "answer": answer,
            "reason": reason
        }

        results.append(result_entry)
        with open(save_file, 'a') as task_file:
            task_file.write(json.dumps(result_entry) + "\n")

    return results

results = generate_topic(datasets)


    

    

    