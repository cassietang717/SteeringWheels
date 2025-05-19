import torch
from PIL import Image
import json
import os
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from fastchat.conversation import get_conv_template
from random import shuffle
from tqdm import tqdm

device = "cuda"

# Load processor and model
processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# Paths
json_path_0_2000 = "/net/scratch/llama/clevr_v1.0/CLEVR_v1.0/questions/train_0_2000.json"
json_path_2000_4000 = "/net/scratch/llama/clevr_v1.0/CLEVR_v1.0/questions/train_2000_4000.json"
image_dir = "/net/scratch/llama/clevr_v1.0/CLEVR_v1.0/images/train"
save_path = "data/color_mat_train_7c.json"

# Load data
with open(json_path_0_2000, "r") as f1:
    data1 = json.load(f1)
with open(json_path_2000_4000, "r") as f2:
    data2 = json.load(f2)
data = {"questions": data1["questions"] + data2["questions"],}


mismatches = []
correct_matches = []

for i, q in tqdm(enumerate(data["questions"]), total=len(data["questions"]), desc="Processing questions"):
    image_id = q["image_index"]
    image_filename = q["image_filename"]
    question = q["question"]
    gt_answer = q["answer"]
    function_type = q["program"][-1]["function"]

    #if function_type not in {"query_color", "equal_color", "equal_material"}:
    if function_type not in {"query_color"}:
        continue
    if gt_answer in {"cyan"}:
        continue

    image_path = os.path.join(image_dir, image_filename)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        continue

    print(f"[{i}] Image loaded: {image_filename}")
    print(f"[{i}] Question: {question}")

    if function_type in {"equal_color", "equal_material"}:
        instruction = "Answer the following question for the image with only 'yes' or 'no' in lowercase: "
    elif function_type == "query_color":
        #instruction = "Answer the following question with only 'red', 'blue', 'green', 'brown', 'purple', or 'yellow', and respond in lowercase: "
        instruction = "Answer the following question by looking only at the image. Your answer must be one of: gray, red, blue, green, brown, purple, or yellow — in lowercase. Do not choose based on the order of the words in this list. Choose only the color that best matches the image."

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction + question},
        ],
    }]

    prompt = processor.apply_chat_template(conversation=conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, use_cache=True, max_new_tokens=100)

    model_output = processor.decode(output[0], skip_special_tokens=True)
    model_answer = model_output.split("ASSISTANT:")[-1].strip().lower()

    print(f"[{i}] Question: {question}")
    print(f"[{i}] Model Answer: {model_answer}")
    print(f"[{i}] Ground Truth Answer: {gt_answer}")
    print('-' * 60)

    entry = {
        "image": image_filename,
        "question": question,
        "gt_answer": gt_answer,
        "model_answer": model_answer,
        "last_function": function_type
    }

    if model_answer != gt_answer:
        mismatches.append(entry)
    else:
        correct_matches.append(entry)

# # Balancing mismatches
# yes_mismatches = [x for x in mismatches if x["gt_answer"] == "yes"]
# no_mismatches = [x for x in mismatches if x["gt_answer"] == "no"]
# diff = abs(len(yes_mismatches) - len(no_mismatches))

# if len(yes_mismatches) > len(no_mismatches):
#     target_gt = "no"
#     flipped_to = "yes"
# else:
#     target_gt = "yes"
#     flipped_to = "no"

# eligible_supplement = [x for x in correct_matches if x["gt_answer"] == target_gt]
# shuffle(eligible_supplement)
# supplemented = []

# for entry in eligible_supplement[:diff]:
#     fake_mismatch = entry.copy()
#     fake_mismatch["model_answer"] = flipped_to
#     fake_mismatch["note"] = "flipped_from_correct"
#     supplemented.append(fake_mismatch)

# mismatches.extend(supplemented)

# # Report final balance
# yes_final = sum(1 for x in mismatches if x["gt_answer"] == "yes")
# no_final = sum(1 for x in mismatches if x["gt_answer"] == "no")

# print(f"Final mismatch counts - yes: {yes_final}, no: {no_final}")

# Save
with open(save_path, "w") as f:
    json.dump(mismatches, f, indent=4)

print(f"Saved {len(mismatches)} mismatched entries to {save_path}")