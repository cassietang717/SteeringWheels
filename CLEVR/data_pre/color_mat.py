import torch
from PIL import Image
import requests
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json
from PIL import Image
import os
from fastchat.conversation import get_conv_template
from tqdm import tqdm


# Load processor and model
print("Loading processor and model...")
device = "cuda"
processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)


# Load and preprocess image
json_path = "/net/scratch/llama/clevr_v1.0/CLEVR_v1.0/questions/train_0_2000.json"
image_dir = "/net/scratch/llama/clevr_v1.0/CLEVR_v1.0/images/train"
save_path = "data/color_mat_lol.json"

# Load data
with open(json_path, "r") as f:
    data = json.load(f)
mismatches = []

# Loop through questions
for i, q in tqdm(enumerate(data["questions"]), total=len(data["questions"]), desc="Processing questions"):
    image_id = q["image_index"]
    image_filename = q["image_filename"]
    question = q["question"]
    gt_answer = q["answer"]
    function_type = q["program"][-1]["function"]

    if function_type not in {"query_color", "query_material", "equal_color", "equal_material"}:
        continue

    # Load image ded
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
        instruction = "Answer the following question with only 'gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', or 'yellow', and respond in lowercase: "
    elif function_type == "query_material":
        instruction = "Answer the following question with only 'rubber', or 'metal', and respond in lowercase: "


    conv = get_conv_template("llava-chatml")
    conv.append_message(conv.roles[0], "<image>\n" + instruction + question)
    conv.append_message(conv.roles[1], "") # Placeholder for assistant's reply
    prompt = conv.get_prompt()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=128)
    decoded_output = processor.decode(output[0], skip_special_tokens=True)

    if ">assistant" in decoded_output:
        decoded_output = decoded_output.split(">assistant", 1)[1].strip()

    model_answer = decoded_output.lower().strip()
    print(f"[{i}] Model Answer: {model_answer}")
    print(f"[{i}] Ground Truth Answer: {gt_answer}")
    print('-' * 60)


    if model_answer != gt_answer.lower().strip():
        mismatches.append({
            "image": image_filename,
            "question": question,
            "gt_answer": gt_answer,
            "model_answer": model_answer,
            "last_function":function_type
        })
        print('saved observation')


with open(save_path, "w") as f:
    json.dump(mismatches, f, indent=4)

print(f"Saved {len(mismatches)} mismatched entries to {save_path}")