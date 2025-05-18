import torch
from PIL import Image
import json
import os
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from fastchat.conversation import get_conv_template
device = "cuda"
# Load processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf",revision="30f8c4f")
model = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

# Load JSON and image directory
json_path = "/net/scratch/llama/clevr_v1.0/CLEVR_v1.0/questions/val_500.json"
image_dir = "/net/scratch/llama/clevr_v1.0/CLEVR_v1.0/images/val"
save_path = "/net/scratch2/steeringwheel/clevr/eval_shape_size.json"

with open(json_path, "r") as f:
    data = json.load(f)

mismatches = []

# Loop through questions
for i, q in enumerate(data["questions"]):
    image_filename = q["image_filename"]
    question_text = q["question"]
    program = q["program"]
    gt_answer = q["answer"]

    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"[{i}] Image not found: {image_path}")
        continue

    image = Image.open(image_path).convert("RGB")
    function_list = [step["function"] for step in program]

    if not function_list or function_list[-1] not in {"query_shape", "equal_size", "equal_shape"}:
        continue
    print(f"[{i}] Image loaded: {image_filename}")
    print(f"[{i}] Question: {question_text}")

    last_function = function_list[-1]
    if last_function in {"equal_size", "equal_shape"}:
        instruction = "Answer the following question for the image with only 'yes' or 'no' in lowercase: "
    # elif last_function == "query_size":
    #     instruction = "Answer the following question with only 'small' or 'large', and respond in lowercase: "
    elif last_function == "query_shape":
        instruction = "Answer the following question with only 'cube', 'sphere', or 'cylinder', and respond in lowercase: "
    else:
        instruction = ""

    conv = get_conv_template("llava-chatml")
    conv.append_message(conv.roles[0], "<image>\n" + instruction+ question_text)
    conv.append_message(conv.roles[1], "")  # Placeholder for assistant's reply
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
            "question": question_text,
            "gt_answer": gt_answer,
            "model_answer": model_answer,
            "last_function": last_function
        })
        print('saved observation')

with open(save_path, "w") as f:
    json.dump(mismatches, f, indent=2)

print(f"Saved {len(mismatches)} mismatched entries to {save_path}")
