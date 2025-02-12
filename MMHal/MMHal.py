from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import json
from io import BytesIO
from tqdm import tqdm

processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 

model.to("cuda:0")
model = torch.compile(model)

with open("image_data/MMHal.json", "r") as json_file:
    data = json.load(json_file)

start_timing = torch.cuda.Event(enable_timing=True)
end_timing = torch.cuda.Event(enable_timing=True)

results = []
start_timing.record()

for entry in tqdm(data, desc="Processing entries"):
    question = entry["question"] + " Answer with reasonable length (not too short or too long)."
    image_url = entry["image_src"]
    gt_answer = entry["gt_answer"]

    image_content = entry["image_content"]
    question_type = entry["question_type"]

    try:
        response = requests.get(image_url)
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
    except Exception as e:
        print(f"Error processing image URL {image_url}: {e}")
        continue

    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, use_cache=True, max_new_tokens=100)

    model_output = processor.decode(output[0], skip_special_tokens=True)
    model_answer = model_output.split("ASSISTANT:")[-1].strip()

    result_entry = {
        "question_type": question_type,
        "image_content": image_content,
        "image_src": image_url,
        "question": question,
        "gt_answer": gt_answer,
        "model_answer": model_answer
    }
    results.append(result_entry)

    print(f"Processed question: {question}")
    print(f"Ground truth: {gt_answer}")
    print(f"Model answer: {model_answer}")
    print("=" * 50)

    if entry == 10:
        break

torch.cuda.synchronize()
end_timing.record()
print(f"Runtime: {.001 * start_timing.elapsed_time(end_timing):.4f} seconds")

with open("output/MMHal_output.json", "w") as outfile:
    json.dump(results, outfile, indent=4)
