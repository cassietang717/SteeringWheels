from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import pandas as pd
from io import BytesIO
from tqdm import tqdm

processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 

model.to("cuda:0")
model = torch.compile(model)

start_timing = torch.cuda.Event(enable_timing=True)
end_timing = torch.cuda.Event(enable_timing=True)

results = []
start_timing.record()

HaloQuest_df = pd.read_csv("data/haloquest.csv")
filtered_HaloQuest_df = HaloQuest_df[HaloQuest_df["hallucination type"] != "visual challenge"]

filtered_HaloQuest_df = filtered_HaloQuest_df.sample(frac=.1)

for _, entry in tqdm(filtered_HaloQuest_df.iterrows(), total=filtered_HaloQuest_df.shape[0], desc="Processing entries"):
    question = entry["question"] + " Answer with reasonable length (not too short or too long)."
    image_url = entry["url"]
    gt_answer = entry["groundtruth responses"]
    hallucination_type = entry["hallucination type"]

    try:
        response = requests.get(image_url)
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        image.load()
    except Exception as e:
        print(f"Error processing image URL {image_url}: {e}")
        continue

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]

    prompt = processor.apply_chat_template(conversation=conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, use_cache=True, max_new_tokens=100)

    model_output = processor.decode(output[0], skip_special_tokens=True)
    model_answer = model_output.split("ASSISTANT:")[-1].strip()

    result_entry = {
        "image_url": image_url,
        "question": question,
        "gt_answer": gt_answer,
        "model_answer": model_answer,
        "hallucination_type": hallucination_type
    }
    results.append(result_entry)

    print(f"Image url: {image_url}")
    print(f"Processed question: {question}")
    print(f"Ground truth: {gt_answer}")
    print(f"Model answer: {model_answer}")
    print(f"Hallucination type: {hallucination_type}")
    print("=" * 50)

torch.cuda.synchronize()
end_timing.record()
print(f"Runtime: {.001 * start_timing.elapsed_time(end_timing):.4f} seconds")

results_df = pd.DataFrame(results)
results_df.to_csv("output/HaloQuest_output.csv", index=False)