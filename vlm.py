from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import time

processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf",use_fast=True)

model = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

# # prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open('false.jpg')

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What color is the flag that is sitting on top of the building on the bottom left corner of the picture? Please do not just describe."},
          {"type": "image"},
        ],
    },
]
start_time = time.time()

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# Start timing before generating response
processing_time = time.time()

output = model.generate(**inputs, max_new_tokens=100)

# End timing
end_time = time.time()

# Print outputs
print(processor.decode(output[0], skip_special_tokens=True))

# Calculate and print times
preprocessing_time = processing_time - start_time
generation_time = end_time - processing_time
total_time = end_time - start_time

print(f"Preprocessing time: {preprocessing_time:.4f} seconds")
print(f"Generation time: {generation_time:.4f} seconds")
print(f"Total time: {total_time:.4f} seconds")