from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf",use_fast=True)

model = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")
model = torch.compile(model)

# # prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open('image_data/hlcn.jpg')

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

start_timing = torch.cuda.Event(enable_timing=True)
processing_timing = torch.cuda.Event(enable_timing=True)
end_timing = torch.cuda.Event(enable_timing=True)

start_timing.record()

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# Start timing before generating response
processing_timing.record()

output = model.generate(**inputs, max_new_tokens=100, use_cache=True)

# End timing
end_timing.record()

# Print outputs
print(processor.decode(output[0], skip_special_tokens=True))

# Calculate and print times
torch.cuda.synchronize()
preprocessing_time = start_timing.elapsed_time(processing_timing)
generation_time = processing_timing.elapsed_time(end_timing)
total_time = start_timing.elapsed_time(end_timing)

print(f"Preprocessing time: {.001 * preprocessing_time:.4f} seconds")
print(f"Generation time: {.001 * generation_time:.4f} seconds")
print(f"Total time: {.001 * total_time:.4f} seconds")