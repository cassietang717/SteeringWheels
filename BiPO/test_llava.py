import torch
from PIL import Image
import requests
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from fastchat.conversation import get_conv_template

# Load processor and model
model_path = "/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf"
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf",revision="30f8c4f")
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
).to("cuda")

# Load and preprocess image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

prompt = "USER: <image>\nWhat is shown in this image? ASSISTANT:"

conv = get_conv_template("llava-chatml")
conv.append_message(conv.roles[0], "<image>\n" + "What is shown in this image?")
conv.append_message(conv.roles[1], "")  # Placeholder for assistant's reply
prompt = conv.get_prompt()

input = processor(prompt, image_cats, padding=True, return_tensors="pt").to("cuda")
output = model.generate(**input, max_new_tokens=128) [:,len(input["input_ids"]):]
decoded_output = processor.decode(output[0], skip_special_tokens=True)
print(decoded_output)