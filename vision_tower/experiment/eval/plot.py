import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import LlavaNextProcessor, CLIPProcessor
import torchvision.transforms as T
from torchvision import transforms

image_path = "/home/cassietang/steeringwheel/vision_tower/test_data/images/1424845359_c4945d38f1_o.jpg"

# Load processor (no need to .to("cpu") for processor)
# processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
# processor =  CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor =  CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
image = Image.open(image_path).convert("RGB")
# image = image_raw.resize((336, 336), Image.LANCZOS)
# image = image_raw

# Process the image to get the 5 views
# processor.image_processor.crop_size={"height": 336, "width": 336}
# processor.image_processor.size= {"shortest_edge": 336}
answer = processor.image_processor(image, return_tensors="pt")
views = answer['pixel_values'][0]  # shape: (5, 3, 336, 336)
print(views.shape)

# fig, axs = plt.subplots(1, 5, figsize=(20, 4))
# for i in range(5):
#     img = T.ToPILImage()(views[i])
#     axs[i].imshow(img)
#     axs[i].axis("off")
#     axs[i].set_title(f"View {i+1}")
img = T.ToPILImage()(views)

# Plot and save
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis("off")
plt.title("CLIP View")
plt.tight_layout()


# Save to PDF
save_path = "/home/cassietang/steeringwheel/vision_tower/experiment/eval/clip_views_336.pdf"
plt.savefig(save_path, format="pdf")
print(f"Saved PDF to: {save_path}")