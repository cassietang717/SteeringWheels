import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from transformers import CLIPProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from transformers import set_seed
import requests
from io import BytesIO

from vti_utils.utils import get_demos_coco, get_demos_Halo, obtain_textual_vti, obtain_visual_vti
from vti_utils.llm_layers import add_vti_layers, remove_vti_layers


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def hall_ans(processor, model):
#     results = []
#     HaloQuest_df = pd.read_csv("/home/cassietang/steeringwheel/HaloQuest/output/sample_HaloQuest_output.csv")
#     # filtered_HaloQuest_df = HaloQuest_df[HaloQuest_df["hallucination_type"] == "visual challenge"].sample(n=300, random_state=42).reset_index(drop=True)
#     filtered_HaloQuest_df = HaloQuest_df[HaloQuest_df["hallucination_type"] != "visual challenge"].sample(n=100, random_state=42).reset_index(drop=True)

#     for _, entry in tqdm(filtered_HaloQuest_df.iterrows(), total=filtered_HaloQuest_df.shape[0], desc="Processing entries"):
#         question = entry["question"]
#         image_url = entry["image_url"]
#         gt_answer = entry["gt_answer"]
#         model_answer_before_vti = entry["model_answer"]
#         hallucination_type = entry["hallucination_type"]

#         try:
#             response = requests.get(image_url)
#             image_bytes = BytesIO(response.content)
#             image = Image.open(image_bytes)
#             image.load()
#         except Exception as e:
#             print(f"Error processing image URL {image_url}: {e}")
#             continue

#         conversation = [{
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": question},
#             ],
#         }]

#         prompt = processor.apply_chat_template(conversation=conversation, add_generation_prompt=True)
#         inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
#         output = model.generate(**inputs, use_cache=True, max_new_tokens=100)

#         model_output = processor.decode(output[0], skip_special_tokens=True)
#         model_answer = model_output.split("ASSISTANT:")[-1].strip()

#         result_entry = {
#             "image_url": image_url,
#             "question": question,
#             "gt_answer": gt_answer,
#             "model_answer_before_vti": model_answer_before_vti,
#             "model_answer_after_vti": model_answer,
#             "hallucination_type": hallucination_type
#         }
#         results.append(result_entry)
    
#     return json.dumps(results, indent=4) 

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def hall_ans(processor, model):
    results = []
    with open("/home/cassietang/steeringwheel/MMHal/output/MMHal_output.json", "r") as json_file:
    # with open("/home/cassietang/steeringwheel/VTI/results/coco_pope_popular_answer.jsonl", "r") as json_file:
        data = json.load(json_file)

    # data = load_jsonl("/home/cassietang/steeringwheel/VTI/results/coco_pope_popular_answer.jsonl")

    for entry in tqdm(data, desc="Processing entries"):
        question = entry["question"]
        image_url = entry["image_src"]
        gt_answer = entry["gt_answer"]
        question_type = entry["question_type"]
        model_answer_vti = entry["model_answer"]

        try:
            response = requests.get(image_url)
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)
            image.load()
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
            "image_url": image_url,
            "question": question,
            "gt_answer": gt_answer,
            "model_answer_vti": model_answer_vti,
            "model_answer_my_vti": model_answer
        }
        results.append(result_entry)
    
    return json.dumps(results, indent=4) 


def eval_model(args, model, image_processor, processor):
    # Step 1: Prepare input images and input ids
    input_images, input_ids = get_demos_coco(args, image_processor, processor)
    # input_images, input_ids = get_demos_Halo(args, image_processor, processor)

    print('Obtaining direction\n')

    if args.alpha_image != 0:
        vti_vision, _ = obtain_visual_vti(
            model, input_images, rank=1
            )

        visual_direction = vti_vision[1:]

    if args.alpha_image != 0:
        add_vti_layers(model.vision_tower.vision_model, torch.stack([visual_direction],dim=1).cuda(), alpha = [args.alpha_image])
    
    if args.alpha_text != 0:

        vti_text, _ = obtain_textual_vti(
            model, input_ids, input_images, rank=1
            )
        textual_direction = vti_text[1:]

    if args.alpha_text != 0:
        add_vti_layers(model, torch.stack([textual_direction],dim=1).cuda(), alpha = [args.alpha_text])

    torch.cuda.empty_cache()

    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    ans_file.write(hall_ans(processor, model))
    ans_file.close()
    print(f"Results saved to {answers_file}")
    remove_vti_layers(model)
    remove_vti_layers(model.vision_tower.vision_model)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="/data/datasets/MSCOCO/val2014")
    parser.add_argument("--answers-file", type=str, default="/home/cassietang/steeringwheel/vision_tower/experiment/results/hallucination_ans.jsonl")
    parser.add_argument("--data-file", type=str, default="/net/scratch2/steeringwheel/coco")
    # parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num_demos", type=int, default=70)
    parser.add_argument("--alpha_image", type=float, default=0.9)
    parser.add_argument("--alpha_text", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--sample", action='store_true')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--num_trials", type=int, default=50)
    
    args = parser.parse_args()
    set_seed(args.seed)
    args.answers_file = f'/home/cassietang/steeringwheel/vision_tower/experiment/results/MMHal_compare_hallucination_ans_{args.alpha_text}_{args.alpha_image}_{args.mask_ratio}.jsonl'
    processor = LlavaNextProcessor.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", use_fast=True)
    # image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model_llava = LlavaNextForConditionalGeneration.from_pretrained("/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model_llava 
    model = model.to(device)
    # print("model.vision_tower.vision_model:")
    # print(model.vision_tower.vision_model)
    # print("model.vision_tower:")
    # print(model.vision_tower)
    eval_model(args, model, processor, processor)