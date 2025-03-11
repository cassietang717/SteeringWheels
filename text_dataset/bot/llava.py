from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import json
import re
from tqdm import tqdm
import os

class Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

  
class LLaVABot:
    def __init__(self, model_path="/net/scratch2/steeringwheel/llava-v1.6-vicuna-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.processor = LlavaNextProcessor.from_pretrained(model_path, use_fast=True)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        ).eval()

        self.num_answers = 1
        self.stop_seq = None
        self.deterministic = False
        self.override_temperature = None
        self.usage = Usage()  

    def _ask(
        self,
        prompt: str,
        num_answers=1,
        deterministic=False,
        stop_seq=None,
        override_temperature=None,
    ) -> Tuple[List[str], Usage]:
        """Handles text + image conversation and uses apply_chat_template for chat history"""

        conversation = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]

        text_input = self.processor.apply_chat_template(
            conversation=conversation, add_generation_prompt=True
        )

        inputs = self.processor(text=text_input, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=200,
                do_sample=not deterministic,  
                # temperature=override_temperature if override_temperature is not None else 1,
                temperature=1e-5,
                # top_p=0.9,
                num_return_sequences=num_answers, 
                stopping_criteria=stop_seq,
            )

        response = self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(response))

        return [response], Usage(prompt_tokens, completion_tokens)
    
    def set_num_answers(self, num: int):
        self.num_answers = num

    def set_stop_seq(self, seq):
        self.stop_seq = seq

    def set_deterministic(self, det: bool):
        self.deterministic = det

    def ask(self, prompt: str, history = None, num_answers=None, deterministic=None, stop_seq = None):
        """Prompts a question to the bot and returns the result"""
        res, cost = self._ask(
            prompt,
            num_answers if num_answers is not None else self.num_answers,
            deterministic if deterministic is not None else self.deterministic,
            stop_seq if stop_seq is not None else self.stop_seq,
            self.override_temperature,
        )

        self.usage.prompt_tokens += cost.prompt_tokens
        self.usage.completion_tokens += cost.completion_tokens
        return res