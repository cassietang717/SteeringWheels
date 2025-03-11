from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
from transformers import LlamaForCausalLM
import json
import re
from tqdm import tqdm
import os

class Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

  
class LLaMABot:
    def __init__(self, model_path="/net/scratch2/steeringwheel/Llama-3.1-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = LlamaForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )

        self.usage = Usage()  
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _ask(
        self,
        prompt: str,
    ) -> Tuple[List[str], Usage]:
        """Handles text + image conversation and uses apply_chat_template for chat history"""

        conversation = [
        {
            "role": "user",
            "content": prompt,
        }]

        text_input = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = text_input.to(self.device)

        attention_mask = text_input.ne(self.tokenizer.pad_token_id).to(self.device)

        output = self.model.generate(
            inputs,
            attention_mask=attention_mask, 
            max_new_tokens=200,
        )
        
        input_len = text_input.shape[-1]
        response = self.tokenizer.decode(output[0,input_len:], skip_special_tokens=True)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(response))

        return [response], Usage(prompt_tokens, completion_tokens)

    def ask(self, prompt: str, history = None, num_answers=None, deterministic=None, stop_seq = None):
        """Prompts a question to the bot and returns the result"""
        res, cost = self._ask(
            prompt,
            self.system_prompt,
            self.system_hist,
            history if history is not None else self.history,
            num_answers if num_answers is not None else self.num_answers,
            deterministic if deterministic is not None else self.deterministic,
            stop_seq if stop_seq is not None else self.stop_seq,
            self.override_temperature,
        )

        self.usage.prompt_tokens += cost.prompt_tokens
        self.usage.completion_tokens += cost.completion_tokens
        return res
