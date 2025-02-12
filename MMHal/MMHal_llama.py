import json
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import re
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("/net/scratch2/steeringwheel/llama-3.1-8B")

# model = LlamaForCausalLM.from_pretrained("/net/scratch2/steeringwheel/llama-3.1-8B", torch_dtype=torch.float16, device_map="auto")
# #model = AutoModelForCausalLM.from_pretrained("/net/scratch2/steeringwheel/llama-3.1-8B")
# model.to("cuda:0")
# model = torch.compile(model)

template = '''Please act as an **impartial and objective judge** and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the given user question. Your evaluation should be based on the following criteria:

1. **Informativeness** – Does the response provide relevant and useful information?
2. **Hallucination** – Does the response introduce any incorrect or unsupported information?

### **Definition of Hallucination**
A hallucination occurs when the LMM **includes details that are not present or implied** in the image, standard human-generated answer, or prior context. This includes:
- Fabricated facts, objects, actions, or descriptions.
- Incorrect interpretations that contradict the given information.
- Over-extrapolations that assume information beyond what is available.

---
### **Input Details**
#### **Image Contents:**
{}

#### **User Question:**
{}

#### **Standard Human-Generated Answer:**
{}

#### **LMM Response to Evaluate:**
{}

---
### **Evaluation Criteria**
1. **Does the response contain hallucination?** (Yes or No)
   - If **Yes**, identify the incorrect information and explain why it is not supported by the provided details.
   - If **No**, confirm that the response aligns with the given information.
   
2. **Provide a confidence score** on a scale of **0 to 1**, where:
   - **0.0** = No confidence (high uncertainty)
   - **0.5** = Somewhat confident
   - **1.0** = Fully confident (high certainty in the evaluation)

3. **Rate the response using the following scale:**
   - **6** → Very informative, no hallucination
   - **5** → Informative, no hallucination
   - **4** → Somewhat informative, no hallucination
   - **3** → Not informative, no hallucination
   - **2** → Very informative, with hallucination
   - **1** → Somewhat informative, with hallucination
   - **0** → Not informative, with hallucination

---
### **Expected Output Format (JSON)**
Do not repeat the prompt, only provide the following JSON output filled in with your final evaluation.

Your response **must** be structured in the following **JSON format**:

```json
{{
  "explanation": "Your explanation here",
  "hallucination": "yes" | "no",
  "rating": rating_value
}}
'''

with open("output/MMHal_output.json", "r") as json_file:
    data = json.load(json_file)

for entry in tqdm(data, desc="Processing entries"):
    question = entry["question"]
    image_content = entry["image_content"]
    question_type = entry["question_type"]

    model_answer = entry["model_answer"]
    gt_answer = entry["gt_answer"]

    image_url = entry["image_src"]

    filled_prompt = template.format(image_content, question, gt_answer, model_answer)

    conversation = [
    {
        "role": "user",
        "content": filled_prompt,
    }]

    tokenized_chat = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    print(tokenizer.decode(tokenized_chat))
    #inputs = tokenizer(filled_prompt, return_tensors="pt").to("cuda:0")
    #inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
    #inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda:0")

   #  output = model.generate(**inputs, use_cache=True)#, max_new_tokens=100)

   #  model_output = tokenizer.decode(output[0], skip_special_tokens=True)

    #print(model_output)
    print("=" * 50)