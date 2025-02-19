import json
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import re

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
Your response **must** be structured in the following **JSON format** only. Do not include explanations, preamble, or any extra text:

```json
{{
  "explanation": "Your explanation here",
  "hallucination": "yes" | "no",
  "rating": rating_value
}}
'''

tokenizer = AutoTokenizer.from_pretrained("/net/scratch2/steeringwheel/Llama-3.1-8B-Instruct")

model = LlamaForCausalLM.from_pretrained("/net/scratch2/steeringwheel/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto")
model.to("cuda:0")
model = torch.compile(model)


with open("output/MMHal_output.json", "r") as json_file:
   data = json.load(json_file)


start_timing = torch.cuda.Event(enable_timing=True)
end_timing = torch.cuda.Event(enable_timing=True)

results = []
start_timing.record()


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
   outputs = model.generate(tokenized_chat.to("cuda:0"), max_new_tokens=256)
   
   input_len = tokenized_chat.shape[-1]
   decoded_model_output = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

   match = re.search(r"({.*?})", decoded_model_output, re.DOTALL)
   model_output = match.group(1)

   parsed_output = json.loads(model_output)

   explanation = parsed_output["explanation"]
   hallucination = parsed_output["hallucination"]
   rating = parsed_output["rating"]

   result_entry = {
        "question_type": question_type,
        "image_content": image_content,
        "image_src": image_url,
        "question": question,
        "gt_answer": gt_answer,
        "llava_model_answer": model_answer,
        "llama_hallucination_analysis": explanation,
        "llama_hallucination_evaluation": hallucination,
        "llama_hallucination_rating": rating
    }
   results.append(result_entry)

   print(f"Processed question: {question}")
   print(f"Ground truth: {gt_answer}")
   print(f"Llava Model answer: {model_answer}")
   print(model_output)
   print("=" * 50)

torch.cuda.synchronize()
end_timing.record()
print(f"Runtime: {.001 * start_timing.elapsed_time(end_timing):.4f} seconds")

with open("output/MMHal_llama.json", "w") as outfile:
    json.dump(results, outfile, indent=4)