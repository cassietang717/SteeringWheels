import json
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import re
import pandas as pd

template = '''You are an evaluator tasked with determining whether LLava’s answer is hallucinating. You do not have access to the image, only the following textual information. Your goal is to check if the **direct answer** to the question contradicts the ground truth. Additional details, even if extraneous, should be ignored as long as the direct answer is correct.

### Instructions:
1. **Direct Answer Focus:** Identify the specific part of LLava’s response that directly answers the question.
2. **Ground Truth Comparison:** Compare the direct answer with the provided ground truth.
3. **Contradiction Check:** 
   - If the direct answer contradicts the ground truth, mark it as a hallucination.
   - If the direct answer is in agreement with the ground truth—even if extra details are present—mark it as not hallucinating.
4. **Extra Details:** Do not consider additional information or context that does not alter the core answer.

---
### **Input Details**
#### **Possible hallucination type:**
{}

#### **User Question:**
{}

#### **Standard Human-Generated Answer:**
{}

#### **Llava Response to Evaluate:**
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
   - **0** → Very informative, no hallucination
   - **1** → Informative, no hallucination
   - **2** → Somewhat informative, no hallucination
   - **3** → Not informative, no hallucination
   - **4** → Very informative, with hallucination
   - **5** → Somewhat informative, with hallucination
   - **6** → Not informative, with hallucination

---
### **Expected Output Format (JSON)**
**IMPORTANT:** Your entire response must be a single, valid JSON object with **no additional text or formatting whatsoever**. Do not include markdown, code fences, or any other commentary. 
All double quotes inside string values must be properly escaped with a backslash (\\) (for example, use `\\\"` for quotes inside strings). This is especially important in the "explanation" field. The JSON object must exactly adhere to the following format:

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

start_timing = torch.cuda.Event(enable_timing=True)
end_timing = torch.cuda.Event(enable_timing=True)

results = []
start_timing.record()

HaloQuest_llava_df = pd.read_csv("output/HaloQuest_output.csv")

for _, entry in tqdm(HaloQuest_llava_df.iterrows(), total=HaloQuest_llava_df.shape[0], desc="Processing entries"):
   question = entry["question"]
   image_url = entry["image_url"]
   gt_answer = entry["gt_answer"]
   model_answer = entry["model_answer"]
   hallucination_type = entry["hallucination_type"]

   filled_prompt = template.format(hallucination_type, question, gt_answer, model_answer)
   conversation = [
   {
      "role": "user",
      "content": filled_prompt,
   }]

   tokenized = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
   outputs = model.generate(tokenized.to("cuda:0"), max_new_tokens=256)

   input_len = tokenized.shape[-1]
   decoded_model_output = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

   match = re.search(r"({.*?})", decoded_model_output, re.DOTALL)
   model_output = match.group(1)
   
   try:
      parsed_output = json.loads(model_output)
   except Exception as e:
      print("Wrong output format from Llama")
      continue

   explanation = parsed_output["explanation"]
   hallucination = parsed_output["hallucination"]
   rating = parsed_output["rating"]

   result_entry = {
        "image_url": image_url,
        "question": question,
        "gt_answer": gt_answer,
        "llava_model_answer": model_answer,
        "hallucination_type": hallucination_type,
        "llama_hallucination_analysis": explanation,
        "llama_hallucination_evaluation": hallucination,
        "llama_hallucination_rating": rating
    }
   results.append(result_entry)

   print(f"Image url: {image_url}")
   print(f"Processed question: {question}")
   print(f"Ground truth: {gt_answer}")
   print(f"Model answer: {model_answer}")
   print(f"Hallucination type: {hallucination_type}")
   print(model_output)
   print("=" * 50)

torch.cuda.synchronize()
end_timing.record()
print(f"Runtime: {.001 * start_timing.elapsed_time(end_timing):.4f} seconds")

results_df = pd.DataFrame(results)
results_df.to_csv("output/HaloQuest_llama.csv", index=False)