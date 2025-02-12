import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from transformers import LlamaTokenizer, LlamaForCausalLM
import re

tokenizer = LlamaTokenizer.from_pretrained("/net/scratch2/steeringwheel/llama-3.1-8B")
model = LlamaForCausalLM.from_pretrained("/net/scratch2/steeringwheel/llama-3.1-8B")

model.to("cuda:0")
model = torch.compile(model)

def extract_explanation(response):
    """Extracts the explanation section from the model's response."""
    match = re.search(r"(?i)Explanation:\s*(.*)", response, re.DOTALL)
    if match:
        return match.group(1).strip()  # Extract the explanation text
    return "No explanation found."


# Load the LLaMA model and tokenizer (make sure to specify the correct model checkpoint)
model_dir = "/net/scratch2/steeringwheel/llama-3.1-8B"  # Update with the correct path
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")
model.eval()

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
Your response **must** be structured in the following **JSON format**:

```json
{
  "explanation": "Your explanation here",
  "hallucination": "yes" | "no",
  "rating": rating_value
}
'''

def generate_response(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the response using the LLaMA model
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512, 
            temperature=0.7, 
            top_p=0.9,
            repetition_penalty=1.2
        )
    # Decode the generated tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='output/MMHal_output.json', help='response file containing images, questions, and model responses')
    args = parser.parse_args()

    # Load the JSON file
    with open(args.response, 'r') as f:
        records = json.load(f)

    # assert len(records) == 94  # Ensure the dataset has 96 entries

    # Open the output file in append mode to write incrementally
    updated_json_file_path = "output/MMHal_llama.json"

    results = []

    for i, record in enumerate(records):
        image_content = ', '.join(record['image_content'])
        input_text = template.format(image_content, record['question'], record['gt_answer'], record['model_answer'])

        # Generate a response using LLaMA
        print(f"Processing record {i}...")
        response = generate_response(input_text)
        print(f"Response: {response}")

        # Extract the score from the response (look for rating)
        score = 0  # Default score is 0
        for s in range(7):
            if f'rating: {s}' in response.lower():
                score = s
                break
        explanation = extract_explanation(response)

        # Determine if hallucination is present based on the score
        hallucination_flag = 0 if score >= 3 else 1

        # Create the result entry
        result_entry = {
            "index": i+1,
            "image": record['image_src'],
            "question": record['question'],
            "gt_answer": record['gt_answer'],
            "model_answer": record['model_answer'],
            "analysis": explanation,
            "similarity_score": score,
            "hallucination": hallucination_flag
        }
        results.append(result_entry)

    with open(updated_json_file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Scores and hallucination information have been saved incrementally to {updated_json_file_path}")
