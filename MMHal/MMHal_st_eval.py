import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("output/MMHal_output.json", "r") as file:
    data = json.load(file)

ground_truth_answers = [entry["gt_answer"] for entry in data]
model_answers = [entry["model_answer"] for entry in data]

gt_embeddings = model.encode(ground_truth_answers, convert_to_tensor=True)
model_embeddings = model.encode(model_answers, convert_to_tensor=True)

similarities = util.cos_sim(gt_embeddings, model_embeddings)

for i in range(len(data)):
    data[i]["similarity_score"] = similarities[i][i].item()

updated_json_file_path = "output/MMHal_st.json"
with open(updated_json_file_path, "w") as outfile:
    json.dump(data, outfile, indent=4)