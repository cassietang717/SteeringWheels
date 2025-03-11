import json
import pandas as pd

# File path
file_path = "./test_dataset/output/answers/hall_answers.jsonl"

subline1 = []
subline2 = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        label_text = data.get("label", "")
        answer_text = data.get("answer", "")
        
        if label_text == "supported" and answer_text == "not_supported":
            subline1.append(data)
        if label_text == "not_supported" and answer_text == "supported":
            subline2.append(data)

save_path_sub1 = "./test_dataset/output/answers/hall_sup.jsonl"
save_path_sub2 = "./test_dataset/output/answers/hall_notsup.jsonl"

with open(save_path_sub1, "w", encoding="utf-8") as file:
    for entry in subline1:
        file.write(json.dumps(entry) + "\n")

with open(save_path_sub2, "w", encoding="utf-8") as file:
    for entry in subline2:
        file.write(json.dumps(entry) + "\n")

print(save_path_sub1)
print(save_path_sub2)