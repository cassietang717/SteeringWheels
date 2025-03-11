import json
import pandas as pd

file_output_path1 = "./output/eval/hallsup_eval.jsonl"
file_output_path2 = "./output/eval/hallnotsup_eval.jsonl"
save_out_path = "./output/eval/hall_eval.jsonl"

outline1 = []
outline2 = []

with open(file_output_path1, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        label_text = data.get("tag", "")
        
        if label_text == "Yes":
            outline1.append(data)

with open(file_output_path2, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        label_text = data.get("tag", "")
        
        if label_text == "Yes":
            outline2.append(data)

whole = outline1+outline2

with open(save_out_path, "w", encoding="utf-8") as file:
    for entry in whole:
        file.write(json.dumps(entry) + "\n")

print(save_out_path)