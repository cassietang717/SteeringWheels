import json
import random
from collections import defaultdict

with open("data/color_train_7c.json", "r") as f:
    data = json.load(f)

# Group samples by gt_answer
grouped = defaultdict(list)
for item in data:
    grouped[item["gt_answer"]].append(item)

# Find minimum group size
min_count = min(len(group) for group in grouped.values())

# Sample min_count items from each group
balanced_data = []
for group in grouped.values():
    balanced_data.extend(random.sample(group, min_count))

# Optional: shuffle the final dataset
random.shuffle(balanced_data)

# Save result
with open("data/color_train_7c_balanced.json", "w") as f:
    json.dump(balanced_data, f, indent=2)