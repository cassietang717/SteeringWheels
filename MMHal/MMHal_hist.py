import json
import pandas as pd
import matplotlib.pyplot as plt

tick_fontsize = 16
label_fontsize = 18
title_fontsize = 20

file_path = 'output/MMHal_llava.json'
with open(file_path, "r") as file:
    new_data = json.load(file)

# Extract the hallucination ratings
new_df = pd.DataFrame(new_data)
new_df = new_df[["llama_hallucination_rating"]]

# Count the occurrences of each score
score_counts_new = new_df["llama_hallucination_rating"].value_counts().sort_index()

# Plot bar chart for the new similarity score counts
plt.figure(figsize=(8, 6))
plt.bar(score_counts_new.index, score_counts_new.values, color="skyblue", edgecolor="black", alpha=0.75)
plt.xlabel("Llama Hallucination Rating", fontsize=label_fontsize)
plt.ylabel("Count", fontsize=label_fontsize)
plt.title("Count of Each Llama Hallucination Rating", fontsize=title_fontsize)
plt.xticks(score_counts_new.index, fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(axis="y", alpha=0.75)

plt.savefig("figures/MMHal_llama_eval.pdf")