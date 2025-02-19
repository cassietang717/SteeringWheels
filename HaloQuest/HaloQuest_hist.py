import pandas as pd
import matplotlib.pyplot as plt

tick_fontsize = 16
label_fontsize = 18
title_fontsize = 20

df = pd.read_csv("output/HaloQuest_llama.csv")

percentage = df['llama_hallucination_evaluation'].value_counts(normalize=True) * 100

plt.figure(figsize=(10, 4))
bars = plt.barh(percentage.index, percentage.values, color=['skyblue', 'salmon'], alpha=.75, height=0.3)

plt.xlabel("Percentage (%)", fontsize=label_fontsize)
plt.ylabel("Hallucination Evaluation", fontsize=label_fontsize)
plt.title("Llama Hallucination Evaluation of Llava", fontsize=title_fontsize)

plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.xlim(0, 100)

plt.tight_layout()
plt.savefig("figures/HaloQuest_llama_eval.pdf")