import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load your data
data = pd.read_csv("haloquest.csv")

# Step 2: Split into train and test
train_data, test_data = train_test_split(
    data,
    test_size=0.1,   # 10% for test set
    random_state=42, # set a random seed for reproducibility
    shuffle=True     # shuffle the data before splitting
)

# Now you have:
# train_data (90%)
# test_data (10%)
# test_data = test_data[test_data["hallucination type"] == "visual challenge"]

test_data.to_csv("test_haloquest.csv", index=False)