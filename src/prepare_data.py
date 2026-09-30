import pandas as pd

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine title and article text
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

# Keep only what we need
fake = fake[["content", "label"]]
true = true[["content", "label"]]

# Combine both datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total articles:", len(data))
print("\nLabel distribution:")
print(data["label"].value_counts())

print("\nFirst 5 rows:")
print(data.head())

# Save prepared dataset
data.to_csv("data/news.csv", index=False)

print("\nPrepared dataset saved to data/news.csv")