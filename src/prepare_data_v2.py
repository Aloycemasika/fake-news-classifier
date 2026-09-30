import pandas as pd
import re


def clean_text(text):
    """Clean common formatting and metadata artifacts from news text."""

    # Convert to string
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")


# Add labels
fake["label"] = 0
true["label"] = 1


# Combine title and article text
fake["content"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
true["content"] = true["title"].fillna("") + " " + true["text"].fillna("")


# Clean the text
fake["content"] = fake["content"].apply(clean_text)
true["content"] = true["content"].apply(clean_text)


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

print("\nFirst 5 cleaned articles:")
print(data.head())


# Save Version 2 dataset
data.to_csv("data/news_v2.csv", index=False)

print("\nCleaned dataset saved to data/news_v2.csv")