import pandas as pd

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print("FAKE NEWS DATA")
print("Shape:", fake.shape)
print("Columns:", fake.columns.tolist())
print(fake.head())

print("\nTRUE NEWS DATA")
print("Shape:", true.shape)
print("Columns:", true.columns.tolist())
print(true.head())