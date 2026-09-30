import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load prepared dataset
data = pd.read_csv("data/news.csv")

# Separate features and labels
X = data["content"]
y = data["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)

# Train classifier
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake", "True"]
))

# Save model and vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved to model/fake_news_model.pkl")
print("Vectorizer saved to model/tfidf_vectorizer.pkl")