import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load Version 2 dataset
data = pd.read_csv("data/news_v2.csv")

# Features and labels
X = data["content"].fillna("")
y = data["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)


# Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)


# Predictions
y_pred = model.predict(X_test_tfidf)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "True"]
    )
)


# Save Version 2 model
joblib.dump(model, "model/fake_news_model_v2.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer_v2.pkl")

print("\nVersion 2 model saved to model/fake_news_model_v2.pkl")
print("Version 2 vectorizer saved to model/tfidf_vectorizer_v2.pkl")