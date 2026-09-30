import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Get feature names
feature_names = np.array(vectorizer.get_feature_names_out())

# Get Logistic Regression coefficients
coefficients = model.coef_[0]

# Sort features by coefficient
sorted_indices = np.argsort(coefficients)

# Strongest Fake features
fake_indices = sorted_indices[:20]

# Strongest True features
true_indices = sorted_indices[-20:][::-1]

print("\nTop features associated with FAKE news:")
print("----------------------------------------")

for index in fake_indices:
    print(f"{feature_names[index]:<20} {coefficients[index]:.4f}")

print("\nTop features associated with TRUE news:")
print("----------------------------------------")

for index in true_indices:
    print(f"{feature_names[index]:<20} {coefficients[index]:+.4f}")