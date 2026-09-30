import joblib

# Load Version 2 model and vectorizer
model = joblib.load("model/fake_news_model_v2.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer_v2.pkl")


def test_text(text):
    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    confidence = probabilities[prediction] * 100

    result = "Fake" if prediction == 0 else "True"

    print(f"\nText: {text}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")


base_text = "The government announced new economic measures today."

print("\n========== BASELINE ==========")
test_text(base_text)

print("\n========== SOURCE WORD TESTS ==========")

test_text(
    "Reuters reports that the government announced new economic measures today."
)

test_text(
    "Washington reports that the government announced new economic measures today."
)

test_text(
    "The government issued a statement announcing new economic measures today."
)

test_text(
    "A video shows the government announcing new economic measures today."
)

test_text(
    "An image shows the government announcing new economic measures today."
)