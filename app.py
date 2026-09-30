from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("model/fake_news_model_v2.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer_v2.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body must contain JSON"}), 400

    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if len(text) < 20:
        return jsonify({
            "error": "Please provide a longer news text for analysis"
        }), 400

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(text_tfidf)[0]

    # Get probability of predicted class
    confidence = probabilities[prediction] * 100

    if prediction == 0:
        result = "Fake"
    else:
        result = "True"

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)