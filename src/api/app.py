from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load Model
model = joblib.load("../../notebooks/models/final_fraud_model.pkl")

# Load Threshold
with open("../../notebooks/models/best_threshold.txt", "r") as f:
    THRESHOLD = float(f.read().strip())

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Credit Card Fraud Detection API is running!",
        "status": "success"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Predict probability
        prob = model.predict_proba(df)[0][1]

        # Apply threshold
        fraud = 1 if prob >= THRESHOLD else 0

        return jsonify({
            "fraud_probability": float(prob),
            "threshold": THRESHOLD,
            "fraud_detected": fraud
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)