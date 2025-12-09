from flask import Flask, request, jsonify
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("regression_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "API de prédiction des ventes avec Flask"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Récupérer les variables
    tv = data.get("tv")
    radio = data.get("radio")

    if tv is None or radio is None:
        return jsonify({"error": "Veuillez fournir tv et radio"}), 400

    X = np.array([[tv, radio]])
    prediction = model.predict(X)[0]

    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
