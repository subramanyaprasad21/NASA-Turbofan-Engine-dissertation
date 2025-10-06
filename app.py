
from flask import Flask, request, jsonify
import joblib, numpy as np, os

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model_gradient_boosting.pkl")

obj = joblib.load(MODEL_FILENAME)
model = obj["model"]
scaler = obj["scaler"]

app = Flask(__name__)

FEATURE_NAMES = [
    "setting1","setting2","setting3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

@app.route("/")
def index():
    return {"status": "ok", "model": MODEL_FILENAME}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    instances = data.get("instances")
    if instances is None:
        return jsonify({"error": "Request must include 'instances' key with a list of feature lists."}), 400

    X = np.array(instances, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(FEATURE_NAMES):
        return jsonify({
            "error": f"Each instance must have {len(FEATURE_NAMES)} features in order: {FEATURE_NAMES}"
        }), 400

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled).tolist()
    return jsonify({"predictions": preds}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
