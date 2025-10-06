# ============================================================
#  NASA Turbofan Engine RUL Prediction API
#  Flask + Scikit-learn (Gradient Boosting)
# ============================================================

from flask import Flask, request, jsonify
import joblib, numpy as np, os, traceback, time

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model_gradient_boosting.pkl")

FEATURE_NAMES = [
    "setting1","setting2","setting3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

# Lazy loading to prevent Railway cold start timeout
_model = None
_scaler = None
_load_time = None


# ============================================================
# Helper: Load Model
# ============================================================
def load_model():
    """Loads model and scaler when first needed."""
    global _model, _scaler, _load_time
    if _model is None or _scaler is None:
        start = time.time()
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(f"Model file not found: {MODEL_FILENAME}")
        obj = joblib.load(MODEL_FILENAME)
        if not isinstance(obj, dict) or "model" not in obj or "scaler" not in obj:
            raise ValueError("Invalid model file format. Expected dict with 'model' and 'scaler' keys.")
        _model = obj["model"]
        _scaler = obj["scaler"]
        _load_time = round(time.time() - start, 2)
        app.logger.info(f"✅ Model loaded successfully in {_load_time}s from {MODEL_FILENAME}")


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Basic health check route."""
    return {
        "status": "ok",
        "message": "NASA Turbofan Engine RUL Prediction API is running",
        "model_file": MODEL_FILENAME
    }


@app.route("/debug", methods=["GET"])
def debug():
    """Check if the model file exists and if model is loaded."""
    files = os.listdir(os.getcwd())
    return {
        "cwd": os.getcwd(),
        "files": files,
        "model_loaded": _model is not None,
        "model_file": MODEL_FILENAME
    }


@app.route("/predict", methods=["POST"])
def predict():
    """Predict Remaining Useful Life (RUL)."""
    try:
        load_model()
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({
            "status": "error",
            "stage": "model_loading",
            "message": str(e),
            "traceback": tb.splitlines()[-5:]
        }), 500

    try:
        data = request.get_json(force=True)
        instances = data.get("instances")
        if instances is None:
            return jsonify({
                "status": "error",
                "stage": "input_validation",
                "message": "Missing 'instances' key in JSON."
            }), 400

        X = np.array(instances, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(FEATURE_NAMES):
            return jsonify({
                "status": "error",
                "stage": "input_validation",
                "message": f"Each instance must have {len(FEATURE_NAMES)} features.",
                "expected_features": FEATURE_NAMES,
                "received_shape": list(X.shape)
            }), 400

        X_scaled = _scaler.transform(X)
        preds = _model.predict(X_scaled)
        preds = [float(x) for x in preds]

        return jsonify({
            "status": "ok",
            "predictions": preds
        }), 200

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Prediction error: %s", tb)
        return jsonify({
            "status": "error",
            "stage": "prediction",
            "message": str(e),
            "traceback": tb.splitlines()[-5:]
        }), 500


# ============================================================
# Entry Point (for local run)
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
