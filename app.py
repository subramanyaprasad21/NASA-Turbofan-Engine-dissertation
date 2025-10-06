# ============================================================
#  NASA Turbofan Engine RUL Prediction API
#  Fully fixed Flask app for Railway
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

# Globals for lazy loading
_model = None
_scaler = None
_load_time = None


# ============================================================
# Helper: Load model
# ============================================================
def load_model():
    """Loads the model and scaler from file if not already loaded."""
    global _model, _scaler, _load_time
    if _model is not None and _scaler is not None:
        return

    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found in {os.getcwd()}")

    start = time.time()
    obj = joblib.load(MODEL_FILENAME)

    # Ensure correct format
    if not isinstance(obj, dict) or "model" not in obj or "scaler" not in obj:
        raise ValueError(
            "Invalid model file. Expected a dictionary with keys 'model' and 'scaler'. "
            "Please retrain and save using: joblib.dump({'model': model, 'scaler': scaler}, 'model_gradient_boosting.pkl')"
        )

    _model = obj["model"]
    _scaler = obj["scaler"]
    _load_time = round(time.time() - start, 2)
    app.logger.info(f"✅ Model loaded successfully in {_load_time}s")


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Simple health check."""
    return {
        "status": "ok",
        "message": "NASA Turbofan Engine RUL Prediction API is running",
        "model_file": MODEL_FILENAME
    }


@app.route("/debug", methods=["GET"])
def debug():
    """List current directory and model status."""
    files = os.listdir(os.getcwd())
    return {
        "cwd": os.getcwd(),
        "files": files,
        "model_file": MODEL_FILENAME,
        "model_loaded": _model is not None
    }


@app.route("/loadtest", methods=["GET"])
def loadtest():
    """Try to load the model and return success or detailed error message."""
    try:
        load_model()
        return {"status": "success", "message": "Model loaded successfully!"}
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-10:]
        return {
            "status": "error",
            "message": str(e),
            "traceback": tb
        }, 500


@app.route("/predict", methods=["POST"])
def predict():
    """Perform prediction using the trained model."""
    try:
        load_model()
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-10:]
        return jsonify({
            "status": "error",
            "stage": "model_loading",
            "message": str(e),
            "traceback": tb
        }), 500

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"status": "error", "stage": "json_parsing", "message": str(e)}), 400

    instances = data.get("instances")
    if instances is None:
        return jsonify({
            "status": "error",
            "stage": "input_validation",
            "message": "Missing 'instances' key in JSON input."
        }), 400

    try:
        X = np.array(instances, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
    except Exception as e:
        return jsonify({
            "status": "error",
            "stage": "data_conversion",
            "message": f"Could not convert input to numeric array: {e}"
        }), 400

    if X.shape[1] != len(FEATURE_NAMES):
        return jsonify({
            "status": "error",
            "stage": "feature_mismatch",
            "message": f"Expected {len(FEATURE_NAMES)} features per instance.",
            "expected_features": FEATURE_NAMES,
            "received_shape": list(X.shape)
        }), 400

    try:
        X_scaled = _scaler.transform(X)
        preds = _model.predict(X_scaled)
        preds = [float(p) for p in preds]
        return jsonify({
            "status": "ok",
            "predictions": preds
        }), 200
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-10:]
        return jsonify({
            "status": "error",
            "stage": "prediction",
            "message": str(e),
            "traceback": tb
        }), 500


# ============================================================
# Run app (for local debugging)
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
