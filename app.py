# app.py (drop-in replacement)
from flask import Flask, request, jsonify
import joblib, numpy as np, os, traceback, time

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model_gradient_boosting.pkl")

app = Flask(__name__)

FEATURE_NAMES = [
    "setting1","setting2","setting3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

# lazy-loaded globals
_model = None
_scaler = None
_model_load_time = None

def load_model():
    global _model, _scaler, _model_load_time
    if _model is None or _scaler is None:
        start = time.time()
        model_path = MODEL_FILENAME
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}' (cwd: {os.getcwd()})")
        obj = joblib.load(model_path)
        # expect dict with keys "model" and "scaler"
        if not isinstance(obj, dict) or "model" not in obj or "scaler" not in obj:
            raise ValueError("Loaded file does not contain dict with keys 'model' and 'scaler'")
        _model = obj["model"]
        _scaler = obj["scaler"]
        _model_load_time = time.time() - start
        app.logger.info(f"Model loaded from {model_path} in {_model_load_time:.2f}s")

@app.route("/")
def index():
    return {"status": "ok", "model_file": MODEL_FILENAME}

@app.route("/debug", methods=["GET"])
def debug():
    try:
        files = os.listdir(os.getcwd())
    except Exception as e:
        files = {"error": str(e)}
    loaded = (_model is not None and _scaler is not None)
    return {
        "cwd": os.getcwd(),
        "files": files,
        "model_loaded": loaded,
        "model_file": MODEL_FILENAME
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ensure model loaded
        load_model()
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Model load error: %s\n%s", e, tb)
        return jsonify({
            "status": "error",
            "stage": "loading_model",
            "message": str(e),
            "traceback": tb.splitlines()[-5:]
        }), 500

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"status": "error", "stage":"parsing_json", "message": str(e)}), 400

    instances = data.get("instances")
    if instances is None:
        return jsonify({"status":"error", "stage":"validation", "message":"Request must include 'instances' key"}), 400

    try:
        X = np.array(instances, dtype=float)
    except Exception as e:
        return jsonify({"status":"error", "stage":"validation", "message":"Could not convert instances to numeric array", "error": str(e)}), 400

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(FEATURE_NAMES):
        return jsonify({
            "status":"error",
            "stage":"validation",
            "message": f"Each instance must have {len(FEATURE_NAMES)} features in order: {FEATURE_NAMES}",
            "received_shape": list(X.shape)
        }), 400

    try:
        Xs = _scaler.transform(X)
        preds = _model.predict(Xs)
        preds = [float(x) for x in preds]
        return jsonify({"status":"ok","predictions": preds}), 200
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Prediction error: %s\n%s", e, tb)
        return jsonify({"status":"error","stage":"predict","message": str(e),"traceback": tb.splitlines()[-5:]}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
