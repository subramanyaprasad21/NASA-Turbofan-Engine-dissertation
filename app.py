# app.py
# NASA Turbofan Engine RUL Prediction API
# Contains routes: /, /debug, /loadtest, /predict
# - /loadtest prints sklearn version and inspects the model file (useful for debugging Railway issues)

from flask import Flask, request, jsonify
import joblib, numpy as np, os, traceback, time, sys

app = Flask(__name__)

# Configuration
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model_gradient_boosting.pkl")

FEATURE_NAMES = [
    "setting1","setting2","setting3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

# Lazy-loaded model globals
_model = None
_scaler = None
_load_time = None


# Helper: load model lazily
def load_model():
    """Load model from MODEL_FILENAME into globals _model and _scaler."""
    global _model, _scaler, _load_time
    if _model is not None and _scaler is not None:
        return

    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found in cwd: {os.getcwd()}")

    start = time.time()
    obj = joblib.load(MODEL_FILENAME)

    # Expect a dict with keys 'model' and 'scaler'
    if not isinstance(obj, dict) or "model" not in obj or "scaler" not in obj:
        raise ValueError(
            "Invalid model file format. Expected a dict with keys 'model' and 'scaler'. "
            "Save with: joblib.dump({'model': model, 'scaler': scaler}, 'model_gradient_boosting.pkl')"
        )

    _model = obj["model"]
    _scaler = obj["scaler"]
    _load_time = round(time.time() - start, 2)
    app.logger.info(f"Model loaded from {MODEL_FILENAME} in {_load_time}s")


# Routes

@app.route("/")
def index():
    return {
        "status": "ok",
        "message": "NASA Turbofan Engine RUL Prediction API is running",
        "model_file": MODEL_FILENAME
    }


@app.route("/debug", methods=["GET"])
def debug():
    """List current working directory contents and model presence (quick check)."""
    try:
        files = os.listdir(os.getcwd())
    except Exception as e:
        files = {"error": str(e)}
    return {
        "cwd": os.getcwd(),
        "files": files,
        "model_file": MODEL_FILENAME,
        "model_present": os.path.exists(MODEL_FILENAME),
        "model_loaded": _model is not None
    }


@app.route("/loadtest", methods=["GET"])
def loadtest():
    """
    Verbose loader:
    - reports installed sklearn version
    - reports whether model file exists on disk
    - attempts to joblib.load the file and returns the object's type and keys (if dict)
    - returns last part of traceback if loading fails
    Use this exact endpoint and paste the JSON here if you need me to debug further.
    """
    info = {}
    # sklearn version
    try:
        import sklearn
        info["sklearn_installed_version"] = sklearn.__version__
    except Exception as e:
        info["sklearn_installed_version"] = f"error_importing_sklearn: {e}"

    info["cwd"] = os.getcwd()
    try:
        info["files"] = os.listdir(os.getcwd())
    except Exception as e:
        info["files"] = f"error_listing_cwd: {e}"

    info["model_file_expected"] = MODEL_FILENAME
    info["model_present_on_disk"] = os.path.exists(MODEL_FILENAME)

    if info["model_present_on_disk"]:
        try:
            obj = joblib.load(MODEL_FILENAME)
            info["loaded_type"] = str(type(obj))
            if isinstance(obj, dict):
                try:
                    info["loaded_keys"] = list(obj.keys())
                except Exception:
                    info["loaded_keys"] = "error_fetching_keys"
                # if user stored sklearn version inside model dict
                info["model_contains_sklearn_version"] = obj.get("sklearn_version") if isinstance(obj, dict) else None
            else:
                info["loaded_keys"] = None
                info["model_contains_sklearn_version"] = None
            info["load_error"] = None
        except Exception as e:
            tb = traceback.format_exc().splitlines()[-40:]
            info["loaded_type"] = None
            info["loaded_keys"] = None
            info["model_contains_sklearn_version"] = None
            info["load_error"] = {"message": str(e), "traceback_tail": tb}
    else:
        info["loaded_type"] = None
        info["loaded_keys"] = None
        info["model_contains_sklearn_version"] = None
        info["load_error"] = {"message": "model file not found on disk"}

    return info


@app.route("/predict", methods=["POST"])
def predict():
    """Predict RUL. Expects JSON: { "instances": [[f1,f2,...,f24], [..]] }"""
    try:
        # ensure model is loaded
        load_model()
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-40:]
        return jsonify({
            "status": "error",
            "stage": "model_loading",
            "message": str(e),
            "traceback_tail": tb
        }), 500

    # parse JSON
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"status": "error", "stage": "json_parsing", "message": str(e)}), 400

    instances = data.get("instances")
    if instances is None:
        return jsonify({"status": "error", "stage": "validation", "message": "Missing 'instances' key in JSON"}), 400

    try:
        X = np.array(instances, dtype=float)
    except Exception as e:
        return jsonify({"status": "error", "stage": "validation", "message": f"Could not convert instances to numeric arrays: {e}"}), 400

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(FEATURE_NAMES):
        return jsonify({
            "status": "error",
            "stage": "feature_mismatch",
            "message": f"Each instance must have {len(FEATURE_NAMES)} features.",
            "expected_features": FEATURE_NAMES,
            "received_shape": list(X.shape)
        }), 400

    try:
        Xs = _scaler.transform(X)
        preds = _model.predict(Xs)
        preds = [float(x) for x in preds]
        return jsonify({"status": "ok", "predictions": preds}), 200
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-40:]
        return jsonify({"status": "error", "stage": "prediction", "message": str(e), "traceback_tail": tb}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
