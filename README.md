
# RUL Prediction API Repository (Ready for Railway)

This repository contains:
- Trained model: model_gradient_boosting.pkl
- API: app.py (Flask)
- Training scripts: train_gradient_boosting.py, train_xgboost.py
- Requirements: requirements.txt
- Procfile for Railway/Heroku
- Example data files: PM_train.csv, PM_test.csv, PM_truth.csv (already included)

## Quickstart (local)
1. Create virtualenv and install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   (If planning to run XGBoost training locally on macOS: `brew install libomp`)

2. Train model (GradientBoosting):
   python train_gradient_boosting.py
   -> produces model_gradient_boosting.pkl

   Or train XGBoost (if you have libomp):
   python train_xgboost.py
   -> produces model_xgboost.pkl

3. Run API locally:
   gunicorn app:app
   or
   python app.py

4. Test:
   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"instances":[[...features...]]}'

## Deployment to Railway
1. Push this repository to GitHub.
2. In Railway, create a new project and connect GitHub repo.
3. Railway will install dependencies from requirements.txt and run the Procfile.
4. Make sure to include the trained model file (model_gradient_boosting.pkl or model_xgboost.pkl) in the repo or host it remotely and download in app startup.

