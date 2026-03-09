import sys
from pathlib import Path

# project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import ROOT_DIR, MODELS_DIR
from src.predict import load_model, predict

print("Starting pipeline test...")

# Load data
data_path = ROOT_DIR / "data" / "raw" / "insurance.csv"
df = pd.read_csv(data_path)

print("Data loaded:", df.shape)

# Load model
model_path = MODELS_DIR / "best_model_random_forest.joblib"
model = load_model(model_path)

print("Model loaded successfully")

# Remove target column if present
if "charges" in df.columns:
    df = df.drop(columns=["charges"])

# Run predictions
preds = predict(model, df)

print("Predictions generated:", len(preds))
print("Sample predictions:", preds[:5])

print("Pipeline test completed successfully")