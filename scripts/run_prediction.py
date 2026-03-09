import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.predict import load_model, predict


# Load trained model
model = load_model("models/best_model_random_forest.joblib")


# Example customer data
sample = pd.DataFrame({
    "age": [40],
    "sex": ["male"],
    "bmi": [28],
    "children": [2],
    "smoker": ["no"],
    "region": ["southeast"]
})


# Generate prediction
prediction = predict(model, sample)

print("Predicted insurance charges:", prediction)