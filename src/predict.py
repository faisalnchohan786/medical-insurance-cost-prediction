import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def load_model(path: str | Path):
    """Load trained model from disk."""
    return joblib.load(path)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Generate insurance charge predictions."""
    
    pred_log = model.predict(df)
    pred = np.expm1(pred_log)

    return np.round(pred, 2)