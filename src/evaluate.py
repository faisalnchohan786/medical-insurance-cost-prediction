from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass(frozen=True)
class Metrics:
    r2: float
    mae: float
    mse: float
    rmse: float

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """
    Compute regression evaluation metrics.
    Returns R², MAE, MSE and RMSE.
    """
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return Metrics(r2=r2, mae=mae, mse=mse, rmse=rmse)
