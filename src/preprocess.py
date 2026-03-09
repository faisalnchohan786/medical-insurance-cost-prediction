import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load dataset and validate required columns."""
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and type enforcement."""
    df = df[REQUIRED_COLUMNS].copy()
    df = df.drop_duplicates()

    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip()

    numeric_cols = ["age", "bmi", "children", "charges"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    return df