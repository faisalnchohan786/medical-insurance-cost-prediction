from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from src.config import Paths
from src.preprocess import load_data, basic_clean
from src.evaluate import regression_metrics
from src.utils import ensure_dir, save_json


# -----------------------------
# Plot style
# -----------------------------

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "figure.figsize": (8, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

MAIN_COLOR = "#2E86AB"
SECONDARY_COLOR = "#F18F01"


# -----------------------------
# Feature definitions
# -----------------------------

NUMERIC = ["age", "bmi"]
CATEGORICAL = ["sex", "smoker", "region", "children"]
TARGET = "charges"


# -----------------------------
# Preprocessing
# -----------------------------

def build_preprocessor():

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )


# -----------------------------
# Models
# -----------------------------

def build_models(seed):

    return {
        "linear_regression": LinearRegression(),

        "ridge_regression": Ridge(alpha=1.0),

        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1
        )
    }


# -----------------------------
# Main Training Pipeline
# -----------------------------

def main():

    parser = argparse.ArgumentParser(description="Train insurance pricing models")

    parser.add_argument("--data", default=Paths().data_path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    paths = Paths()

    ensure_dir(paths.reports_dir)
    ensure_dir(paths.images_dir)

    # -----------------------------
    # Load + Clean Data
    # -----------------------------

    df = basic_clean(load_data(args.data))

    X = df[NUMERIC + CATEGORICAL]
    y = np.log1p(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed
    )

    preprocessor = build_preprocessor()
    models = build_models(args.seed)

    all_metrics = {}
    preds_out = []

    best_model_name = None
    best_r2 = -np.inf
    best_pipe = None
    best_pred = None

    # -----------------------------
    # Train Models
    # -----------------------------

    for name, model in models.items():

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        m = regression_metrics(y_test, y_pred)

        all_metrics[name] = {
            "r2": m.r2,
            "mae": m.mae,
            "mse": m.mse,
            "rmse": m.rmse
        }

        if m.r2 > best_r2:
            best_r2 = m.r2
            best_model_name = name
            best_pipe = pipe
            best_pred = y_pred

        preds_out.append(
            pd.DataFrame({
                "model": name,
                "y_true_log": y_test,
                "y_pred_log": y_pred
            })
        )

    # -----------------------------
    # Combine Predictions
    # -----------------------------

    pred_df = pd.concat(preds_out, ignore_index=True)

    pred_df["y_true_charges"] = np.expm1(pred_df["y_true_log"])
    pred_df["y_pred_charges"] = np.expm1(pred_df["y_pred_log"])
    pred_df["error_charges"] = pred_df["y_true_charges"] - pred_df["y_pred_charges"]

    pred_df = pred_df[[
        "model",
        "y_true_charges",
        "y_pred_charges",
        "error_charges"
    ]].round(2)

    # -----------------------------
    # Save metrics
    # -----------------------------

    save_json(
        all_metrics,
        os.path.join(paths.reports_dir, "metrics.json")
    )

    metrics_df = pd.DataFrame(all_metrics).T.round(4)

    metrics_df.to_csv(
        os.path.join(paths.reports_dir, "model_comparison.csv")
    )

    # -----------------------------
    # Save predictions
    # -----------------------------

    pred_df.to_csv(
        os.path.join(paths.reports_dir, "predictions.csv"),
        index=False
    )

    # -----------------------------
    # Save best model
    # -----------------------------

    joblib.dump(
        best_pipe,
        os.path.join(paths.models_dir, f"best_model_{best_model_name}.joblib")
    )

    # -----------------------------
    # Model comparison table image
    # -----------------------------

    plt.figure(figsize=(6,3))
    plt.axis("off")

    table = plt.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        rowLabels=metrics_df.index,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(
        os.path.join(paths.images_dir, "model_comparison.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Actual vs Predicted
    # -----------------------------

    y_true_real = np.expm1(y_test)
    y_pred_real = np.expm1(best_pred)

    plt.scatter(
        y_true_real,
        y_pred_real,
        alpha=0.6,
        color=MAIN_COLOR
    )

    max_val = max(y_true_real.max(), y_pred_real.max())

    plt.plot([0, max_val], [0, max_val], "r--")

    plt.xlabel("Actual Charges ($)")
    plt.ylabel("Predicted Charges ($)")
    plt.title(f"Actual vs Predicted ({best_model_name})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(paths.images_dir, "actual_vs_predicted.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Residual Plot
    # -----------------------------

    residuals = y_true_real - y_pred_real

    plt.scatter(
        y_pred_real,
        residuals,
        alpha=0.6,
        color=MAIN_COLOR
    )

    plt.axhline(0, color=SECONDARY_COLOR)

    plt.xlabel("Predicted Charges")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted ({best_model_name})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(paths.images_dir, "residuals_best_model.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Feature Importance
    # -----------------------------

    if best_model_name == "random_forest":

        feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
        importances = best_pipe.named_steps["model"].feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })

        importance_df["feature"] = (
            importance_df["feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
            .str.replace("_", " ")
        )

        importance_df = importance_df.sort_values("importance", ascending=False).head(10)

        plt.barh(
            importance_df["feature"],
            importance_df["importance"],
            color=MAIN_COLOR
        )

        plt.gca().invert_yaxis()

        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title("Top 10 Feature Importances — Random Forest")

        plt.savefig(
            os.path.join(paths.images_dir, "feature_importance.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # -----------------------------
    # Learning Curve
    # -----------------------------

    train_sizes, train_scores, test_scores = learning_curve(
        best_pipe,
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label="Training Score", color=MAIN_COLOR)
    plt.plot(train_sizes, test_mean, label="Validation Score", color=SECONDARY_COLOR)

    plt.xlabel("Training Size")
    plt.ylabel("R² Score")
    plt.title(f"Learning Curve ({best_model_name})")

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(paths.images_dir, "learning_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Artifacts written to reports/ and images/")


if __name__ == "__main__":
    main()