# src/models/evaluate_model.py
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = Path("data/features/train_FD001_features.csv")
MODEL_PATH = Path("models/model.pkl")
EVAL_METRICS_PATH = Path("evaluation_metrics.json")  # Different from training metrics
EVAL_PLOTS_DIR = Path("plots")  # To match dvc.yaml

def main():
    EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data + model
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    X = df.drop(columns=["RUL"])
    y = df["RUL"]

    # Use same split as training for consistency
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predictions
    y_pred = model.predict(X_val)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mae = np.mean(np.abs(y_val - y_pred))
    
    metrics = {
        "rmse": float(rmse), 
        "r2": float(r2),
        "mae": float(mae)
    }

    # Save metrics (DVC expects eval_metrics.json)
    with open("eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Create residuals plot
    residuals = y_val - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted RUL")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.savefig(EVAL_PLOTS_DIR / "residuals.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save residuals data for DVC plots
    residuals_df = pd.DataFrame({
        'predicted': y_pred,
        'residuals': residuals,
        'actual': y_val
    })
    residuals_df.to_csv(EVAL_PLOTS_DIR / "residuals.csv", index=False)

    print("✅ Evaluation complete. Metrics + plots saved.")
    print(f"RMSE: {rmse:.2f}, R²: {r2:.3f}, MAE: {mae:.2f}")

if __name__ == "__main__":
    main()