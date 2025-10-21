# src/models/train.py
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from datetime import datetime

# Paths
DATA_PATH = Path("data/features/train_FD001_features.csv")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("train_metrics.json")
FIGURES_DIR = Path("reports/figures")

def main():
    # Create dirs if not exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("file:///D:/predictive-maintenance-with-MLOPS/mlruns")  # Your MLflow server
    mlflow.set_experiment("predictive-maintenance")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset shape: {df.shape}")
        
        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("n_features", len(df.columns) - 1)

        # Assume "RUL" is target
        X = df.drop(columns=["RUL"])
        y = df["RUL"]

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log split info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_ratio", 0.2)

        # Model hyperparameters
        n_estimators = 100
        random_state = 42
        max_depth = 10  # Adding for better control
        
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForestRegressor")

        # Model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            n_jobs=-1
        )
        
        print("Training model...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        # Calculate additional metrics
        train_mae = np.mean(np.abs(y_train - y_pred_train))
        val_mae = np.mean(np.abs(y_val - y_pred_val))
        
        metrics = {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "rmse": float(val_rmse),
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "train_mae": float(train_mae),
            "val_mae": float(val_mae)
        }

        # Log metrics to MLflow
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("overfitting_ratio", val_rmse / train_rmse)

        # Save metrics.json for DVC
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        # Save model locally
        joblib.dump(model, MODEL_PATH)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="predictive-maintenance-rf"
        )

        # --- PLOTS ---
        # 1. Predicted vs Actual
        plt.figure(figsize=(10,8))
        
        plt.subplot(2,2,1)
        plt.scatter(y_val, y_pred_val, alpha=0.6, label='Validation')
        plt.scatter(y_train, y_pred_train, alpha=0.3, label='Training')
        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")
        plt.title("Predicted vs Actual RUL")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        plt.legend()

        # 2. Residuals
        plt.subplot(2,2,2)
        residuals_val = y_val - y_pred_val
        plt.scatter(y_pred_val, residuals_val, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")

        # 3. Feature Importance
        plt.subplot(2,2,3)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel("Importance")
        plt.title("Top 10 Feature Importance")

        # 4. Error Distribution
        plt.subplot(2,2,4)
        plt.hist(residuals_val, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")

        plt.tight_layout()
        plot_path = FIGURES_DIR / "training_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log plot to MLflow
        mlflow.log_artifact(str(plot_path))
        plt.close()

        # Individual plots for DVC
        plt.figure(figsize=(6,6))
        plt.scatter(y_val, y_pred_val, alpha=0.6)
        plt.xlabel("Actual RUL")
        plt.ylabel("Predicted RUL")
        plt.title("Predicted vs Actual RUL")
        plt.plot([y_val.min(), y_val.max()],
                 [y_val.min(), y_val.max()],
                 "r--")
        plt.savefig(FIGURES_DIR / "pred_vs_actual.png", dpi=300)
        plt.close()

        residuals = y_val - y_pred_val
        plt.figure(figsize=(6,4))
        plt.scatter(y_pred_val, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted RUL")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")
        plt.savefig(FIGURES_DIR / "residuals.png", dpi=300)
        plt.close()

        # Log feature importance as artifact
        feature_importance.to_csv(FIGURES_DIR / "feature_importance.csv", index=False)
        mlflow.log_artifact(str(FIGURES_DIR / "feature_importance.csv"))

        print("✅ Training complete. Model + metrics + plots saved.")
        print(f"Validation RMSE: {val_rmse:.2f}")
        print(f"Validation R²: {val_r2:.3f}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()