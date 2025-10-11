# tests/test_model_performance.py
"""
Model performance validation tests
"""
import pytest
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestModelPerformance:
    """Test model performance meets minimum requirements"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data and model paths"""
        self.model_path = Path("models/model.pkl")
        self.metrics_path = Path("train_metrics.json")
        self.test_data_path = Path("data/features/train_FD001_features.csv")
        
        # Performance thresholds
        self.min_r2_score = 0.8  # Minimum R² score
        self.max_rmse = 50.0     # Maximum RMSE
        self.max_mae = 30.0      # Maximum MAE
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        assert self.model_path.exists(), f"Model file not found at {self.model_path}"
    
    def test_model_loads_successfully(self):
        """Test that model can be loaded without errors"""
        try:
            model = joblib.load(self.model_path)
            assert model is not None
            assert hasattr(model, 'predict'), "Model should have predict method"
        except Exception as e:
            pytest.fail(f"Failed to load model: {str(e)}")
    
    def test_model_performance_thresholds(self):
        """Test that model meets minimum performance requirements"""
        if not self.metrics_path.exists():
            pytest.skip("No metrics file found - run training first")
        
        with open(self.metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Test R² score
        val_r2 = metrics.get('val_r2', 0)
        assert val_r2 >= self.min_r2_score, \
            f"R² score {val_r2:.3f} below minimum {self.min_r2_score}"
        
        # Test RMSE
        val_rmse = metrics.get('val_rmse', float('inf'))
        assert val_rmse <= self.max_rmse, \
            f"RMSE {val_rmse:.2f} above maximum {self.max_rmse}"
        
        # Test overfitting
        train_rmse = metrics.get('train_rmse', 0)
        if train_rmse > 0:
            overfitting_ratio = val_rmse / train_rmse
            assert overfitting_ratio <= 1.5, \
                f"Model overfitting detected: ratio {overfitting_ratio:.2f}"
    
    def test_model_prediction_quality(self):
        """Test model predictions on sample data"""
        if not self.model_path.exists():
            pytest.skip("Model file not found - run training first")
        
        if not self.test_data_path.exists():
            pytest.skip("Test data not found")
        
        # Load model and data
        model = joblib.load(self.model_path)
        df = pd.read_csv(self.test_data_path)
        
        # Prepare test data
        X = df.drop(columns=["RUL"]).head(100)  # Use first 100 samples
        y = df["RUL"].head(100)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Test prediction validity
        assert len(predictions) == len(y), "Prediction length mismatch"
        assert not np.any(np.isnan(predictions)), "Model produced NaN predictions"
        assert not np.any(np.isinf(predictions)), "Model produced infinite predictions"
        
        # Test prediction range (RUL should be positive)
        assert np.all(predictions >= 0), "Model produced negative RUL predictions"
        assert np.all(predictions <= 1000), "Model produced unreasonably high RUL predictions"
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        
        # Assert performance thresholds
        assert r2 >= self.min_r2_score, f"R² score {r2:.3f} below minimum"
        assert rmse <= self.max_rmse, f"RMSE {rmse:.2f} above maximum"
        assert mae <= self.max_mae, f"MAE {mae:.2f} above maximum"
    
    def test_model_feature_importance(self):
        """Test that model has reasonable feature importance"""
        if not self.model_path.exists():
            pytest.skip("Model file not found")
        
        model = joblib.load(self.model_path)
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Test feature importance properties
            assert len(importance) > 0, "Model has no feature importances"
            assert np.sum(importance) > 0, "All feature importances are zero"
            assert np.all(importance >= 0), "Negative feature importances found"
            
            # Check that importance sums approximately to 1 (for tree-based models)
            if hasattr(model, 'n_estimators'):
                importance_sum = np.sum(importance)
                assert 0.99 <= importance_sum <= 1.01, \
                    f"Feature importance sum {importance_sum:.3f} not normalized"
    
    def test_model_consistency(self):
        """Test that model predictions are consistent"""
        if not self.model_path.exists():
            pytest.skip("Model file not found")
        
        if not self.test_data_path.exists():
            pytest.skip("Test data not found")
        
        model = joblib.load(self.model_path)
        df = pd.read_csv(self.test_data_path)
        X_sample = df.drop(columns=["RUL"]).head(10)
        
        # Make multiple predictions on same data
        pred1 = model.predict(X_sample)
        pred2 = model.predict(X_sample)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10, 
                                   err_msg="Model predictions are not consistent")
    def test_model_memory_usage(self):
        """Test that model memory usage is reasonable"""
        if not self.model_path.exists():
            pytest.skip("Model file not found")
        
        import os
        
        # Check model file size (should be less than 100MB)
        file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        assert file_size_mb < 100, f"Model file too large: {file_size_mb:.2f} MB"
        
        # Check memory usage when loading
        import psutil
        import gc
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        model = joblib.load(self.model_path)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = memory_after - memory_before
        
        # Model should use less than 500MB when loaded
        assert memory_used < 500, f"Model uses too much memory: {memory_used:.2f} MB"
        
        # Cleanup
        del model
        gc.collect()

class TestModelRobustness:
    """Test model robustness and edge cases"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for robustness tests"""
        self.model_path = Path("models/model.pkl")
        self.test_data_path = Path("data/features/train_FD001_features.csv")
    
    def test_model_handles_edge_cases(self):
        """Test model behavior with edge case inputs"""
        if not self.model_path.exists():
            pytest.skip("Model file not found")
        
        model = joblib.load(self.model_path)
        
        if not self.test_data_path.exists():
            pytest.skip("Test data not found")
        
        df = pd.read_csv(self.test_data_path)
        X_normal = df.drop(columns=["RUL"]).iloc[0:1]  # Single row
        
        # Test with normal data
        pred_normal = model.predict(X_normal)
        assert len(pred_normal) == 1
        assert not np.isnan(pred_normal[0])
        
        # Test with extreme values (within sensor ranges but at boundaries)
        X_extreme = X_normal.copy()
        
        # Set some sensors to high values (but within reasonable bounds)
        for col in X_extreme.columns:
            if 'sensor' in col.lower():
                X_extreme[col] = X_extreme[col] * 1.5  # 50% increase
        
        pred_extreme = model.predict(X_extreme)
        assert len(pred_extreme) == 1
        assert not np.isnan(pred_extreme[0])
        
        # Predictions should be different but reasonable
        assert abs(pred_extreme[0] - pred_normal[0]) < 1000  # Not wildly different
    
    def test_model_batch_vs_single_predictions(self):
        """Test that batch predictions match single predictions"""
        if not self.model_path.exists():
            pytest.skip("Model file not found")
        
        if not self.test_data_path.exists():
            pytest.skip("Test data not found")
        
        model = joblib.load(self.model_path)
        df = pd.read_csv(self.test_data_path)
        X_test = df.drop(columns=["RUL"]).head(5)
        
        # Batch prediction
        batch_pred = model.predict(X_test)
        
        # Individual predictions
        single_preds = []
        for i in range(len(X_test)):
            single_pred = model.predict(X_test.iloc[i:i+1])
            single_preds.append(single_pred[0])
        
        # Should be identical
        np.testing.assert_array_almost_equal(
            batch_pred, single_preds, decimal=6,
            err_msg="Batch and single predictions don't match"
        )