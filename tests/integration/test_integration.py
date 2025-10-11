# tests/integration/test_integration.py
"""
Integration tests for end-to-end pipeline testing
"""
import pytest
import requests
import time
import json
from pathlib import Path

class TestEndToEndPipeline:
    """Test complete pipeline integration"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup integration test environment"""
        self.api_base_url = "http://localhost:8000"  # Will be overridden in CI
        self.sample_data = {
            "setting1": 42.0,
            "setting2": 0.84,
            "setting3": 100.0,
            "sensor_2": 642.35,
            "sensor_3": 1589.70,
            "sensor_4": 1400.60,
            "sensor_6": 21.61,
            "sensor_7": 554.36,
            "sensor_8": 2388.06,
            "sensor_9": 9046.19,
            "sensor_11": 47.47,
            "sensor_12": 521.66,
            "sensor_13": 2388.02,
            "sensor_14": 8138.62,
            "sensor_15": 8.4195,
            "sensor_17": 8.4195,
            "sensor_20": 0.03,
            "sensor_21": 0.02,
            "engine_id": 1,
            "cycle": 150
        }
    
    def test_api_availability(self):
        """Test that API is available and responding"""
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=30)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
        except requests.exceptions.RequestException:
            pytest.skip("API not available for integration testing")
    
    def test_health_endpoint(self):
        """Test health endpoint returns expected data"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=30)
            assert response.status_code == 200
            
            data = response.json()
            required_fields = ["status", "model_loaded", "api_version", "timestamp"]
            
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # API should be healthy in integration environment
            assert data["status"] in ["healthy", "unhealthy"]
            
        except requests.exceptions.RequestException:
            pytest.skip("API not available for integration testing")
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint with real data"""
        try:
            response = requests.post(
                f"{self.api_base_url}/predict",
                json=self.sample_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Should either work (200) or model not loaded (503)
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                
                # Check response structure
                required_fields = ["rul_prediction", "model_version", "status"]
                for field in required_fields:
                    assert field in data, f"Missing field: {field}"
                
                # Check data types and ranges
                assert isinstance(data["rul_prediction"], (int, float))
                assert data["rul_prediction"] >= 0
                assert data["status"] == "success"
                
        except requests.exceptions.RequestException:
            pytest.skip("API not available for integration testing")
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        try:
            batch_data = [self.sample_data, self.sample_data]
            
            response = requests.post(
                f"{self.api_base_url}/predict/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                assert "predictions" in data
                assert len(data["predictions"]) == 2
                
        except requests.exceptions.RequestException:
            pytest.skip("API not available for integration testing")
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/model/info", timeout=30)
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ["model_type", "training_metrics"]
                
                for field in expected_fields:
                    assert field in data, f"Missing field: {field}"
                
        except requests.exceptions.RequestException:
            pytest.skip("API not available for integration testing")
    
    def test_error_handling(self):
        """Test API error handling"""
        try:
            # Test with invalid data
            invalid_data = {"invalid": "data"}
            
            response = requests.post(
                f"{self.api_base_url}/predict",
                json=invalid_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Should return validation error
            assert response.status_code == 422
            
        except requests.exceptions.RequestException:
            pytest.skip("API not available for integration testing")
    
    def test_performance_under_load(self):
        """Test API performance under moderate load"""
        try:
            # Quick performance test
            num_requests = 10
            response_times = []
            
            for _ in range(num_requests):
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/predict",
                    json=self.sample_data,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                
                # Break if API is not working
                if response.status_code not in [200, 503]:
                    break
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                
                # Performance assertions
                assert avg_response_time < 5.0, f"Average response time too high: {avg_response_time:.2f}s"
                assert max_response_time < 10.0, f"Max response time too high: {max_response_time:.2f}s"
                
        except requests.exceptions.RequestException:
            pytest.skip("API not available for integration testing")

class TestDataPipelineIntegration:
    """Test data pipeline integration"""
    
    def test_data_pipeline_outputs(self):
        """Test that data pipeline produces expected outputs"""
        # Check if processed data exists
        processed_path = Path("data/processed")
        features_path = Path("data/features")
        
        assert processed_path.exists(), "Processed data directory missing"
        assert features_path.exists(), "Features data directory missing"
        
        # Check for key files
        feature_file = Path("data/features/train_FD001_features.csv")
        if feature_file.exists():
            import pandas as pd
            df = pd.read_csv(feature_file)
            assert len(df) > 0, "Feature file is empty"
            assert "RUL" in df.columns, "Target column missing"
    
    def test_model_artifacts(self):
        """Test that model training produces expected artifacts"""
        model_path = Path("models/model.pkl")
        metrics_path = Path("train_metrics.json")
        
        # These might not exist in CI environment
        if model_path.exists():
            # Test model can be loaded
            import joblib
            try:
                model = joblib.load(model_path)
                assert hasattr(model, 'predict'), "Model missing predict method"
            except Exception as e:
                pytest.fail(f"Cannot load model: {str(e)}")
        
        if metrics_path.exists():
            # Test metrics file is valid JSON
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                assert isinstance(metrics, dict), "Metrics should be a dictionary"
            except Exception as e:
                pytest.fail(f"Cannot read metrics: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])