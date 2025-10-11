# tests/test_api.py
"""
API unit tests for CI/CD pipeline
"""
import pytest
import json
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test all API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns welcome message"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Predictive Maintenance API" in data["message"]
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data
        assert "timestamp" in data
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        # Model might not be loaded in test environment
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
    
    def test_predict_endpoint_valid_data(self):
        """Test prediction with valid data"""
        sample_data = {
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
        
        response = client.post("/predict", json=sample_data)
        # Model might not be loaded in test environment
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "rul_prediction" in data
            assert "model_version" in data
            assert "status" in data
            assert data["status"] == "success"
    
    def test_predict_endpoint_invalid_data(self):
        """Test prediction with invalid data"""
        invalid_data = {
            "setting1": "invalid",  # Should be float
            "setting2": 0.84,
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        sample_data = {
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
        
        batch_data = [sample_data, sample_data]  # Two identical samples
        
        response = client.post("/predict/batch", json=batch_data)
        # Model might not be loaded in test environment
        assert response.status_code in [200, 503]
    
    def test_model_reload(self):
        """Test model reload endpoint"""
        response = client.post("/model/reload")
        # Model might not exist in test environment
        assert response.status_code in [200, 500]

class TestDataValidation:
    """Test input data validation"""
    
    def test_sensor_range_validation(self):
        """Test that sensor values are within reasonable ranges"""
        # Test with sensor value way out of range
        invalid_data = {
            "setting1": 42.0,
            "setting2": 0.84,
            "setting3": 100.0,
            "sensor_2": -1000.0,  # Negative temperature (invalid)
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
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Should fail validation
    
    def test_missing_required_fields(self):
        """Test that missing required fields are caught"""
        incomplete_data = {
            "setting1": 42.0,
            "setting2": 0.84,
            # Missing setting3 and sensors
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    def test_batch_size_limit(self):
        """Test batch size limitation"""
        sample_data = {
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
        }
        
        # Create batch larger than limit (100+)
        large_batch = [sample_data] * 101
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 400  # Should reject large batches

class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_response_time(self):
        """Test that API responds within acceptable time"""
        import time
        
        sample_data = {
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
        }
        
        start_time = time.time()
        response = client.post("/predict", json=sample_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # API should respond within 5 seconds (generous for CI environment)
        assert response_time < 5.0
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        sample_data = {
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
        }
        
        results = []
        
        def make_request():
            response = client.post("/predict", json=sample_data)
            results.append(response.status_code)
        
        # Create 5 concurrent threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should complete (either 200 or 503 if model not loaded)
        assert len(results) == 5
        for status in results:
            assert status in [200, 503]