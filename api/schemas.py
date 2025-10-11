# api/schemas.py
"""
Pydantic schemas for API request/response models
These define the structure and validation for API data
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime

class EngineDataInput(BaseModel):
    """
    Schema for individual engine sensor readings
    Field() provides additional validation and documentation
    """
    
    # Operational Settings
    setting1: float = Field(..., description="Operational setting 1", example=42.0)
    setting2: float = Field(..., description="Operational setting 2", example=0.84)
    setting3: float = Field(..., description="Operational setting 3", example=100.0)
    
    # Sensor Measurements
    sensor_2: float = Field(..., description="Temperature sensor", ge=0, le=2000)
    sensor_3: float = Field(..., description="Temperature sensor", ge=0, le=2000)
    sensor_4: float = Field(..., description="Temperature sensor", ge=0, le=2000)
    sensor_6: float = Field(..., description="Temperature sensor", ge=0, le=2000)
    sensor_7: float = Field(..., description="Pressure sensor", ge=0, le=1000)
    sensor_8: float = Field(..., description="Pressure sensor", ge=0, le=1000)
    sensor_9: float = Field(..., description="Pressure sensor", ge=0, le=1000)
    sensor_11: float = Field(..., description="Vibration sensor", ge=0, le=100)
    sensor_12: float = Field(..., description="Vibration sensor", ge=0, le=100)
    sensor_13: float = Field(..., description="Vibration sensor", ge=0, le=100)
    sensor_14: float = Field(..., description="Vibration sensor", ge=0, le=100)
    sensor_15: float = Field(..., description="Pressure sensor", ge=0, le=1000)
    sensor_17: float = Field(..., description="Vibration sensor", ge=0, le=100)
    sensor_20: float = Field(..., description="Temperature ratio", ge=0, le=2)
    sensor_21: float = Field(..., description="Temperature ratio", ge=0, le=2)
    
    # Optional metadata
    engine_id: Optional[int] = Field(None, description="Engine identifier", example=1)
    cycle: Optional[int] = Field(None, description="Engine cycle number", ge=1, example=150)
    
    @validator('sensor_*', pre=True, allow_reuse=True)
    def validate_sensor_readings(cls, v):
        """Validate that sensor readings are reasonable"""
        if v is None:
            raise ValueError("Sensor reading cannot be None")
        if not isinstance(v, (int, float)):
            raise ValueError("Sensor reading must be a number")
        return float(v)

class PredictionOutput(BaseModel):
    """
    Schema for prediction response
    """
    rul_prediction: float = Field(..., description="Predicted Remaining Useful Life", example=45.5)
    confidence_score: Optional[float] = Field(None, description="Prediction confidence (0-1)", ge=0, le=1)
    prediction_interval: Optional[Dict[str, float]] = Field(None, description="Prediction interval bounds")
    
    # Model information
    model_version: str = Field(..., description="Model version used", example="v1.0")
    model_type: str = Field(default="RandomForestRegressor", description="Model algorithm")
    
    # Request metadata
    engine_id: Optional[int] = Field(None, description="Engine ID from request")
    prediction_timestamp: str = Field(..., description="When prediction was made")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    # Status
    status: str = Field(default="success", description="Prediction status")
    warnings: Optional[List[str]] = Field(None, description="Any warnings during prediction")

class BatchPredictionInput(BaseModel):
    """
    Schema for batch prediction requests
    """
    engine_data: List[EngineDataInput] = Field(..., description="List of engine readings", max_items=100)
    batch_id: Optional[str] = Field(None, description="Batch identifier for tracking")
    
    @validator('engine_data')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 items")
        return v

class BatchPredictionOutput(BaseModel):
    """
    Schema for batch prediction response
    """
    predictions: List[PredictionOutput] = Field(..., description="Individual predictions")
    batch_summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    timestamp: str = Field(..., description="Batch completion timestamp")

class HealthCheckResponse(BaseModel):
    """
    Schema for health check response
    """
    status: str = Field(..., description="API health status", example="healthy")
    model_status: str = Field(..., description="Model loading status", example="loaded")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    api_version: str = Field(..., description="API version", example="1.0.0")
    model_version: str = Field(..., description="Model version", example="v1.0")
    uptime_seconds: Optional[float] = Field(None, description="API uptime in seconds")
    timestamp: str = Field(..., description="Health check timestamp")

class ModelInfoResponse(BaseModel):
    """
    Schema for model information response
    """
    model_type: str = Field(..., description="ML algorithm type")
    model_parameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    feature_count: int = Field(..., description="Number of input features")
    training_metrics: Dict[str, float] = Field(..., description="Training performance metrics")
    model_size_mb: Optional[float] = Field(None, description="Model file size in MB")
    last_trained: Optional[str] = Field(None, description="When model was last trained")
    last_loaded: str = Field(..., description="When model was loaded into API")

class ErrorResponse(BaseModel):
    """
    Schema for error responses
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error description")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

# Example data for API documentation
class ExampleData:
    """
    Example data for API documentation and testing
    """
    
    @staticmethod
    def sample_engine_data():
        return {
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
    
    @staticmethod
    def sample_prediction_response():
        return {
            "rul_prediction": 45.5,
            "confidence_score": 0.85,
            "prediction_interval": {"lower": 40.2, "upper": 50.8},
            "model_version": "v1.0",
            "model_type": "RandomForestRegressor",
            "engine_id": 1,
            "prediction_timestamp": "2025-09-18T22:30:00",
            "processing_time_ms": 125.5,
            "status": "success",
            "warnings": None
        }