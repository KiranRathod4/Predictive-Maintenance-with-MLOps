"""
Predictive Maintenance API with Monitoring
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from starlette.responses import Response
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear any existing metrics to prevent duplicates during reload
def clear_metrics():
    """Clear existing metrics to prevent duplicate registration during reload"""
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)

# Initialize metrics only once
clear_metrics()

# Prometheus metrics definitions
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions made'
)

PREDICTION_ERRORS = Counter(
    'prediction_errors_total',
    'Total number of prediction errors'
)

PREDICTION_VALUE = Histogram(
    'prediction_value',
    'Distribution of prediction values',
    buckets=[0, 50, 100, 150, 200, 250, 300, 400, 500]
)

MODEL_LOAD_TIME = Gauge(
    'model_load_seconds',
    'Time taken to load the model'
)

MODEL_INFO = Gauge(
    'model_info',
    'Model information',
    ['model_type', 'version']
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests'
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Predictive Maintenance API with monitoring...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup!")
    else:
        logger.info("API startup completed successfully")
    
    yield
    
    # Shutdown (if needed)
    logger.info("API shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="REST API for predicting Remaining Useful Life (RUL) with monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ✅ Prometheus FastAPI Instrumentation (added as requested)
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    ACTIVE_REQUESTS.dec()
    
    return response

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_metadata = {}

# Paths
MODEL_PATH = Path("models/model.pkl")
METADATA_PATH = Path("train_metrics.json")

class EngineData(BaseModel):
    """
    Request model for engine sensor data - UPDATED to match your actual data schema
    """
    engine_id: int
    cycle: int
    sensor_2: float 
    sensor_3: float 
    sensor_4: float 
    sensor_5: float 
    sensor_6: float 
    sensor_7: float 
    sensor_8: float 
    sensor_9: float 
    sensor_10: float 
    sensor_11: float 
    sensor_12: float 
    sensor_13: float 
    sensor_14: float 
    sensor_15: float 
    sensor_16: float 
    sensor_17: float 
    sensor_18: float 
    sensor_19: float 
    sensor_20: float 
    sensor_21: float 
    sensor_22: float 
    sensor_23: float 
    sensor_24: float 
    sensor_25: float 
    max_cycle: int
    
    class Config:
        validate_by_name = True
        populate_by_name = True

class PredictionResponse(BaseModel):
    rul_prediction: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_version: str
    prediction_timestamp: str
    engine_id: Optional[int] = None
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    api_version: str
    timestamp: str

def load_model():
    global model, model_metadata
    
    start_time = time.time()
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.set(load_time)
            
            MODEL_INFO.labels(
                model_type=type(model).__name__,
                version="v1.0"
            ).set(1)
            
            logger.info(f"Model loaded successfully in {load_time:.2f}s from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
            
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Model metadata loaded: RMSE={model_metadata.get('val_rmse', 'N/A')}")
        else:
            logger.warning(f"Metadata file not found: {METADATA_PATH}")
            model_metadata = {"val_rmse": "unknown", "val_r2": "unknown"}
            
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Predictive Maintenance API with Monitoring",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version="v1.0",
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": str(type(model).__name__),
        "model_params": model.get_params() if hasattr(model, 'get_params') else {},
        "feature_count": getattr(model, 'n_features_in_', 'unknown'),
        "training_metrics": model_metadata,
        "model_path": str(MODEL_PATH),
        "last_loaded": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(engine_data: EngineData):
    if model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server health.")
    
    try:
        data_dict = engine_data.model_dump()
        input_df = pd.DataFrame([data_dict])
        rename_map = {f"sensor_{i}": str(i) for i in range(2, 26)}
        input_df.rename(columns=rename_map, inplace=True)
        expected_features = model.feature_names_in_
        input_df = input_df[expected_features]
        prediction = model.predict(input_df)[0]
        PREDICTION_COUNT.inc()
        PREDICTION_VALUE.observe(prediction)
        logger.info(f"Prediction made: RUL={prediction:.2f}, Engine={engine_data.engine_id}")
        
        return PredictionResponse(
            rul_prediction=float(prediction),
            confidence_interval=None,
            model_version="v1.0",
            prediction_timestamp=datetime.now().isoformat(),
            engine_id=engine_data.engine_id,
            status="success"
        )
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(engine_data_list: List[EngineData]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(engine_data_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        predictions = []
        for engine_data in engine_data_list:
            data_dict = engine_data.model_dump()
            input_df = pd.DataFrame([data_dict])
            rename_map = {f"sensor_{i}": str(i) for i in range(2, 26)}
            input_df.rename(columns=rename_map, inplace=True)
            expected_features = model.feature_names_in_
            input_df = input_df[expected_features]
            prediction = model.predict(input_df)[0]
            PREDICTION_COUNT.inc()
            PREDICTION_VALUE.observe(prediction)
            
            predictions.append({
                "rul_prediction": float(prediction),
                "engine_id": engine_data.engine_id,
                "status": "success"
            })
        
        logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        
        return {
            "predictions": predictions,
            "batch_size": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    try:
        success = load_model()
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            PREDICTION_ERRORS.inc()
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
