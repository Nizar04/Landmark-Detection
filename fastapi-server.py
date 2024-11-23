# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from logging.handlers import RotatingFileHandler
import uvicorn
from datetime import datetime
from pathlib import Path
import shutil
import json
from typing import List, Dict
import numpy as np

from prometheus_client import make_asgi_app, Counter, Histogram

from .models import PredictionResponse, HealthCheckResponse
from .inference import LandmarkPredictor
from .config import Config

# Initialize logging
logging.basicConfig(
    handlers=[RotatingFileHandler('app.log', maxBytes=100000, backupCount=10)],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize metrics
PREDICTION_REQUEST_COUNT = Counter(
    'landmark_prediction_requests_total',
    'Total number of prediction requests'
)
PREDICTION_LATENCY = Histogram(
    'landmark_prediction_latency_seconds',
    'Time spent processing prediction requests'
)

app = FastAPI(
    title="Landmark Detection API",
    description="API for detecting landmarks in images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize predictor
predictor = LandmarkPredictor(
    model_path=Config.MODEL_PATH,
    label_encoder=Config.LABEL_ENCODER_PATH
)

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        PREDICTION_REQUEST_COUNT.inc()
        start_time = datetime.utcnow()
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")
        
        # Save uploaded file temporarily
        temp_path = Path("temp") / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Make prediction
        with PREDICTION_LATENCY.time():
            predictions = predictor.predict(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        # Log request
        logger.info(
            f"Prediction made for {file.filename} - "
            f"top prediction: {predictions[0]['landmark_id']} "
            f"({predictions[0]['confidence']:.2f})"
        )
        
        return {
            "filename": file.filename,
            "predictions": predictions,
            "processing_time": (datetime.utcnow() - start_time).total_seconds()
        }
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# app/models.py
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime

class PredictionResponse(BaseModel):
    filename: str
    predictions: List[Dict[str, float]]
    processing_time: float

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
