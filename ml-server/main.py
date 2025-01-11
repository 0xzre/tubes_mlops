# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import mlflow
import logging
from typing import List, Dict, Optional
from prometheus_fastapi_instrumentator import Instrumentator
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLflow Model Server", version="1.0.0")

# Add prometheus metrics
Instrumentator().instrument(app).expose(app)

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    model_stage: str
    timestamp: str

# Global variables
model = None
model_info = {
    "version": "none",
    "stage": "none",
    "last_loaded": None,
    "name": os.getenv("MLFLOW_MODEL_NAME", "default_model")
}

def load_production_model():
    """Load the latest production model from MLflow"""
    global model, model_info
    try:
        # Get the latest production model
        client = mlflow.tracking.MlflowClient()
        
        # Get latest version in production stage
        filter_string = f"name='{model_info['name']}'"
        versions = client.search_model_versions(filter_string)
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if not prod_versions:
            raise Exception("No production model found")
            
        latest_version = sorted(prod_versions, key=lambda x: x.version, reverse=True)[0]
        
        # Load the model
        model_uri = f"models:/{model_info['name']}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Update model info
        model_info.update({
            "version": latest_version.version,
            "stage": "Production",
            "last_loaded": datetime.now().isoformat()
        })
        
        logger.info(f"Loaded model version {latest_version.version} from MLflow")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    load_production_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_info": model_info
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make predictions using the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(request.features)
        
        # Make prediction
        predictions = model.predict(features)
        
        # Convert numpy types to Python native types
        predictions = predictions.tolist()
        
        return PredictionResponse(
            predictions=predictions,
            model_version=model_info["version"],
            model_stage=model_info["stage"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh-model")
async def refresh_model(background_tasks: BackgroundTasks):
    """Endpoint to refresh the model (load latest production version)"""
    try:
        background_tasks.add_task(load_production_model)
        return {"status": "Model refresh initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get current model information"""
    return model_info