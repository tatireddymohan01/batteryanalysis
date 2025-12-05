"""
FastAPI application for battery SOH prediction.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from logger import setup_logger
from utils import load_model

# Setup logger
logger = setup_logger(__name__, log_file="./logs/api.log")

# Initialize FastAPI app
app = FastAPI(
    title="Battery Degradation Prediction API",
    description="Predict battery State of Health (SOH) based on operational parameters",
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

# Load configuration
config = get_config()
api_config = config.get_api_config()

# Global variables for model and preprocessors
model = None
preprocessor = None
feature_engineer = None


class BatteryInput(BaseModel):
    """Input schema for battery prediction."""
    
    cycles: int = Field(..., description="Number of charge/discharge cycles", ge=0, le=5000)
    temperature: float = Field(..., description="Operating temperature in °C", ge=-20, le=80)
    c_rate: float = Field(..., description="Charging/discharging rate", ge=0.05, le=5.0)
    voltage_min: float = Field(..., description="Minimum voltage (V)", ge=2.5, le=3.5)
    voltage_max: float = Field(..., description="Maximum voltage (V)", ge=3.8, le=4.5)
    usage_hours: float = Field(..., description="Total usage hours", ge=0, le=50000)
    humidity: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    
    @validator('voltage_max')
    def validate_voltage_range(cls, v, values):
        """Ensure voltage_max > voltage_min."""
        if 'voltage_min' in values and v <= values['voltage_min']:
            raise ValueError('voltage_max must be greater than voltage_min')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "cycles": 500,
                "temperature": 25.0,
                "c_rate": 1.0,
                "voltage_min": 3.0,
                "voltage_max": 4.2,
                "usage_hours": 1200,
                "humidity": 50.0
            }
        }


class BatteryBatchInput(BaseModel):
    """Batch input schema for multiple predictions."""
    
    predictions: List[BatteryInput]


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    
    soh: float = Field(..., description="Predicted State of Health (%)")
    confidence_interval: Optional[Dict[str, float]] = None
    degradation_stage: str = Field(..., description="Battery degradation stage")
    recommendation: str = Field(..., description="Recommendation based on SOH")
    
    class Config:
        schema_extra = {
            "example": {
                "soh": 87.5,
                "confidence_interval": {"lower": 85.2, "upper": 89.8},
                "degradation_stage": "Good",
                "recommendation": "Battery in good condition. Continue normal usage."
            }
        }


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    
    predictions: List[PredictionOutput]
    total_count: int


def load_models():
    """Load trained model and preprocessors."""
    global model, preprocessor, feature_engineer
    
    try:
        # Load model
        model_path = api_config.get('model_path', './models/saved/best_model.pkl')
        model_info = load_model(model_path)
        model = model_info['model']
        logger.info(f"Loaded model: {model_info.get('model_name', 'unknown')}")
        
        # Load preprocessor
        preprocessor_path = "./models/saved/preprocessor.pkl"
        preprocessor_state = load_model(preprocessor_path)
        
        # Create preprocessor instance
        from preprocessing.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.scaler = preprocessor_state['scaler']
        preprocessor.imputer = preprocessor_state['imputer']
        preprocessor.feature_names = preprocessor_state['feature_names']
        logger.info("Loaded preprocessor")
        
        # Load feature engineer
        feature_engineer_path = "./models/saved/feature_engineer.pkl"
        feature_engineer_state = load_model(feature_engineer_path)
        
        from features.feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()
        feature_engineer.polynomial_features = feature_engineer_state['polynomial_features']
        feature_engineer.feature_names = feature_engineer_state['feature_names']
        logger.info("Loaded feature engineer")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def get_degradation_stage(soh: float) -> str:
    """
    Determine degradation stage based on SOH.
    
    Args:
        soh: State of Health percentage
        
    Returns:
        Degradation stage description
    """
    if soh >= 95:
        return "Excellent"
    elif soh >= 90:
        return "Very Good"
    elif soh >= 85:
        return "Good"
    elif soh >= 80:
        return "Fair"
    elif soh >= 70:
        return "Degraded"
    else:
        return "Critical"


def get_recommendation(soh: float) -> str:
    """
    Get recommendation based on SOH.
    
    Args:
        soh: State of Health percentage
        
    Returns:
        Recommendation string
    """
    if soh >= 95:
        return "Battery in excellent condition. Continue normal usage."
    elif soh >= 90:
        return "Battery in very good condition. No action required."
    elif soh >= 85:
        return "Battery in good condition. Continue monitoring."
    elif soh >= 80:
        return "Battery shows signs of aging. Monitor closely and plan for replacement."
    elif soh >= 70:
        return "Battery is degraded. Consider replacement soon to avoid performance issues."
    else:
        return "Battery is critically degraded. Immediate replacement recommended."


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting API server")
    load_models()
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Mounted static files from {static_dir}")
    
    logger.info("API server ready")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - serves the UI."""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(static_file)
    return {
        "message": "Battery Degradation Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    preprocessor_loaded = preprocessor is not None
    feature_engineer_loaded = feature_engineer is not None
    
    return {
        "status": "healthy" if all([model_loaded, preprocessor_loaded, feature_engineer_loaded]) else "unhealthy",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "feature_engineer_loaded": feature_engineer_loaded
    }


@app.post("/predict", tags=["Prediction"])
async def predict(input_data: BatteryInput):
    """
    Predict battery State of Health (SOH).
    
    Args:
        input_data: Battery operational parameters
        
    Returns:
        Prediction with SOH, degradation stage, and recommendation
    """
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocess
        X_processed, _ = preprocessor.preprocess(df, fit=False, remove_outliers=False)
        
        # Feature engineering
        X_engineered = feature_engineer.engineer_features(X_processed, fit=False)
        
        # Ensure feature order matches training
        if feature_engineer.feature_names:
            missing_cols = set(feature_engineer.feature_names) - set(X_engineered.columns)
            for col in missing_cols:
                X_engineered[col] = 0
            X_engineered = X_engineered[feature_engineer.feature_names]
        
        # Predict
        prediction = model.predict(X_engineered)[0]
        
        # Ensure SOH is in valid range
        prediction = np.clip(prediction, 60, 100)
        
        # Get degradation stage and recommendation
        degradation_stage = get_degradation_stage(prediction)
        recommendation = get_recommendation(prediction)
        
        # Calculate confidence interval (simplified)
        # In production, you might use prediction intervals from the model
        confidence_interval = {
            "lower": max(prediction - 2.0, 60),
            "upper": min(prediction + 2.0, 100)
        }
        
        logger.info(f"Prediction: SOH={prediction:.2f}%, Stage={degradation_stage}")
        
        # Return in format expected by UI
        return {
            "predicted_soh": round(prediction, 2),
            "soh": round(prediction, 2),  # Legacy compatibility
            "confidence_interval": confidence_interval,
            "confidence": "High" if abs(confidence_interval['upper'] - confidence_interval['lower']) < 5 else "Medium",
            "degradation_stage": degradation_stage,
            "recommendation": recommendation,
            "recommendations": [recommendation]  # Array format for UI
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(batch_input: BatteryBatchInput):
    """
    Predict battery SOH for multiple inputs.
    
    Args:
        batch_input: List of battery operational parameters
        
    Returns:
        List of predictions
    """
    try:
        predictions = []
        
        for input_data in batch_input.predictions:
            result = await predict(input_data)
            predictions.append(result)
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_count=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "features": feature_engineer.feature_names if feature_engineer else None,
        "n_features": len(feature_engineer.feature_names) if feature_engineer else None
    }


if __name__ == "__main__":
    import uvicorn
    
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    reload = api_config.get('reload', False)
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload
    )
