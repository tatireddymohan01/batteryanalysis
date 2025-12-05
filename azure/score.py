"""
Scoring script for Azure ML deployment.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


def init():
    """
    Initialize the model and preprocessors.
    Called once when the deployment is created or updated.
    """
    global model, preprocessor, feature_engineer

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "best_model.pkl")
    preprocessor_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "preprocessor.pkl")
    feature_engineer_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "feature_engineer.pkl"
    )

    # Load model
    model_info = joblib.load(model_path)
    model = model_info["model"]

    # Load preprocessor
    preprocessor_state = joblib.load(preprocessor_path)

    # Load feature engineer
    feature_engineer_state = joblib.load(feature_engineer_path)

    # Store states
    preprocessor = preprocessor_state
    feature_engineer = feature_engineer_state


def run(raw_data):
    """
    Make predictions on input data.

    Args:
        raw_data: JSON string with input data

    Returns:
        JSON string with predictions
    """
    try:
        # Parse input
        data = json.loads(raw_data)

        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])

        # Preprocess (simplified - assuming data is already preprocessed)
        # In production, you'd apply the same preprocessing pipeline
        X = df

        # Apply scaling if scaler exists
        if "scaler" in preprocessor and preprocessor["scaler"] is not None:
            feature_names = preprocessor["feature_names"]
            X[feature_names] = preprocessor["scaler"].transform(X[feature_names])

        # Apply feature engineering (simplified)
        # Add derived features
        if "cycles" in X.columns and "usage_hours" in X.columns:
            X["cycle_rate"] = X["cycles"] / (X["usage_hours"] + 1)

        if "temperature" in X.columns and "cycles" in X.columns:
            X["temperature_stress"] = np.abs(X["temperature"] - 25) * X["cycles"] / 1000

        if "voltage_max" in X.columns and "voltage_min" in X.columns:
            X["voltage_range"] = X["voltage_max"] - X["voltage_min"]

        if "c_rate" in X.columns and "cycles" in X.columns:
            X["c_rate_stress"] = X["c_rate"] * X["cycles"] / 1000

        # Ensure feature order
        if feature_engineer["feature_names"]:
            missing_cols = set(feature_engineer["feature_names"]) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[feature_engineer["feature_names"]]

        # Predict
        predictions = model.predict(X)
        predictions = np.clip(predictions, 60, 100)

        # Format output
        results = []
        for pred in predictions:
            results.append(
                {
                    "soh": float(pred),
                    "degradation_stage": get_degradation_stage(pred),
                    "recommendation": get_recommendation(pred),
                }
            )

        return json.dumps(results)

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})


def get_degradation_stage(soh: float) -> str:
    """Determine degradation stage."""
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
    """Get recommendation."""
    if soh >= 95:
        return "Battery in excellent condition. Continue normal usage."
    elif soh >= 90:
        return "Battery in very good condition. No action required."
    elif soh >= 85:
        return "Battery in good condition. Continue monitoring."
    elif soh >= 80:
        return "Battery shows signs of aging. Monitor closely."
    elif soh >= 70:
        return "Battery is degraded. Consider replacement soon."
    else:
        return "Battery is critically degraded. Immediate replacement recommended."
