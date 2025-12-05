"""
Utility functions for the battery degradation prediction project.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model: Any, filepath: str) -> None:
    """
    Save a model to disk using joblib.

    Args:
        model: Model object to save
        filepath: Path to save the model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a model from disk.

    Args:
        filepath: Path to the model file

    Returns:
        Loaded model object
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Loaded dictionary
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def calculate_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate percentage error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Array of percentage errors
    """
    return np.abs((y_true - y_pred) / y_true) * 100


def get_feature_names(include_derived: bool = True) -> List[str]:
    """
    Get list of feature names.

    Args:
        include_derived: Whether to include derived features

    Returns:
        List of feature names
    """
    base_features = [
        "cycles",
        "temperature",
        "c_rate",
        "voltage_min",
        "voltage_max",
        "usage_hours",
        "humidity",
    ]

    if include_derived:
        derived_features = [
            "cycle_rate",
            "temperature_stress",
            "voltage_range",
            "c_rate_stress",
            "cumulative_stress",
            "temperature_humidity_interaction",
        ]
        return base_features + derived_features

    return base_features


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """
    Print formatted metrics summary.

    Args:
        metrics: Dictionary of metric names and values
    """
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():20s}: {value:.4f}")
    print("=" * 50 + "\n")


def split_features_target(
    df: pd.DataFrame, target_column: str = "soh"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Args:
        df: Input dataframe
        target_column: Name of target column

    Returns:
        Tuple of (features_df, target_series)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    # Assuming this file is in src/utils.py
    return Path(__file__).parent.parent


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
