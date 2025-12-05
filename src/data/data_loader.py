"""
Data loading utilities.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from logger import get_logger

logger = get_logger(__name__)


def load_raw_data(filepath: str = "./data/raw/battery_degradation.csv") -> pd.DataFrame:
    """
    Load raw battery degradation data.

    Args:
        filepath: Path to the raw data file

    Returns:
        DataFrame with raw data
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw data from {filepath}: {df.shape}")

    return df


def load_processed_data(
    train_path: str = "./data/processed/train.csv",
    val_path: str = "./data/processed/val.csv",
    test_path: str = "./data/processed/test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed train, validation, and test sets.

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded train data: {train_df.shape}")
    logger.info(f"Loaded validation data: {val_df.shape}")
    logger.info(f"Loaded test data: {test_df.shape}")

    return train_df, val_df, test_df


def save_data(df: pd.DataFrame, filepath: str, create_dirs: bool = True) -> None:
    """
    Save dataframe to CSV.

    Args:
        df: DataFrame to save
        filepath: Path to save the file
        create_dirs: Whether to create parent directories if they don't exist
    """
    if create_dirs:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}: {df.shape}")
