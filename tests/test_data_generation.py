"""Tests for data generation module."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generate_data import BatteryDegradationGenerator


def test_generator_initialization():
    """Test generator initialization."""
    generator = BatteryDegradationGenerator()
    assert generator is not None
    assert generator.random_state == 42


def test_dataset_generation():
    """Test dataset generation."""
    generator = BatteryDegradationGenerator()
    df = generator.generate_dataset(n_samples=100)
    
    # Check shape
    assert len(df) == 100
    assert len(df.columns) == 8  # 7 features + 1 target
    
    # Check columns
    expected_columns = ['cycles', 'temperature', 'c_rate', 'voltage_min', 
                       'voltage_max', 'usage_hours', 'humidity', 'soh']
    assert list(df.columns) == expected_columns
    
    # Check SOH range
    assert df['soh'].min() >= 60
    assert df['soh'].max() <= 100


def test_train_test_split():
    """Test train/val/test split."""
    generator = BatteryDegradationGenerator()
    df = generator.generate_dataset(n_samples=100)
    
    train_df, val_df, test_df = generator.generate_train_test_split(
        df, test_size=0.2, validation_size=0.1
    )
    
    # Check sizes
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(test_df) == 20
    assert len(val_df) == 10
    assert len(train_df) == 70
