"""
Synthetic Battery Degradation Dataset Generator

Generates realistic battery degradation data based on physical degradation models.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from logger import setup_logger
from utils import ensure_dir

# Set up logger
logger = setup_logger(__name__, log_file="./logs/data_generation.log")


class BatteryDegradationGenerator:
    """Generate synthetic battery degradation data."""

    def __init__(self, config: Dict = None):
        """
        Initialize the generator.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        if config is None:
            self.config = get_config().get_data_generation_config()
        else:
            self.config = config

        self.random_state = self.config.get("random_state", 42)
        np.random.seed(self.random_state)

        logger.info("BatteryDegradationGenerator initialized")

    def generate_feature(self, n_samples: int, feature_config: Dict) -> np.ndarray:
        """
        Generate feature values based on configuration.

        Args:
            n_samples: Number of samples
            feature_config: Feature configuration dictionary

        Returns:
            Array of feature values
        """
        distribution = feature_config.get("distribution", "uniform")

        if distribution == "uniform":
            min_val = feature_config.get("min", 0)
            max_val = feature_config.get("max", 1)
            return np.random.uniform(min_val, max_val, n_samples)

        elif distribution == "normal":
            mean = feature_config.get("mean", 0)
            std = feature_config.get("std", 1)
            values = np.random.normal(mean, std, n_samples)

            # Clip to min/max if specified
            if "min" in feature_config:
                values = np.maximum(values, feature_config["min"])
            if "max" in feature_config:
                values = np.minimum(values, feature_config["max"])

            return values

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def calculate_soh(
        self,
        cycles: np.ndarray,
        temperature: np.ndarray,
        c_rate: np.ndarray,
        voltage_min: np.ndarray,
        voltage_max: np.ndarray,
        humidity: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate State of Health (SOH) based on degradation model.

        Physical degradation model considers:
        - Cycle-induced degradation (capacity fade per cycle)
        - Temperature-accelerated degradation (Arrhenius-like)
        - C-rate stress (higher current = faster degradation)
        - Voltage stress (high voltage = faster degradation)
        - Environmental factors (humidity effects)

        Args:
            cycles: Charge/discharge cycles
            temperature: Operating temperature (°C)
            c_rate: Charging/discharging rate
            voltage_min: Minimum voltage
            voltage_max: Maximum voltage
            humidity: Relative humidity (%)

        Returns:
            Array of SOH values (percentage of original capacity)
        """
        deg_config = self.config.get("degradation", {})

        # Base degradation from cycling
        base_rate = deg_config.get("base_degradation_rate", 0.02)
        cycle_degradation = base_rate * (cycles / 100)

        # Temperature-accelerated degradation
        temp_coef = deg_config.get("temperature_coefficient", 0.002)
        temp_stress = np.maximum(temperature - 25, 0)  # Only above 25°C
        temp_degradation = temp_coef * temp_stress * (cycles / 100)

        # C-rate induced stress
        c_rate_coef = deg_config.get("c_rate_coefficient", 0.01)
        c_rate_degradation = c_rate_coef * (c_rate - 1) * (cycles / 100)
        c_rate_degradation = np.maximum(c_rate_degradation, 0)

        # Voltage stress (higher voltage = faster degradation)
        voltage_coef = deg_config.get("voltage_stress_coefficient", 0.005)
        voltage_stress = np.maximum(voltage_max - 4.1, 0)  # Stress above 4.1V
        voltage_degradation = voltage_coef * voltage_stress * (cycles / 100)

        # Humidity effects (extreme humidity accelerates degradation)
        humidity_coef = deg_config.get("humidity_coefficient", 0.0005)
        humidity_stress = np.abs(humidity - 50) / 50  # Normalized distance from ideal
        humidity_degradation = humidity_coef * humidity_stress * (cycles / 100)

        # Total degradation
        total_degradation = (
            cycle_degradation
            + temp_degradation
            + c_rate_degradation
            + voltage_degradation
            + humidity_degradation
        )

        # SOH = 100% - degradation, with some random noise
        noise = np.random.normal(0, 0.5, len(cycles))
        soh = 100 - total_degradation + noise

        # Ensure SOH is in valid range
        soh = np.clip(soh, 60, 100)  # Battery typically replaced at 60-80% SOH

        return soh

    def generate_dataset(
        self, n_samples: int = None, save_path: str = None
    ) -> pd.DataFrame:
        """
        Generate complete synthetic battery degradation dataset.

        Args:
            n_samples: Number of samples to generate. If None, uses config value.
            save_path: Path to save the dataset. If None, uses default location.

        Returns:
            DataFrame with generated data
        """
        if n_samples is None:
            n_samples = self.config.get("n_samples", 10000)

        logger.info(f"Generating {n_samples} battery degradation samples")

        features_config = self.config.get("features", {})

        # Generate base features
        cycles = self.generate_feature(n_samples, features_config["cycles"])
        temperature = self.generate_feature(n_samples, features_config["temperature"])
        c_rate = self.generate_feature(n_samples, features_config["c_rate"])

        # Voltage features with some noise
        voltage_min_config = features_config["voltage_min"]
        voltage_min = voltage_min_config["value"] + np.random.normal(
            0, voltage_min_config["noise"], n_samples
        )

        voltage_max_config = features_config["voltage_max"]
        voltage_max = voltage_max_config["value"] + np.random.normal(
            0, voltage_max_config["noise"], n_samples
        )

        # Usage hours (correlated with cycles)
        usage_hours_config = features_config["usage_hours"]
        correlation = usage_hours_config["correlation_with_cycles"]
        base_hours = usage_hours_config["base_hours_per_cycle"]
        noise_std = usage_hours_config["noise"]

        usage_hours = cycles * base_hours + np.random.normal(
            0, noise_std * cycles * base_hours, n_samples
        )
        usage_hours = np.maximum(usage_hours, 0)

        # Humidity
        humidity = self.generate_feature(n_samples, features_config["humidity"])
        humidity = np.clip(humidity, 0, 100)

        # Calculate SOH (target variable)
        soh = self.calculate_soh(
            cycles, temperature, c_rate, voltage_min, voltage_max, humidity
        )

        # Create DataFrame
        df = pd.DataFrame(
            {
                "cycles": cycles.astype(int),
                "temperature": np.round(temperature, 2),
                "c_rate": np.round(c_rate, 2),
                "voltage_min": np.round(voltage_min, 3),
                "voltage_max": np.round(voltage_max, 3),
                "usage_hours": np.round(usage_hours, 1),
                "humidity": np.round(humidity, 1),
                "soh": np.round(soh, 2),
            }
        )

        logger.info(f"Dataset generated with shape: {df.shape}")
        logger.info(f"SOH statistics:\n{df['soh'].describe()}")

        # Save dataset
        if save_path is None:
            save_path = "./data/raw/battery_degradation.csv"

        save_dir = Path(save_path).parent
        ensure_dir(save_dir)

        df.to_csv(save_path, index=False)
        logger.info(f"Dataset saved to {save_path}")

        return df

    def generate_train_test_split(
        self,
        df: pd.DataFrame = None,
        test_size: float = None,
        validation_size: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.

        Time-based split to prevent data leakage (assuming cycles represent time).

        Args:
            df: Input dataframe. If None, generates new dataset.
            test_size: Proportion for test set
            validation_size: Proportion for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df is None:
            df = self.generate_dataset()

        if test_size is None:
            test_size = self.config.get("test_size", 0.2)
        if validation_size is None:
            validation_size = self.config.get("validation_size", 0.1)

        # Sort by cycles for time-based split
        df_sorted = df.sort_values("cycles").reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * (1 - test_size - validation_size))
        val_end = int(n * (1 - test_size))

        train_df = df_sorted[:train_end]
        val_df = df_sorted[train_end:val_end]
        test_df = df_sorted[val_end:]

        logger.info(f"Train set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Test set size: {len(test_df)}")

        # Save splits
        processed_dir = Path("./data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(processed_dir / "train.csv", index=False)
        val_df.to_csv(processed_dir / "val.csv", index=False)
        test_df.to_csv(processed_dir / "test.csv", index=False)

        logger.info(f"Train/val/test splits saved to {processed_dir}")

        return train_df, val_df, test_df


def main():
    """Main function to generate dataset."""
    logger.info("Starting battery degradation dataset generation")

    generator = BatteryDegradationGenerator()

    # Generate dataset
    df = generator.generate_dataset()

    # Create train/val/test splits
    train_df, val_df, test_df = generator.generate_train_test_split(df)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nFeature Statistics:")
    print(df.describe())
    print("=" * 60 + "\n")

    logger.info("Dataset generation completed successfully")


if __name__ == "__main__":
    main()
