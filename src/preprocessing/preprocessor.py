"""
Data preprocessing pipeline for battery degradation data.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from logger import get_logger
from utils import load_model, save_model

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocess battery degradation data."""

    def __init__(self, config: dict = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        if config is None:
            self.config = get_config().get_preprocessing_config()
        else:
            self.config = config

        self.scaler = None
        self.imputer = None
        self.feature_names = None

        logger.info("DataPreprocessor initialized")

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input dataframe
            strategy: Strategy for handling missing values ('drop' or 'impute')

        Returns:
            DataFrame with missing values handled
        """
        if strategy is None:
            strategy = self.config.get("handle_missing", "drop")

        missing_count = df.isnull().sum().sum()

        if missing_count == 0:
            logger.info("No missing values found")
            return df

        logger.info(f"Found {missing_count} missing values")

        if strategy == "drop":
            df_clean = df.dropna()
            logger.info(
                f"Dropped rows with missing values. New shape: {df_clean.shape}"
            )
            return df_clean

        elif strategy == "impute":
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy="median")
                df_imputed = pd.DataFrame(
                    self.imputer.fit_transform(df), columns=df.columns
                )
            else:
                df_imputed = pd.DataFrame(
                    self.imputer.transform(df), columns=df.columns
                )

            logger.info(f"Imputed missing values using median strategy")
            return df_imputed

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def remove_outliers(
        self, df: pd.DataFrame, method: str = None, threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Args:
            df: Input dataframe
            method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        if method is None:
            method = self.config.get("outlier_method", "iqr")
        if threshold is None:
            threshold = self.config.get("outlier_threshold", 3.0)

        initial_shape = df.shape

        if method == "iqr":
            # Use Interquartile Range method
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Filter outliers
            mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
            df_clean = df[mask]

        elif method == "zscore":
            # Use Z-score method
            z_scores = np.abs((df - df.mean()) / df.std())
            mask = (z_scores < threshold).all(axis=1)
            df_clean = df[mask]

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df)
            df_clean = df[outlier_labels == 1]

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        removed_count = initial_shape[0] - df_clean.shape[0]
        logger.info(
            f"Removed {removed_count} outliers using {method} method. "
            f"New shape: {df_clean.shape}"
        )

        return df_clean

    def scale_features(
        self, X: pd.DataFrame, method: str = None, fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using specified method.

        Args:
            X: Feature dataframe
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler (True for training, False for test)

        Returns:
            Scaled feature dataframe
        """
        if method is None:
            method = self.config.get("scaling", {}).get("method", "standard")

        # Get features to scale
        features_to_scale = self.config.get("scaling", {}).get(
            "features_to_scale", X.columns.tolist()
        )
        features_to_scale = [f for f in features_to_scale if f in X.columns]

        if not features_to_scale:
            logger.warning("No features to scale found")
            return X

        X_scaled = X.copy()

        if fit:
            # Initialize scaler
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            elif method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            # Fit and transform
            X_scaled[features_to_scale] = self.scaler.fit_transform(
                X[features_to_scale]
            )
            logger.info(f"Fitted {method} scaler and transformed features")

        else:
            # Only transform
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")

            X_scaled[features_to_scale] = self.scaler.transform(X[features_to_scale])
            logger.info(f"Transformed features using fitted {method} scaler")

        return X_scaled

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and ranges.

        Args:
            df: Input dataframe

        Returns:
            True if validation passes, raises ValueError otherwise
        """
        # Check for required columns
        required_columns = [
            "cycles",
            "temperature",
            "c_rate",
            "voltage_min",
            "voltage_max",
            "usage_hours",
            "humidity",
        ]

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check data types
        numeric_columns = required_columns
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

        # Check value ranges
        if (df["cycles"] < 0).any():
            raise ValueError("Cycles cannot be negative")

        if (df["temperature"] < -50).any() or (df["temperature"] > 100).any():
            raise ValueError("Temperature out of valid range (-50 to 100°C)")

        if (df["c_rate"] < 0).any() or (df["c_rate"] > 10).any():
            raise ValueError("C-rate out of valid range (0 to 10)")

        if (df["voltage_min"] < 0).any() or (df["voltage_min"] > 5).any():
            raise ValueError("Voltage_min out of valid range (0 to 5V)")

        if (df["voltage_max"] < 0).any() or (df["voltage_max"] > 5).any():
            raise ValueError("Voltage_max out of valid range (0 to 5V)")

        if (df["voltage_min"] >= df["voltage_max"]).any():
            raise ValueError("Voltage_min must be less than voltage_max")

        if (df["usage_hours"] < 0).any():
            raise ValueError("Usage hours cannot be negative")

        if (df["humidity"] < 0).any() or (df["humidity"] > 100).any():
            raise ValueError("Humidity out of valid range (0 to 100%)")

        logger.info("Data validation passed")
        return True

    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str = "soh",
        fit: bool = True,
        remove_outliers: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Input dataframe
            target_column: Name of target column
            fit: Whether to fit scalers (True for training, False for test)
            remove_outliers: Whether to remove outliers

        Returns:
            Tuple of (preprocessed features, target) or (preprocessed features, None)
        """
        logger.info(f"Starting preprocessing pipeline. Input shape: {df.shape}")

        # Separate features and target
        if target_column in df.columns:
            y = df[target_column].copy()
            X = df.drop(columns=[target_column])
        else:
            y = None
            X = df.copy()

        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()

        # Validate data
        self.validate_data(X)

        # Handle missing values
        X = self.handle_missing_values(X)

        if y is not None and len(X) != len(y):
            y = y.loc[X.index]

        # Remove outliers (only during training)
        if fit and remove_outliers:
            combined = X.copy()
            if y is not None:
                combined[target_column] = y

            combined = self.remove_outliers(combined)

            if target_column in combined.columns:
                y = combined[target_column]
                X = combined.drop(columns=[target_column])
            else:
                X = combined

        # Scale features
        X_scaled = self.scale_features(X, fit=fit)

        logger.info(f"Preprocessing complete. Output shape: {X_scaled.shape}")

        return X_scaled, y

    def save_preprocessor(self, filepath: str = "./models/saved/preprocessor.pkl"):
        """
        Save preprocessor state.

        Args:
            filepath: Path to save the preprocessor
        """
        preprocessor_state = {
            "scaler": self.scaler,
            "imputer": self.imputer,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        save_model(preprocessor_state, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath: str = "./models/saved/preprocessor.pkl"):
        """
        Load preprocessor state.

        Args:
            filepath: Path to load the preprocessor from
        """
        preprocessor_state = load_model(filepath)
        self.scaler = preprocessor_state["scaler"]
        self.imputer = preprocessor_state["imputer"]
        self.feature_names = preprocessor_state["feature_names"]
        self.config = preprocessor_state["config"]
        logger.info(f"Preprocessor loaded from {filepath}")


def main():
    """Main function to demonstrate preprocessing."""
    from data.data_loader import load_processed_data

    logger.info("Starting data preprocessing")

    # Load data
    train_df, val_df, test_df = load_processed_data()

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Preprocess training data
    X_train, y_train = preprocessor.preprocess(train_df, fit=True)

    # Preprocess validation and test data
    X_val, y_val = preprocessor.preprocess(val_df, fit=False, remove_outliers=False)
    X_test, y_test = preprocessor.preprocess(test_df, fit=False, remove_outliers=False)

    # Save preprocessor
    preprocessor.save_preprocessor()

    # Save preprocessed data
    X_train["soh"] = y_train
    X_val["soh"] = y_val
    X_test["soh"] = y_test

    X_train.to_csv("./data/processed/train_preprocessed.csv", index=False)
    X_val.to_csv("./data/processed/val_preprocessed.csv", index=False)
    X_test.to_csv("./data/processed/test_preprocessed.csv", index=False)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print("=" * 60 + "\n")

    logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    main()
