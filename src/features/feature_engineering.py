"""
Feature engineering for battery degradation prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from sklearn.preprocessing import PolynomialFeatures
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from logger import get_logger
from utils import save_model, load_model

logger = get_logger(__name__)


class FeatureEngineer:
    """Engineer features for battery degradation prediction."""
    
    def __init__(self, config: dict = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        if config is None:
            self.config = get_config().get_feature_engineering_config()
        else:
            self.config = config
        
        self.polynomial_features = None
        self.feature_names = None
        
        logger.info("FeatureEngineer initialized")
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features based on domain knowledge.
        
        Args:
            df: Input dataframe with base features
            
        Returns:
            DataFrame with additional derived features
        """
        df_features = df.copy()
        
        derived_features_config = self.config.get('derived_features', [])
        
        # Cycle rate (cycles per usage hour)
        if 'cycle_rate' in derived_features_config:
            df_features['cycle_rate'] = df_features['cycles'] / (df_features['usage_hours'] + 1)
            logger.info("Created feature: cycle_rate")
        
        # Temperature stress (temperature deviation from optimal * cycles)
        if 'temperature_stress' in derived_features_config:
            optimal_temp = 25  # °C
            df_features['temperature_stress'] = (
                np.abs(df_features['temperature'] - optimal_temp) * 
                df_features['cycles'] / 1000  # Normalize
            )
            logger.info("Created feature: temperature_stress")
        
        # Voltage range
        if 'voltage_range' in derived_features_config:
            df_features['voltage_range'] = (
                df_features['voltage_max'] - df_features['voltage_min']
            )
            logger.info("Created feature: voltage_range")
        
        # C-rate stress (c_rate * cycles)
        if 'c_rate_stress' in derived_features_config:
            df_features['c_rate_stress'] = (
                df_features['c_rate'] * df_features['cycles'] / 1000  # Normalize
            )
            logger.info("Created feature: c_rate_stress")
        
        # Cumulative stress indicator
        if 'cumulative_stress' in derived_features_config:
            # Weighted combination of stress factors
            temp_stress = np.abs(df_features['temperature'] - 25) / 50  # Normalized
            voltage_stress = np.maximum(df_features['voltage_max'] - 4.1, 0) / 0.3  # Normalized
            c_rate_stress = (df_features['c_rate'] - 1) / 2  # Normalized
            
            df_features['cumulative_stress'] = (
                df_features['cycles'] * 
                (0.4 * temp_stress + 0.3 * voltage_stress + 0.3 * np.maximum(c_rate_stress, 0))
            ) / 100  # Normalize
            
            logger.info("Created feature: cumulative_stress")
        
        # Temperature-humidity interaction
        if 'temperature_humidity_interaction' in derived_features_config:
            # Both high temp and high/low humidity accelerate degradation
            humidity_stress = np.abs(df_features['humidity'] - 50) / 50  # Distance from optimal
            temp_stress_norm = np.maximum(df_features['temperature'] - 25, 0) / 35  # Normalized
            
            df_features['temperature_humidity_interaction'] = (
                temp_stress_norm * humidity_stress
            )
            logger.info("Created feature: temperature_humidity_interaction")
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} derived features")
        
        return df_features
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        degree: int = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: Input dataframe
            degree: Polynomial degree
            fit: Whether to fit the transformer
            
        Returns:
            DataFrame with polynomial features
        """
        if not self.config.get('create_polynomial_features', False):
            return df
        
        if degree is None:
            degree = self.config.get('polynomial_degree', 2)
        
        logger.info(f"Creating polynomial features with degree {degree}")
        
        # Select numeric features for polynomial expansion
        # Typically we exclude already derived features to avoid explosion
        base_features = ['cycles', 'temperature', 'c_rate', 'voltage_min', 
                        'voltage_max', 'usage_hours', 'humidity']
        features_to_expand = [f for f in base_features if f in df.columns]
        
        if fit:
            self.polynomial_features = PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=self.config.get('interaction_features', False)
            )
            
            poly_features = self.polynomial_features.fit_transform(
                df[features_to_expand]
            )
        else:
            if self.polynomial_features is None:
                raise ValueError("Polynomial features not fitted. Call with fit=True first.")
            
            poly_features = self.polynomial_features.transform(
                df[features_to_expand]
            )
        
        # Create feature names
        poly_feature_names = self.polynomial_features.get_feature_names_out(
            features_to_expand
        )
        
        # Combine with original features
        df_poly = df.copy()
        
        # Add polynomial features (excluding the original features that are already in df)
        n_original = len(features_to_expand)
        for i, name in enumerate(poly_feature_names[n_original:], start=n_original):
            df_poly[name] = poly_features[:, i]
        
        logger.info(f"Added {len(poly_feature_names) - n_original} polynomial features")
        
        return df_poly
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create specific interaction features based on domain knowledge.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with interaction features
        """
        if not self.config.get('interaction_features', False):
            return df
        
        df_interact = df.copy()
        
        # Temperature x C-rate (both stress factors)
        if 'temperature' in df.columns and 'c_rate' in df.columns:
            df_interact['temp_crate_interaction'] = (
                df['temperature'] * df['c_rate']
            )
        
        # Cycles x Temperature (age + heat stress)
        if 'cycles' in df.columns and 'temperature' in df.columns:
            df_interact['cycles_temp_interaction'] = (
                df['cycles'] * df['temperature']
            ) / 1000  # Normalize
        
        # Voltage_max x C-rate (charging stress)
        if 'voltage_max' in df.columns and 'c_rate' in df.columns:
            df_interact['voltage_crate_interaction'] = (
                df['voltage_max'] * df['c_rate']
            )
        
        logger.info("Created interaction features")
        
        return df_interact
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input dataframe
            fit: Whether to fit transformers
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering. Input shape: {df.shape}")
        
        # Create derived features
        df_engineered = self.create_derived_features(df)
        
        # Create interaction features
        if self.config.get('interaction_features', False):
            df_engineered = self.create_interaction_features(df_engineered)
        
        # Create polynomial features (if enabled)
        # Note: This can significantly increase dimensionality
        if self.config.get('create_polynomial_features', False):
            df_engineered = self.create_polynomial_features(df_engineered, fit=fit)
        
        # Store feature names
        if fit:
            self.feature_names = df_engineered.columns.tolist()
        
        logger.info(f"Feature engineering complete. Output shape: {df_engineered.shape}")
        logger.info(f"Feature names: {df_engineered.columns.tolist()}")
        
        return df_engineered
    
    def save_feature_engineer(self, filepath: str = "./models/saved/feature_engineer.pkl"):
        """
        Save feature engineer state.
        
        Args:
            filepath: Path to save the feature engineer
        """
        feature_engineer_state = {
            'polynomial_features': self.polynomial_features,
            'feature_names': self.feature_names,
            'config': self.config
        }
        save_model(feature_engineer_state, filepath)
        logger.info(f"FeatureEngineer saved to {filepath}")
    
    def load_feature_engineer(self, filepath: str = "./models/saved/feature_engineer.pkl"):
        """
        Load feature engineer state.
        
        Args:
            filepath: Path to load the feature engineer from
        """
        feature_engineer_state = load_model(filepath)
        self.polynomial_features = feature_engineer_state['polynomial_features']
        self.feature_names = feature_engineer_state['feature_names']
        self.config = feature_engineer_state['config']
        logger.info(f"FeatureEngineer loaded from {filepath}")


def main():
    """Main function to demonstrate feature engineering."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.data_loader import load_processed_data
    
    logger.info("Starting feature engineering")
    
    # Load preprocessed data
    train_df = pd.read_csv("./data/processed/train_preprocessed.csv")
    val_df = pd.read_csv("./data/processed/val_preprocessed.csv")
    test_df = pd.read_csv("./data/processed/test_preprocessed.csv")
    
    # Separate features and target
    X_train = train_df.drop(columns=['soh'])
    y_train = train_df['soh']
    
    X_val = val_df.drop(columns=['soh'])
    y_val = val_df['soh']
    
    X_test = test_df.drop(columns=['soh'])
    y_test = test_df['soh']
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Engineer features
    X_train_eng = feature_engineer.engineer_features(X_train, fit=True)
    X_val_eng = feature_engineer.engineer_features(X_val, fit=False)
    X_test_eng = feature_engineer.engineer_features(X_test, fit=False)
    
    # Save feature engineer
    feature_engineer.save_feature_engineer()
    
    # Save engineered features
    X_train_eng['soh'] = y_train.values
    X_val_eng['soh'] = y_val.values
    X_test_eng['soh'] = y_test.values
    
    X_train_eng.to_csv("./data/processed/train_engineered.csv", index=False)
    X_val_eng.to_csv("./data/processed/val_engineered.csv", index=False)
    X_test_eng.to_csv("./data/processed/test_engineered.csv", index=False)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Train shape: {X_train_eng.shape}")
    print(f"Validation shape: {X_val_eng.shape}")
    print(f"Test shape: {X_test_eng.shape}")
    print(f"\nFeatures: {X_train_eng.columns.tolist()}")
    print("="*60 + "\n")
    
    logger.info("Feature engineering completed successfully")


if __name__ == "__main__":
    main()
