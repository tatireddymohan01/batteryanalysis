"""
Complete end-to-end pipeline script.
Runs all steps from data generation to model training.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from src.data.generate_data import BatteryDegradationGenerator
from src.features.feature_engineering import FeatureEngineer
from src.logger import setup_logger
from src.models.train_model import ModelTrainer
from src.preprocessing.preprocessor import DataPreprocessor

logger = setup_logger(__name__, log_file="./logs/pipeline.log")


def run_pipeline():
    """Execute complete ML pipeline."""

    print("\n" + "=" * 70)
    print("BATTERY DEGRADATION PREDICTION - END-TO-END ML PIPELINE")
    print("=" * 70 + "\n")

    # Step 1: Data Generation
    print("Step 1/5: Generating synthetic dataset...")
    logger.info("Starting data generation")

    generator = BatteryDegradationGenerator()
    df = generator.generate_dataset()
    train_df, val_df, test_df = generator.generate_train_test_split(df)

    print(f"✓ Generated {len(df)} samples")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Validation: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples\n")

    # Step 2: Data Preprocessing
    print("Step 2/5: Preprocessing data...")
    logger.info("Starting data preprocessing")

    preprocessor = DataPreprocessor()

    X_train, y_train = preprocessor.preprocess(train_df, fit=True)
    X_val, y_val = preprocessor.preprocess(val_df, fit=False, remove_outliers=False)
    X_test, y_test = preprocessor.preprocess(test_df, fit=False, remove_outliers=False)

    preprocessor.save_preprocessor()

    # Save preprocessed data
    X_train["soh"] = y_train
    X_val["soh"] = y_val
    X_test["soh"] = y_test

    X_train.to_csv("./data/processed/train_preprocessed.csv", index=False)
    X_val.to_csv("./data/processed/val_preprocessed.csv", index=False)
    X_test.to_csv("./data/processed/test_preprocessed.csv", index=False)

    print(f"✓ Preprocessing complete")
    print(f"  - Train shape: {X_train.shape}")
    print(f"  - Validation shape: {X_val.shape}")
    print(f"  - Test shape: {X_test.shape}\n")

    # Step 3: Feature Engineering
    print("Step 3/5: Engineering features...")
    logger.info("Starting feature engineering")

    X_train_clean = X_train.drop(columns=["soh"])
    X_val_clean = X_val.drop(columns=["soh"])
    X_test_clean = X_test.drop(columns=["soh"])

    feature_engineer = FeatureEngineer()

    X_train_eng = feature_engineer.engineer_features(X_train_clean, fit=True)
    X_val_eng = feature_engineer.engineer_features(X_val_clean, fit=False)
    X_test_eng = feature_engineer.engineer_features(X_test_clean, fit=False)

    feature_engineer.save_feature_engineer()

    # Save engineered features
    X_train_eng["soh"] = y_train.values
    X_val_eng["soh"] = y_val.values
    X_test_eng["soh"] = y_test.values

    X_train_eng.to_csv("./data/processed/train_engineered.csv", index=False)
    X_val_eng.to_csv("./data/processed/val_engineered.csv", index=False)
    X_test_eng.to_csv("./data/processed/test_engineered.csv", index=False)

    print(f"✓ Feature engineering complete")
    print(
        f"  - Created {len(X_train_eng.columns) - len(X_train_clean.columns)} new features"
    )
    print(f"  - Total features: {X_train_eng.shape[1]}\n")

    # Step 4: Model Training
    print("Step 4/5: Training models...")
    logger.info("Starting model training")

    X_train_final = X_train_eng.drop(columns=["soh"])
    X_val_final = X_val_eng.drop(columns=["soh"])
    X_test_final = X_test_eng.drop(columns=["soh"])

    trainer = ModelTrainer()

    results = trainer.train_all_models(
        X_train_final,
        y_train,
        X_val_final,
        y_val,
        tune_hyperparameters=False,  # Set to True for full hyperparameter tuning
    )

    trainer.save_best_model()

    print(f"✓ Model training complete")
    print(f"  - Models trained: {len(results)}")
    print(f"  - Best model: {trainer.best_model_name}")
    print(f"  - Best validation RMSE: {trainer.best_score:.4f}\n")

    # Step 5: Final Evaluation
    print("Step 5/5: Evaluating on test set...")
    logger.info("Starting final evaluation")

    test_metrics = trainer.evaluate_on_test_set(X_test_final, y_test)

    print(f"✓ Evaluation complete\n")

    # Final Summary
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"\nTest Set Performance:")
    print(f"  - RMSE: {test_metrics['rmse']:.4f}")
    print(f"  - MAE: {test_metrics['mae']:.4f}")
    print(f"  - R² Score: {test_metrics['r2']:.4f}")
    print(f"  - MAPE: {test_metrics['mape']:.2f}%")
    print(f"\nModel saved to: ./models/saved/best_model.pkl")
    print(f"Preprocessor saved to: ./models/saved/preprocessor.pkl")
    print(f"Feature engineer saved to: ./models/saved/feature_engineer.pkl")
    print("\n" + "=" * 70 + "\n")

    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)
