"""
Model training and hyperparameter tuning.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

import lightgbm as lgb

# MLflow for experiment tracking
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# ML Models
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score

from config import get_config
from evaluation.metrics import ModelEvaluator
from logger import get_logger, setup_logger
from utils import format_time, load_model, print_metrics_summary, save_model

logger = setup_logger(__name__, log_file="./logs/model_training.log")


class ModelTrainer:
    """Train and evaluate regression models for battery degradation prediction."""

    def __init__(self, config: dict = None):
        """
        Initialize model trainer.

        Args:
            config: Configuration dictionary. If None, loads from config.yaml
        """
        if config is None:
            self.config = get_config()
        else:
            self.config = config

        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = float("inf")

        self.evaluator = ModelEvaluator()

        # Setup MLflow
        mlflow_config = self.config.get_mlflow_config()
        mlflow.set_tracking_uri(
            mlflow_config.get("tracking_uri", "./models/experiments")
        )
        mlflow.set_experiment(
            mlflow_config.get("experiment_name", "battery_degradation")
        )

        logger.info("ModelTrainer initialized")

    def get_model(self, model_name: str, params: Dict = None) -> Any:
        """
        Get model instance by name.

        Args:
            model_name: Name of the model
            params: Model parameters

        Returns:
            Model instance
        """
        if params is None:
            params = {}

        models_dict = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(**params),
            "lasso": Lasso(**params),
            "random_forest": RandomForestRegressor(
                random_state=42, n_jobs=-1, **params
            ),
            "gradient_boosting": GradientBoostingRegressor(random_state=42, **params),
            "xgboost": xgb.XGBRegressor(random_state=42, n_jobs=-1, **params),
            "lightgbm": lgb.LGBMRegressor(
                random_state=42, n_jobs=-1, verbose=-1, **params
            ),
        }

        if model_name not in models_dict:
            raise ValueError(f"Unknown model: {model_name}")

        return models_dict[model_name]

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Dict = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single model.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters

        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        logger.info(f"Training {model_name} model")
        start_time = time.time()

        # Get model
        model = self.get_model(model_name, params)

        # Train model
        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        logger.info(f"{model_name} training completed in {format_time(training_time)}")

        # Evaluate on training set
        y_train_pred = model.predict(X_train)
        train_metrics = self.evaluator.calculate_metrics(y_train, y_train_pred)

        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_metrics = self.evaluator.calculate_metrics(y_val, y_val_pred)
            logger.info(f"{model_name} validation RMSE: {val_metrics['rmse']:.4f}")

        # Combine metrics
        metrics = {
            "training_time": training_time,
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "train_mape": train_metrics["mape"],
        }

        if val_metrics:
            metrics.update(
                {
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                    "val_mape": val_metrics["mape"],
                }
            )

        return model, metrics

    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict,
    ) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search

        Returns:
            Tuple of (best model, best parameters)
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}")

        tuning_config = self.config.get_hyperparameter_tuning_config()

        # Get base model
        base_model = self.get_model(model_name)

        # Setup GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tuning_config.get("cv_folds", 5),
            scoring=tuning_config.get("scoring", "neg_root_mean_squared_error"),
            n_jobs=tuning_config.get("n_jobs", -1),
            verbose=1,
        )

        # Perform search
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time

        logger.info(f"Hyperparameter tuning completed in {format_time(tuning_time)}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, Dict]:
        """
        Train all configured models.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary of model results
        """
        logger.info("Starting training for all models")

        models_config = self.config.get_models_config()
        results = {}

        for model_name, model_config in models_config.items():
            if not model_config.get("enabled", True):
                logger.info(f"Skipping {model_name} (disabled in config)")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {model_name}")
            logger.info(f"{'='*60}")

            with mlflow.start_run(run_name=model_name):
                try:
                    # Get parameters
                    params = model_config.get("params", {})

                    # Hyperparameter tuning
                    if tune_hyperparameters and params:
                        # Check if params is a grid (dict of lists)
                        is_grid = any(isinstance(v, list) for v in params.values())

                        if is_grid:
                            model, best_params = self.hyperparameter_tuning(
                                model_name, X_train, y_train, params
                            )
                            mlflow.log_params(best_params)
                        else:
                            model, metrics = self.train_model(
                                model_name, X_train, y_train, X_val, y_val, params
                            )
                            mlflow.log_params(params)
                    else:
                        model, metrics = self.train_model(
                            model_name, X_train, y_train, X_val, y_val, params
                        )
                        # Log parameters if they exist and aren't hyperparameter grids
                        if params and not any(
                            isinstance(v, list) for v in params.values()
                        ):
                            mlflow.log_params(params)

                    # Re-evaluate if we did hyperparameter tuning
                    if (
                        tune_hyperparameters
                        and params
                        and any(isinstance(v, list) for v in params.values())
                    ):
                        y_train_pred = model.predict(X_train)
                        y_val_pred = model.predict(X_val)

                        train_metrics = self.evaluator.calculate_metrics(
                            y_train, y_train_pred
                        )
                        val_metrics = self.evaluator.calculate_metrics(
                            y_val, y_val_pred
                        )

                        metrics = {
                            "train_rmse": train_metrics["rmse"],
                            "train_mae": train_metrics["mae"],
                            "train_r2": train_metrics["r2"],
                            "train_mape": train_metrics["mape"],
                            "val_rmse": val_metrics["rmse"],
                            "val_mae": val_metrics["mae"],
                            "val_r2": val_metrics["r2"],
                            "val_mape": val_metrics["mape"],
                        }

                    # Log metrics
                    mlflow.log_metrics(metrics)

                    # Log model
                    if model_name in ["xgboost"]:
                        mlflow.xgboost.log_model(model, "model")
                    elif model_name in ["lightgbm"]:
                        mlflow.lightgbm.log_model(model, "model")
                    else:
                        mlflow.sklearn.log_model(model, "model")

                    # Generate evaluation report
                    y_val_pred = model.predict(X_val)

                    # Get feature importances if available
                    feature_importances = None
                    if hasattr(model, "feature_importances_"):
                        feature_importances = model.feature_importances_
                    elif hasattr(model, "coef_"):
                        feature_importances = np.abs(model.coef_)

                    eval_metrics = self.evaluator.generate_evaluation_report(
                        y_val,
                        y_val_pred,
                        feature_names=X_train.columns.tolist(),
                        feature_importances=feature_importances,
                        model_name=model_name,
                    )

                    # Store model and results
                    self.models[model_name] = model
                    results[model_name] = {
                        "model": model,
                        "metrics": metrics,
                        "eval_metrics": eval_metrics,
                    }

                    # Track best model
                    val_rmse = metrics.get("val_rmse", float("inf"))
                    if val_rmse < self.best_score:
                        self.best_score = val_rmse
                        self.best_model = model
                        self.best_model_name = model_name
                        logger.info(
                            f"New best model: {model_name} (RMSE: {val_rmse:.4f})"
                        )

                    # Print summary
                    print_metrics_summary(metrics)

                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue

        logger.info(
            f"\nBest model: {self.best_model_name} (RMSE: {self.best_score:.4f})"
        )

        return results

    def save_best_model(self, save_path: str = "./models/saved/best_model.pkl") -> None:
        """
        Save the best model.

        Args:
            save_path: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")

        model_info = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "best_score": self.best_score,
        }

        save_model(model_info, save_path)
        logger.info(f"Best model ({self.best_model_name}) saved to {save_path}")

    def evaluate_on_test_set(
        self, X_test: pd.DataFrame, y_test: pd.Series, model: Any = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test target
            model: Model to evaluate. If None, uses best model.

        Returns:
            Dictionary of test metrics
        """
        if model is None:
            model = self.best_model

        if model is None:
            raise ValueError("No model provided and no best model available")

        logger.info("Evaluating on test set")

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        test_metrics = self.evaluator.calculate_metrics(y_test, y_pred)

        logger.info("Test set evaluation:")
        print_metrics_summary(test_metrics)

        # Generate evaluation report
        model_name = self.best_model_name if model == self.best_model else "test_model"

        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            feature_importances = np.abs(model.coef_)

        self.evaluator.generate_evaluation_report(
            y_test,
            y_pred,
            feature_names=X_test.columns.tolist(),
            feature_importances=feature_importances,
            model_name=f"{model_name}_test",
            save_dir="./models/experiments",
        )

        return test_metrics


def main():
    """Main function to train models."""
    logger.info("Starting model training pipeline")

    # Load engineered features
    train_df = pd.read_csv("./data/processed/train_engineered.csv")
    val_df = pd.read_csv("./data/processed/val_engineered.csv")
    test_df = pd.read_csv("./data/processed/test_engineered.csv")

    # Separate features and target
    X_train = train_df.drop(columns=["soh"])
    y_train = train_df["soh"]

    X_val = val_df.drop(columns=["soh"])
    y_val = val_df["soh"]

    X_test = test_df.drop(columns=["soh"])
    y_test = test_df["soh"]

    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    # Initialize trainer
    trainer = ModelTrainer()

    # Train all models
    results = trainer.train_all_models(
        X_train, y_train, X_val, y_val, tune_hyperparameters=True
    )

    # Save best model
    trainer.save_best_model()

    # Evaluate on test set
    test_metrics = trainer.evaluate_on_test_set(X_test, y_test)

    # Compare models
    comparison_metrics = {}
    for model_name, result in results.items():
        comparison_metrics[model_name] = result["eval_metrics"]

    evaluator = ModelEvaluator()
    evaluator.compare_models(
        comparison_metrics, save_path="./models/experiments/plots/model_comparison.png"
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Validation RMSE: {trainer.best_score:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print("=" * 60 + "\n")

    logger.info("Model training pipeline completed successfully")


if __name__ == "__main__":
    main()
