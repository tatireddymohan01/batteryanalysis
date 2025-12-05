"""
Model evaluation metrics and utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from logger import get_logger
from utils import ensure_dir

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate regression model performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        logger.info("ModelEvaluator initialized")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'max_error': max_error(y_true, y_pred)
        }
        
        # Additional custom metrics
        metrics['mean_percentage_error'] = np.mean(
            np.abs((y_true - y_pred) / y_true) * 100
        )
        
        # Adjusted R²
        n = len(y_true)
        p = 1  # Number of predictors (will be updated by caller if needed)
        metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        return metrics
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predicted vs Actual SOH",
        save_path: str = None
    ) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual SOH (%)', fontsize=12)
        plt.ylabel('Predicted SOH (%)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² text
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Plot",
        save_path: str = None
    ) -> None:
        """
        Plot residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted SOH (%)', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Error Distribution",
        save_path: str = None
    ) -> None:
        """
        Plot error distribution analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
        """
        errors = np.abs(y_true - y_pred)
        percentage_errors = (errors / y_true) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Absolute errors
        axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Absolute Error', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Absolute Error Distribution', fontsize=11, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Percentage errors
        axes[0, 1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[0, 1].set_xlabel('Percentage Error (%)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Percentage Error Distribution', fontsize=11, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error vs True value
        axes[1, 0].scatter(y_true, errors, alpha=0.5, s=20)
        axes[1, 0].set_xlabel('True SOH (%)', fontsize=11)
        axes[1, 0].set_ylabel('Absolute Error', fontsize=11)
        axes[1, 0].set_title('Error vs True Value', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative error distribution
        sorted_errors = np.sort(percentage_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        axes[1, 1].plot(sorted_errors, cumulative, linewidth=2)
        axes[1, 1].set_xlabel('Percentage Error (%)', fontsize=11)
        axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=11)
        axes[1, 1].set_title('Cumulative Error Distribution', fontsize=11, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(x=5, color='r', linestyle='--', label='5% Error')
        axes[1, 1].legend()
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: list,
        importances: np.ndarray,
        title: str = "Feature Importance",
        save_path: str = None,
        top_n: int = 20
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Array of feature importance values
            title: Plot title
            save_path: Path to save the plot
            top_n: Number of top features to display
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: list = None,
        feature_importances: np.ndarray = None,
        model_name: str = "Model",
        save_dir: str = "./models/experiments"
    ) -> Dict[str, float]:
        """
        Generate comprehensive evaluation report with plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            feature_names: List of feature names
            feature_importances: Array of feature importances
            model_name: Name of the model
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Generating evaluation report for {model_name}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Create plots
        plots_dir = Path(save_dir) / "plots" / model_name.lower().replace(" ", "_")
        ensure_dir(plots_dir)
        
        self.plot_predictions(
            y_true, y_pred,
            title=f"{model_name}: Predicted vs Actual",
            save_path=str(plots_dir / "predictions.png")
        )
        
        self.plot_residuals(
            y_true, y_pred,
            title=f"{model_name}: Residual Analysis",
            save_path=str(plots_dir / "residuals.png")
        )
        
        self.plot_error_distribution(
            y_true, y_pred,
            title=f"{model_name}: Error Distribution",
            save_path=str(plots_dir / "error_distribution.png")
        )
        
        if feature_names is not None and feature_importances is not None:
            self.plot_feature_importance(
                feature_names, feature_importances,
                title=f"{model_name}: Feature Importance",
                save_path=str(plots_dir / "feature_importance.png")
            )
        
        logger.info(f"Evaluation report generated for {model_name}")
        logger.info(f"Metrics: {metrics}")
        
        return metrics
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: str = None
    ) -> None:
        """
        Compare multiple models.
        
        Args:
            results: Dictionary of model names and their metrics
            save_path: Path to save the comparison plot
        """
        metrics_df = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['rmse', 'mae', 'r2', 'mape']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            if metric in metrics_df.columns:
                metrics_df[metric].sort_values().plot(kind='barh', ax=ax, color='skyblue')
                ax.set_xlabel('Value', fontsize=11)
                ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function to demonstrate evaluation."""
    # Example usage
    evaluator = ModelEvaluator()
    
    # Generate dummy data
    np.random.seed(42)
    y_true = np.random.uniform(70, 100, 1000)
    y_pred = y_true + np.random.normal(0, 2, 1000)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate report
    evaluator.generate_evaluation_report(
        y_true, y_pred,
        model_name="Example Model"
    )
    
    logger.info("Evaluation demonstration completed")


if __name__ == "__main__":
    main()
