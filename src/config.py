"""
Configuration Management Module

Handles loading and accessing configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration handler for the project."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            # __file__ is in src/config.py, so parent.parent gets to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._override_with_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _override_with_env(self):
        """Override configuration with environment variables where applicable."""
        # Azure configuration
        if os.getenv('AZURE_SUBSCRIPTION_ID'):
            self.config['azure']['subscription_id'] = os.getenv('AZURE_SUBSCRIPTION_ID')
        if os.getenv('AZURE_RESOURCE_GROUP'):
            self.config['azure']['resource_group'] = os.getenv('AZURE_RESOURCE_GROUP')
        if os.getenv('AZURE_WORKSPACE_NAME'):
            self.config['azure']['workspace_name'] = os.getenv('AZURE_WORKSPACE_NAME')
        
        # API configuration
        if os.getenv('API_HOST'):
            self.config['api']['host'] = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.config['api']['port'] = int(os.getenv('API_PORT'))
        if os.getenv('API_WORKERS'):
            self.config['api']['workers'] = int(os.getenv('API_WORKERS'))
        
        # MLflow configuration
        if os.getenv('MLFLOW_TRACKING_URI'):
            self.config['mlflow']['tracking_uri'] = os.getenv('MLFLOW_TRACKING_URI')
        if os.getenv('MLFLOW_EXPERIMENT_NAME'):
            self.config['mlflow']['experiment_name'] = os.getenv('MLFLOW_EXPERIMENT_NAME')
        
        # Logging configuration
        if os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.config['logging']['log_file'] = os.getenv('LOG_FILE')
        
        # Model paths
        if os.getenv('MODEL_PATH'):
            self.config['api']['model_path'] = os.getenv('MODEL_PATH')
        if os.getenv('SCALER_PATH'):
            self.config['api']['scaler_path'] = os.getenv('SCALER_PATH')
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'azure.subscription_id')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_generation_config(self) -> Dict[str, Any]:
        """Get data generation configuration."""
        return self.config.get('data_generation', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.config.get('preprocessing', {})
    
    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.config.get('feature_engineering', {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self.config.get('models', {})
    
    def get_hyperparameter_tuning_config(self) -> Dict[str, Any]:
        """Get hyperparameter tuning configuration."""
        return self.config.get('hyperparameter_tuning', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.config.get('mlflow', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config.get('api', {})
    
    def get_azure_config(self) -> Dict[str, Any]:
        """Get Azure configuration."""
        return self.config.get('azure', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})


# Global configuration instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to config file. Only used on first call.
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
