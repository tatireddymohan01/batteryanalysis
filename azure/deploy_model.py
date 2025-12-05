"""
Azure ML deployment script for battery degradation prediction model.
"""

import os
import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential

sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.logger import setup_logger

logger = setup_logger(__name__, log_file="./logs/azure_deployment.log")


class AzureMLDeployer:
    """Deploy model to Azure ML."""

    def __init__(self):
        """Initialize Azure ML deployer."""
        self.config = get_config()
        azure_config = self.config.get_azure_config()

        # Initialize ML Client
        try:
            self.ml_client = MLClient(
                DefaultAzureCredential(),
                azure_config["subscription_id"],
                azure_config["resource_group"],
                azure_config["workspace_name"],
            )
            logger.info("Azure ML Client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML Client: {str(e)}")
            raise

    def register_model(self, model_path: str, model_name: str = None) -> Model:
        """
        Register model in Azure ML.

        Args:
            model_path: Path to the model file
            model_name: Name for the registered model

        Returns:
            Registered model object
        """
        if model_name is None:
            model_name = (
                self.config.get("azure", {})
                .get("model_registry", {})
                .get("model_name", "battery_soh_predictor")
            )

        logger.info(f"Registering model: {model_name}")

        model = Model(
            path=model_path,
            name=model_name,
            description="Battery State of Health (SOH) prediction model",
            tags={
                "task": "regression",
                "framework": "scikit-learn",
                "domain": "battery-degradation",
            },
        )

        registered_model = self.ml_client.models.create_or_update(model)
        logger.info(
            f"Model registered: {registered_model.name}, Version: {registered_model.version}"
        )

        return registered_model

    def create_endpoint(self, endpoint_name: str = None) -> ManagedOnlineEndpoint:
        """
        Create managed online endpoint.

        Args:
            endpoint_name: Name for the endpoint

        Returns:
            Created endpoint object
        """
        if endpoint_name is None:
            endpoint_name = (
                self.config.get("azure", {})
                .get("deployment", {})
                .get("endpoint_name", "battery-prediction-endpoint")
            )

        logger.info(f"Creating endpoint: {endpoint_name}")

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Battery SOH prediction endpoint",
            auth_mode="key",
            tags={"project": "battery-degradation", "version": "1.0"},
        )

        try:
            created_endpoint = self.ml_client.online_endpoints.begin_create_or_update(
                endpoint
            ).result()
            logger.info(f"Endpoint created: {created_endpoint.name}")
            return created_endpoint
        except Exception as e:
            logger.error(f"Failed to create endpoint: {str(e)}")
            raise

    def create_deployment(
        self, endpoint_name: str, model: Model, deployment_name: str = None
    ) -> ManagedOnlineDeployment:
        """
        Create deployment under endpoint.

        Args:
            endpoint_name: Name of the endpoint
            model: Registered model object
            deployment_name: Name for the deployment

        Returns:
            Created deployment object
        """
        if deployment_name is None:
            deployment_name = (
                self.config.get("azure", {})
                .get("deployment", {})
                .get("deployment_name", "battery-prediction-v1")
            )

        logger.info(f"Creating deployment: {deployment_name}")

        deployment_config = self.config.get("azure", {}).get("deployment", {})

        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            instance_type=deployment_config.get("instance_type", "Standard_DS2_v2"),
            instance_count=deployment_config.get("instance_count", 1),
            code_configuration=CodeConfiguration(
                code="./src", scoring_script="score.py"
            ),
            environment=Environment(
                conda_file="./azure/conda_env.yml",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            ),
        )

        try:
            created_deployment = (
                self.ml_client.online_deployments.begin_create_or_update(
                    deployment
                ).result()
            )
            logger.info(f"Deployment created: {created_deployment.name}")

            # Set traffic to 100% for this deployment
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            endpoint.traffic = {deployment_name: 100}
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logger.info(f"Traffic set to 100% for {deployment_name}")

            return created_deployment
        except Exception as e:
            logger.error(f"Failed to create deployment: {str(e)}")
            raise

    def deploy(
        self,
        model_path: str = "./models/saved/best_model.pkl",
        model_name: str = None,
        endpoint_name: str = None,
        deployment_name: str = None,
    ):
        """
        Complete deployment pipeline.

        Args:
            model_path: Path to the model file
            model_name: Name for the registered model
            endpoint_name: Name for the endpoint
            deployment_name: Name for the deployment
        """
        logger.info("Starting Azure ML deployment pipeline")

        # Register model
        model = self.register_model(model_path, model_name)

        # Create endpoint
        endpoint = self.create_endpoint(endpoint_name)

        # Create deployment
        deployment = self.create_deployment(endpoint.name, model, deployment_name)

        # Get endpoint details
        endpoint_uri = self.ml_client.online_endpoints.get(endpoint.name).scoring_uri

        logger.info("Deployment completed successfully")
        logger.info(f"Endpoint URI: {endpoint_uri}")

        print("\n" + "=" * 60)
        print("AZURE ML DEPLOYMENT COMPLETE")
        print("=" * 60)
        print(f"Endpoint: {endpoint.name}")
        print(f"Deployment: {deployment.name}")
        print(f"Model: {model.name} (v{model.version})")
        print(f"Scoring URI: {endpoint_uri}")
        print("=" * 60 + "\n")


def main():
    """Main function to deploy model to Azure ML."""
    deployer = AzureMLDeployer()
    deployer.deploy()


if __name__ == "__main__":
    main()
