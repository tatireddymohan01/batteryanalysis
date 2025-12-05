"""
Azure ML workspace setup script.
"""

import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, ComputeInstance, Workspace
from azure.identity import DefaultAzureCredential

sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.logger import setup_logger

logger = setup_logger(__name__, log_file="./logs/azure_setup.log")


def create_workspace():
    """Create or get Azure ML workspace."""
    config = get_config()
    azure_config = config.get_azure_config()

    logger.info("Setting up Azure ML workspace")

    # Initialize credential
    credential = DefaultAzureCredential()

    # Get or create workspace
    try:
        ml_client = MLClient(
            credential, azure_config["subscription_id"], azure_config["resource_group"]
        )

        # Try to get existing workspace
        try:
            workspace = ml_client.workspaces.get(azure_config["workspace_name"])
            logger.info(f"Using existing workspace: {workspace.name}")
        except:
            # Create new workspace
            workspace = Workspace(
                name=azure_config["workspace_name"],
                location="eastus",  # Change as needed
                display_name="Battery Degradation ML Workspace",
                description="Workspace for battery SOH prediction models",
                tags={"project": "battery-degradation"},
            )
            workspace = ml_client.workspaces.begin_create(workspace).result()
            logger.info(f"Created workspace: {workspace.name}")

        return workspace

    except Exception as e:
        logger.error(f"Error setting up workspace: {str(e)}")
        raise


def create_compute_cluster():
    """Create compute cluster for training."""
    config = get_config()
    azure_config = config.get_azure_config()
    compute_config = azure_config.get("compute", {})

    logger.info("Setting up compute cluster")

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        azure_config["subscription_id"],
        azure_config["resource_group"],
        azure_config["workspace_name"],
    )

    try:
        # Try to get existing compute
        compute = ml_client.compute.get(compute_config["name"])
        logger.info(f"Using existing compute: {compute.name}")
    except:
        # Create new compute
        compute = AmlCompute(
            name=compute_config["name"],
            size=compute_config["vm_size"],
            min_instances=compute_config["min_nodes"],
            max_instances=compute_config["max_nodes"],
            idle_time_before_scale_down=120,
            tier="dedicated",
        )
        compute = ml_client.compute.begin_create_or_update(compute).result()
        logger.info(f"Created compute: {compute.name}")

    return compute


def main():
    """Main setup function."""
    print("Setting up Azure ML environment...")

    # Create workspace
    workspace = create_workspace()
    print(f"✓ Workspace: {workspace.name}")

    # Create compute cluster
    compute = create_compute_cluster()
    print(f"✓ Compute: {compute.name}")

    print("\nAzure ML environment setup complete!")


if __name__ == "__main__":
    main()
