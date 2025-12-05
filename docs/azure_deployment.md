# Azure Deployment Guide

## Prerequisites

1. **Azure Account**: Active Azure subscription
2. **Azure CLI**: Install from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
3. **Azure ML SDK**: Installed via `requirements.txt`
4. **Service Principal**: For automated deployments

---

## Setup Azure ML Workspace

### 1. Login to Azure
```bash
az login
```

### 2. Set Subscription
```bash
az account set --subscription <subscription-id>
```

### 3. Create Resource Group
```bash
az group create --name battery-ml-rg --location eastus
```

### 4. Create Azure ML Workspace
```bash
az ml workspace create --name battery-ml-workspace \
  --resource-group battery-ml-rg \
  --location eastus
```

Or use the Python script:
```bash
python azure/setup_workspace.py
```

---

## Update Configuration

Edit `config/config.yaml` with your Azure details:

```yaml
azure:
  subscription_id: "your-subscription-id"
  resource_group: "battery-ml-rg"
  workspace_name: "battery-ml-workspace"
```

Or set environment variables:
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="battery-ml-rg"
export AZURE_WORKSPACE_NAME="battery-ml-workspace"
```

---

## Deploy Model to Azure ML

### 1. Train and Save Model Locally
```bash
python scripts/run_pipeline.py
```

### 2. Deploy to Azure ML
```bash
python azure/deploy_model.py
```

This will:
- Register the model in Azure ML
- Create a managed online endpoint
- Deploy the model with scoring script
- Set up automatic scaling

### 3. Test the Endpoint
```python
import requests
import json

# Get endpoint URI from Azure ML Studio or deployment output
endpoint_uri = "https://<endpoint-name>.<region>.inference.ml.azure.com/score"

# Get API key from Azure ML Studio
api_key = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "cycles": 500,
    "temperature": 25.0,
    "c_rate": 1.0,
    "voltage_min": 3.0,
    "voltage_max": 4.2,
    "usage_hours": 1200,
    "humidity": 50.0
}

response = requests.post(endpoint_uri, json=data, headers=headers)
print(response.json())
```

---

## Deploy as Azure Container Instance (ACI)

### 1. Build Docker Image
```bash
docker build -t battery-prediction-api .
```

### 2. Create Azure Container Registry
```bash
az acr create --resource-group battery-ml-rg \
  --name batterymlregistry --sku Basic
```

### 3. Login to ACR
```bash
az acr login --name batterymlregistry
```

### 4. Tag and Push Image
```bash
docker tag battery-prediction-api batterymlregistry.azurecr.io/battery-prediction-api:v1
docker push batterymlregistry.azurecr.io/battery-prediction-api:v1
```

### 5. Deploy to ACI
```bash
az container create \
  --resource-group battery-ml-rg \
  --name battery-api \
  --image batterymlregistry.azurecr.io/battery-prediction-api:v1 \
  --cpu 2 \
  --memory 4 \
  --registry-login-server batterymlregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label battery-prediction \
  --ports 8000
```

### 6. Get Container IP
```bash
az container show --resource-group battery-ml-rg \
  --name battery-api --query ipAddress.fqdn
```

---

## Deploy to Azure Kubernetes Service (AKS)

### 1. Create AKS Cluster
```bash
az aks create \
  --resource-group battery-ml-rg \
  --name battery-aks-cluster \
  --node-count 2 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

### 2. Get AKS Credentials
```bash
az aks get-credentials --resource-group battery-ml-rg --name battery-aks-cluster
```

### 3. Create Kubernetes Deployment
Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: battery-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: battery-prediction
  template:
    metadata:
      labels:
        app: battery-prediction
    spec:
      containers:
      - name: api
        image: batterymlregistry.azurecr.io/battery-prediction-api:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: battery-prediction-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: battery-prediction
```

### 4. Deploy to AKS
```bash
kubectl apply -f k8s-deployment.yaml
```

### 5. Get Service IP
```bash
kubectl get service battery-prediction-service
```

---

## Monitoring and Logging

### Azure Application Insights
```python
# Add to src/api/app.py
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=<your-key>'
))
```

### View Logs
```bash
# Azure ML Endpoint Logs
az ml online-endpoint get-logs --name <endpoint-name> \
  --resource-group battery-ml-rg --workspace-name battery-ml-workspace

# ACI Logs
az container logs --resource-group battery-ml-rg --name battery-api

# AKS Logs
kubectl logs -l app=battery-prediction
```

---

## CI/CD with GitHub Actions

The project includes a GitHub Actions workflow (`.github/workflows/ci-cd.yml`) that:

1. Runs tests on every push
2. Builds Docker image on main branch
3. Deploys to Azure ML automatically

### Setup Secrets
Add these secrets to your GitHub repository:

- `AZURE_CREDENTIALS`: Azure service principal JSON
- `AZURE_SUBSCRIPTION_ID`: Your subscription ID
- `AZURE_RESOURCE_GROUP`: Resource group name
- `AZURE_WORKSPACE_NAME`: Workspace name
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password

---

## Cost Optimization

1. **Use appropriate instance sizes** - Start with Standard_DS2_v2
2. **Enable autoscaling** - Scale down when not in use
3. **Use spot instances** - For development/testing
4. **Monitor usage** - Set up cost alerts
5. **Delete unused resources** - Clean up test deployments

---

## Troubleshooting

### Common Issues

**Issue: Model fails to load**
- Ensure all files (model, preprocessor, feature_engineer) are in the correct paths
- Check Azure ML logs for detailed error messages

**Issue: Out of memory**
- Increase instance size in deployment config
- Optimize model size (use quantization, pruning)

**Issue: Slow predictions**
- Enable batch predictions
- Use caching for common inputs
- Optimize feature engineering pipeline

**Issue: Authentication errors**
- Verify service principal credentials
- Check Azure RBAC permissions
- Regenerate API keys if needed

---

## Security Best Practices

1. **Use managed identities** instead of service principals
2. **Enable network isolation** - Use VNet integration
3. **Rotate API keys regularly**
4. **Use Azure Key Vault** for secrets
5. **Enable HTTPS only**
6. **Implement rate limiting**
7. **Add request validation**
8. **Monitor for anomalies**

---

## Resources

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
