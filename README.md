# Battery Degradation Prediction - ML Project

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLOps](https://img.shields.io/badge/MLOps-Azure%20ML-blue.svg)

## 🎯 Business Problem

Predict battery State of Health (SOH) degradation using regression models to optimize battery lifecycle management, maintenance scheduling, and replacement planning.

## 📊 Dataset Features

The model predicts battery capacity fade based on:

- **Charge/Discharge Cycles**: Total number of charge-discharge cycles
- **Temperature**: Operating temperature during usage (°C)
- **C-Rate**: Charging/discharging current rate
- **Voltage Ranges**: Min/max voltage during operation
- **Usage Hours**: Cumulative operational hours
- **Environmental Conditions**: Humidity, ambient temperature, storage conditions

**Target Variable**: Battery SOH (State of Health) as percentage of original capacity

## 🏗️ Project Structure

```
BatteriAnalysis/
├── src/
│   ├── data/               # Data generation and loading
│   ├── preprocessing/      # Data cleaning and transformation
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and selection
│   ├── evaluation/        # Model evaluation metrics
│   └── api/               # FastAPI prediction service
├── data/
│   ├── raw/               # Original datasets
│   ├── processed/         # Cleaned and transformed data
│   └── external/          # External data sources
├── models/
│   ├── saved/             # Trained model artifacts
│   └── experiments/       # MLflow experiment tracking
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit and integration tests
├── config/                # Configuration files
├── azure/                 # Azure ML deployment configs
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── .github/workflows/     # CI/CD pipelines

```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python src/data/generate_data.py
```

### 3. Train Model

```bash
python src/models/train_model.py
```

### 4. Start API Service

```bash
uvicorn src.api.app:app --reload --port 8000
```

### 5. Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cycles": 500,
    "temperature": 25.0,
    "c_rate": 1.0,
    "voltage_min": 3.0,
    "voltage_max": 4.2,
    "usage_hours": 1200,
    "humidity": 50.0
  }'
```

## 📈 Model Performance

| Model | RMSE | MAE | R² Score | MAPE |
|-------|------|-----|----------|------|
| Linear Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| Neural Network | TBD | TBD | TBD | TBD |

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Data generation parameters
- Model hyperparameters
- Feature engineering settings
- Azure ML workspace details

## ☁️ Azure Deployment

### Deploy to Azure ML

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription <subscription-id>

# Deploy model
python azure/deploy_model.py
```

### Deploy API to Azure Container Instances

```bash
# Build Docker image
docker build -t battery-prediction-api .

# Push to Azure Container Registry
az acr login --name <registry-name>
docker push <registry-name>.azurecr.io/battery-prediction-api

# Deploy to ACI
az container create --resource-group <rg-name> \
  --name battery-api \
  --image <registry-name>.azurecr.io/battery-prediction-api \
  --cpu 2 --memory 4
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

- [Data Generation](docs/data_generation.md)
- [Feature Engineering](docs/feature_engineering.md)
- [Model Training](docs/model_training.md)
- [API Documentation](docs/api_documentation.md)
- [Azure Deployment](docs/azure_deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Data Science Team

## 🙏 Acknowledgments

- Battery degradation physics research papers
- NASA Battery Dataset
- Azure ML community
