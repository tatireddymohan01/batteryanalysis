# Getting Started with Battery Degradation Prediction

## Quick Start Guide

This guide will help you get the battery degradation prediction system up and running in minutes.

---

## Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

---

## Installation

### 1. Clone or Navigate to the Project
```bash
cd d:\GitHubRepos\BatteriAnalysis
```

### 2. Create Virtual Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Or using Command Prompt
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- Machine learning libraries (scikit-learn, XGBoost, LightGBM)
- API framework (FastAPI, uvicorn)
- Azure ML SDK
- Data science tools (pandas, numpy, matplotlib)

---

## Running the Complete Pipeline

### Option 1: Run Everything at Once
```bash
python scripts/run_pipeline.py
```

This will:
1. ✓ Generate synthetic battery degradation dataset (10,000 samples)
2. ✓ Preprocess and clean the data
3. ✓ Engineer domain-specific features
4. ✓ Train multiple ML models (Linear Regression, Random Forest, XGBoost, LightGBM)
5. ✓ Evaluate and select the best model
6. ✓ Save models and preprocessors

**Expected Output:**
```
=====================================================================
PIPELINE EXECUTION COMPLETE
=====================================================================

Best Model: xgboost

Test Set Performance:
  - RMSE: 1.2345
  - MAE: 0.9876
  - R² Score: 0.9850
  - MAPE: 1.23%

Model saved to: ./models/saved/best_model.pkl
=====================================================================
```

### Option 2: Run Step-by-Step

#### Step 1: Generate Data
```bash
python src/data/generate_data.py
```

#### Step 2: Preprocess Data
```bash
python src/preprocessing/preprocessor.py
```

#### Step 3: Engineer Features
```bash
python src/features/feature_engineering.py
```

#### Step 4: Train Models
```bash
python src/models/train_model.py
```

---

## Starting the API Server

### Local Development
```bash
uvicorn src.api.app:app --reload --port 8000
```

The API will be available at: http://localhost:8000

### Production Mode
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Testing the API

### 1. Open API Documentation
Visit http://localhost:8000/docs in your browser for interactive API documentation.

### 2. Test with Python Script
```bash
python scripts/test_api.py
```

### 3. Test with cURL
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

### 4. Test with Python Requests
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "cycles": 500,
        "temperature": 25.0,
        "c_rate": 1.0,
        "voltage_min": 3.0,
        "voltage_max": 4.2,
        "usage_hours": 1200,
        "humidity": 50.0
    }
)

print(response.json())
```

**Expected Response:**
```json
{
  "soh": 87.5,
  "confidence_interval": {
    "lower": 85.2,
    "upper": 89.8
  },
  "degradation_stage": "Good",
  "recommendation": "Battery in good condition. Continue monitoring."
}
```

---

## Exploring with Jupyter Notebooks

### 1. Start Jupyter
```bash
jupyter notebook
```

### 2. Open Notebook
Navigate to `notebooks/01_exploratory_analysis.ipynb`

This notebook includes:
- Data exploration and visualization
- Feature engineering examples
- Model training demonstrations
- Interactive predictions

---

## Docker Deployment

### 1. Build Docker Image
```bash
docker build -t battery-prediction-api .
```

### 2. Run Container
```bash
docker run -p 8000:8000 battery-prediction-api
```

### 3. Or Use Docker Compose
```bash
docker-compose up
```

---

## Project Structure Overview

```
BatteriAnalysis/
├── src/                    # Source code
│   ├── data/              # Data generation and loading
│   ├── preprocessing/     # Data cleaning and transformation
│   ├── features/          # Feature engineering
│   ├── models/            # Model training
│   ├── evaluation/        # Metrics and evaluation
│   └── api/               # FastAPI application
├── data/                  # Generated datasets
├── models/                # Saved models and experiments
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── azure/                 # Azure ML deployment
└── config/                # Configuration files
```

---

## Configuration

### Edit Configuration
Modify `config/config.yaml` to customize:
- Data generation parameters
- Model hyperparameters
- API settings
- Azure ML configuration

### Environment Variables
Copy `.env.template` to `.env` and fill in your values:
```bash
cp .env.template .env
```

---

## Common Commands

### Generate New Dataset
```bash
python src/data/generate_data.py
```

### Train Models with Custom Config
```bash
python src/models/train_model.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Check Code Quality
```bash
black src/
flake8 src/
```

### View MLflow Experiments
```bash
mlflow ui
```
Then visit http://localhost:5000

---

## Troubleshooting

### Issue: Module not found
**Solution:** Ensure virtual environment is activated and dependencies are installed
```bash
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: Port already in use
**Solution:** Change the port or kill the process
```bash
# Change port
uvicorn src.api.app:app --port 8001

# Or find and kill process (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: Out of memory during training
**Solution:** Reduce dataset size in `config/config.yaml`
```yaml
data_generation:
  n_samples: 5000  # Reduce from 10000
```

### Issue: Models not loading in API
**Solution:** Ensure pipeline has been run first
```bash
python scripts/run_pipeline.py
```

---

## Next Steps

1. **Explore the Data**: Open `notebooks/01_exploratory_analysis.ipynb`
2. **Customize Models**: Edit `config/config.yaml` to try different algorithms
3. **Deploy to Azure**: Follow `docs/azure_deployment.md`
4. **Set up CI/CD**: Configure GitHub Actions with your secrets
5. **Add Tests**: Extend `tests/` with your own test cases

---

## Getting Help

- **Documentation**: See `docs/` folder
- **API Docs**: http://localhost:8000/docs (when server is running)
- **Issues**: Check logs in `logs/` folder
- **Configuration**: Review `config/config.yaml`

---

## Important Files

| File | Purpose |
|------|---------|
| `scripts/run_pipeline.py` | Complete end-to-end pipeline |
| `src/api/app.py` | FastAPI application |
| `config/config.yaml` | Main configuration |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker container definition |
| `README.md` | Project overview |

---

## Performance Expectations

With default settings (10,000 samples):
- **Data Generation**: ~5 seconds
- **Preprocessing**: ~2 seconds
- **Feature Engineering**: ~3 seconds
- **Model Training**: 2-5 minutes (depending on hardware)
- **API Response Time**: <100ms per prediction

---

## Best Practices

1. **Always activate virtual environment** before running scripts
2. **Run the complete pipeline** at least once before starting the API
3. **Check logs** in `logs/` folder if something goes wrong
4. **Use version control** - commit your changes regularly
5. **Test locally** before deploying to production
6. **Monitor performance** using MLflow UI
7. **Update requirements.txt** if you add new dependencies

---

## Success Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Pipeline executed successfully
- [ ] Models saved in `models/saved/`
- [ ] API server starts without errors
- [ ] Can make predictions via API
- [ ] Jupyter notebook runs without errors

---

## Support

For issues or questions:
1. Check the logs in `logs/` folder
2. Review documentation in `docs/` folder
3. Ensure all dependencies are installed
4. Verify Python version is 3.9+

---

**Congratulations!** You now have a complete, production-ready ML system for battery degradation prediction. 🎉
