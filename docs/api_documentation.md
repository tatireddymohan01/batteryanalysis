# API Documentation

## Battery Degradation Prediction API

### Base URL
```
http://localhost:8000
```

### Authentication
Currently no authentication required for local deployment. For production, implement API key authentication.

---

## Endpoints

### 1. Health Check

Check if the API is running and models are loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "feature_engineer_loaded": true
}
```

---

### 2. Single Prediction

Predict battery State of Health (SOH) for a single input.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "cycles": 500,
  "temperature": 25.0,
  "c_rate": 1.0,
  "voltage_min": 3.0,
  "voltage_max": 4.2,
  "usage_hours": 1200,
  "humidity": 50.0
}
```

**Parameters:**
- `cycles` (int): Number of charge/discharge cycles (0-5000)
- `temperature` (float): Operating temperature in °C (-20 to 80)
- `c_rate` (float): Charging/discharging rate (0.05 to 5.0)
- `voltage_min` (float): Minimum voltage in V (2.5 to 3.5)
- `voltage_max` (float): Maximum voltage in V (3.8 to 4.5)
- `usage_hours` (float): Total usage hours (0 to 50000)
- `humidity` (float): Relative humidity in % (0 to 100)

**Response:**
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

### 3. Batch Prediction

Predict SOH for multiple inputs.

**Endpoint:** `POST /predict/batch`

**Request Body:**
```json
{
  "predictions": [
    {
      "cycles": 500,
      "temperature": 25.0,
      "c_rate": 1.0,
      "voltage_min": 3.0,
      "voltage_max": 4.2,
      "usage_hours": 1200,
      "humidity": 50.0
    },
    {
      "cycles": 1000,
      "temperature": 35.0,
      "c_rate": 1.5,
      "voltage_min": 3.0,
      "voltage_max": 4.2,
      "usage_hours": 2500,
      "humidity": 60.0
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "soh": 87.5,
      "confidence_interval": {...},
      "degradation_stage": "Good",
      "recommendation": "..."
    },
    {
      "soh": 78.2,
      "confidence_interval": {...},
      "degradation_stage": "Fair",
      "recommendation": "..."
    }
  ],
  "total_count": 2
}
```

---

### 4. Model Information

Get information about the loaded model.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "model_type": "XGBRegressor",
  "features": ["cycles", "temperature", ...],
  "n_features": 13
}
```

---

## Degradation Stages

| SOH Range | Stage | Description |
|-----------|-------|-------------|
| ≥ 95% | Excellent | Battery in excellent condition |
| 90-95% | Very Good | Battery in very good condition |
| 85-90% | Good | Battery in good condition |
| 80-85% | Fair | Battery shows signs of aging |
| 70-80% | Degraded | Battery is degraded |
| < 70% | Critical | Battery is critically degraded |

---

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 422 | Validation Error - Invalid input parameters |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

---

## Example Usage

### Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "cycles": 500,
    "temperature": 25.0,
    "c_rate": 1.0,
    "voltage_min": 3.0,
    "voltage_max": 4.2,
    "usage_hours": 1200,
    "humidity": 50.0
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL
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

### JavaScript
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    cycles: 500,
    temperature: 25.0,
    c_rate: 1.0,
    voltage_min: 3.0,
    voltage_max: 4.2,
    usage_hours: 1200,
    humidity: 50.0
  })
})
.then(response => response.json())
.then(data => console.log(data));
```
