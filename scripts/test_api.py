"""
Quick start script to test the API locally.
"""

import requests
import json


def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_prediction():
    """Test prediction endpoint."""
    print("Testing prediction endpoint...")
    
    data = {
        "cycles": 500,
        "temperature": 25.0,
        "c_rate": 1.0,
        "voltage_min": 3.0,
        "voltage_max": 4.2,
        "usage_hours": 1200,
        "humidity": 50.0
    }
    
    response = requests.post(
        "http://localhost:8000/predict",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("Testing batch prediction endpoint...")
    
    data = {
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
    
    response = requests.post(
        "http://localhost:8000/predict/batch",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_model_info():
    """Test model info endpoint."""
    print("Testing model info endpoint...")
    response = requests.get("http://localhost:8000/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("API TESTING SUITE")
    print("="*60 + "\n")
    print("Make sure the API is running at http://localhost:8000")
    print("Start with: uvicorn src.api.app:app --reload\n")
    
    try:
        test_health()
        test_model_info()
        test_prediction()
        test_batch_prediction()
        
        print("="*60)
        print("ALL TESTS COMPLETED")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("Please start the API server first:")
        print("  uvicorn src.api.app:app --reload\n")
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
