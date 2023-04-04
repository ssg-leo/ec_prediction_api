
import json
from fastapi.testclient import TestClient
from app import app, InputData

import warnings
warnings.filterwarnings("ignore", message="During the encoding, NaN values were introduced")
warnings.filterwarnings("ignore", message="ntree_limit is deprecated")

client = TestClient(app)

def test_predict():
    # Define a test input
    input_data = {
        "x": 17172.45151,
        "y": 35519.53147,
        "area": 122.0,
        "floor_range": "06-10",
        "type_of_sale": "Resale",
        "district": 22,
        "district_name": "Boon Lay, Jurong, Tuas",
        "remaining_lease": 72.31506849315069,
        "price_index": 168.1
    }

    # Send a POST request to the /predict endpoint with the test input
    response = client.post("/predict/", json=input_data)

    # Check that the response has a 200 status code
    assert response.status_code == 200

    # Check that the response contains the expected keys
    assert set(response.json().keys()) == {"prediction", "shap_values"}

    # Check that the prediction is a float
    assert isinstance(response.json()["prediction"], float)

    # Check that the SHAP values are a dictionary with at least one key
    assert isinstance(response.json()["shap_values"], dict)
    assert len(response.json()["shap_values"].keys()) > 0

    # Check that the SHAP values are dictionaries with the expected keys
    for label_name, shap_dict in response.json()["shap_values"].items():
        assert set(shap_dict.keys()) == set(input_data.keys())

    # Check that the SHAP values are floats
    for label_name, shap_dict in response.json()["shap_values"].items():
        for value in shap_dict.values():
            assert isinstance(value, float)