# test_app.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_predict():
    response = client.post("/predict", json={
        "x": 17172.45151,
        "y": 35519.53147,
        "area": 122.0,
        "floor_range": "06-10",
        "type_of_sale": "Resale",
        "district": 22,
        "district_name": "Boon Lay, Jurong, Tuas",
        "remaining_lease": 72.31506849315069,
        "price_index": 168.1
    })
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_invalid_input():
    response = client.post("/predict", json={
        "x": 17172.45151,
        "y": 35519.53147,
        "area": 122.0,
        "floor_range": "06-10",
        "type_of_sale": "Resale",
        "district": 22,
        "district_name": "Boon Lay, Jurong, Tuas",
        "remaining_lease": 72.31506849315069,
        # Missing price_index
    })
    assert response.status_code == 422
