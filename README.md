# EC Price Prediction API

This repository contains a FastAPI application for the EC Price Prediction model. It accepts input data and returns a prediction of the EC price per square meter (psm).

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/ec_price_prediction_api.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the EC Price Prediction model package:
```bash
pip install git+https://github.com/yourusername/ec_price_prediction_model.git
```

## Usage
1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```
The API should now be running on http://localhost:8000.

2. Make a POST request to the /predict endpoint with the required input data:
```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x": 17172.45151, "y": 35519.53147, "area": 122.0, "floor_range": "06-10", "type_of_sale": "Resale", "district": 22, "district_name": "Boon Lay, Jurong, Tuas", "remaining_lease": 72.31506849315069, "price_index": 168.1}'
```
The API should return a JSON object containing the predicted EC price per square meter (psm):
```json
{
  "prediction": 9836.065573770491
}
```

## Testing
To run tests for the API, execute the following command:

```bash
pytest test_app.py
```