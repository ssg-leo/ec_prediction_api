# EC Price Prediction API
This repository provides a FastAPI application for the EC Price Prediction model, which processes input data and returns a predicted EC price per square meter (psm).

## Installation
1. Copy the production.dvc from the model repository
```bash
bash copy_production_dvc.sh
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Configure DVC to work with your AWS account:
```bash
dvc remote modify myremote access_key_id <your_aws_access_key_id>
dvc remote modify myremote secret_access_key <your_aws_secret_access_key>
```
OR authorise using AWS CLI
4. Pull the latest model artifact using DVC:
```bash
dvc pull
```
## Usage
1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```
The API should now be accessible at http://localhost:8000.

2. Send a POST request to the /predict endpoint with the required input data:
```bash
curl -X POST "http://localhost:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x": 17172.45151, "y": 35519.53147, "area": 122.0, "floor_range": "06-10", "type_of_sale": "Resale", "district": 22, "district_name": "Boon Lay, Jurong, Tuas", "remaining_lease": 72.31506849315069, "price_index": 168.1}'
```
The API should return a JSON object containing the predicted EC price per square meter (psm) and the SHAP values:

```json
{
  "prediction": 9836.065573770491,
  "shap_values": {
    "prediction": {
      "x": 12.3456789,
      "y": 23.4567890,
      "area": 34.5678901,
      "floor_range": 45.6789012,
      "type_of_sale": 56.7890123,
      "district": 67.8901234,
      "district_name": 78.9012345,
      "remaining_lease": 89.0123456,
      "price_index": 90.1234567
    }
  }
}
```

## Testing
To execute tests for the API, run the following command:
```bash
pytest
```