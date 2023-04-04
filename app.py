import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Load the optimized pipeline and explainer when the application starts
pipeline = joblib.load("production/best_pipeline.pkl")
explainer = joblib.load("production/explainer.pkl")


class InputData(BaseModel):
    x: float
    y: float
    area: float
    floor_range: str
    type_of_sale: str
    district: int
    district_name: str
    remaining_lease: float
    price_index: Optional[float]


def validate_input(input_data: InputData):
    # Add any necessary validation logic here
    return input_data


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/")
async def predict(input_data: InputData):
    # Convert the input data to a DataFrame
    instance = pd.DataFrame([dict(input_data)])
    instance.columns = ['x', 'y', 'area', 'floor_range', 'type_of_sale', 'district',
                        'district_name', 'remaining_lease', 'price_index']

    data_transformed = pipeline.named_steps["mean_encoder"].transform(instance)
    instance_transformed = pipeline.named_steps["scaler"].transform(data_transformed)
    instance_transformed = pd.DataFrame(instance_transformed)
    instance_transformed.columns = ['x', 'y', 'area', 'floor_range', 'type_of_sale', 'district',
                                    'district_name', 'remaining_lease', 'price_index']

    # Perform a prediction using the pipeline
    prediction = pipeline.predict(instance_transformed)

    # Compute SHAP values using the explainer
    shap_values = explainer(instance_transformed)
    feature_names = instance_transformed.columns
    label_name = "prediction"
    shap_values_dict = {}
    label_name = "prediction"
    shap_values_dict = {}
    label_name = "prediction"
    shap_values_for_label = shap_values[0].values
    shap_dict = {feature_names[i]: float(shap_values_for_label[i]) for i in range(len(feature_names))}
    shap_values_dict[label_name] = shap_dict

    # Convert the prediction and SHAP values to a nested dictionary
    response_content = {"prediction": float(prediction[0]), "shap_values": shap_values_dict}

    return response_content
