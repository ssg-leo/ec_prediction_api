# app.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

# import my_model_package

app = FastAPI()


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


# @app.post("/predict")
# def predict(input_data: InputData = Depends(validate_input)):
#     prediction = my_model_package.predict(np.array([input_data.dict().values()]))
#     return {"prediction": prediction[0]}
