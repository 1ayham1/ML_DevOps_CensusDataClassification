""" FastApi app to make ML inference

run from command line: uvicorn main:app --reload
default port from browser: 127.0.0.1:8000
default docs: 127.0.0.1:8000/docs
"""

import os
import sys
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd

# give Heroku the ability to pull in data from DVC upon app start up.
# uses buildpack
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        sys.exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class InData(BaseModel):
    """data definition to be used in the app."""

    age: int = Field(..., example=45)
    workclass: str = Field(..., example="State-gov", )
    education: str = Field(..., example="Doctorate")
    mstatus: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Farming-fishing")
    relationship: str = Field(..., example="Unmarried")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    hpw: int = Field(..., example=20, alias="hours-per-week")
    country: str = Field(..., example="United-States", alias="native-country")


model_path = './src/model/'
print("?" * 50)
print(model_path)
model_name = os.path.join(model_path, "trainedmodel.pkl")

model = joblib.load(model_name)

# instantiate the app
app = FastAPI()


@app.get("/")
async def welcome():
    return {'message': 'Welcome... !'}


@app.post("/")
async def predict_item(item: InData):
    """ perform inference

    consider improving by:
    item_dict = item.dict()
    for key,val in input_data.items():
        item.dict.update({key:val})
    """

    input_data = {
        "age": item.age, "workclass": item.workclass,
        "education": item.education, "marital-status": item.mstatus,
        "occupation": item.occupation, "relationship": item.relationship,
        "race": item.race, "sex": item.sex,
        "hours-per-week": item.hpw, "native-country": item.country
    }

    input_df = pd.DataFrame(input_data, index=[0, ])

    preds_vals = model.predict(input_df)

    if preds_vals[0] == 0:
        output_msg = "salary is prabaly is less than 50k"

    else:
        output_msg = "salary is prabaly is more than 50k"

    return output_msg
