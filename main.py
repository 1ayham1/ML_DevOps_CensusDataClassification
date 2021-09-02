"""
run with: uvicorn main:app --reload
default port: 127.0.0.1:8000
default docs: 127.0.0.1:8000/docs
"""

import os
import joblib
import pandas as pd

#give Heroku the ability to pull in data from DVC upon app start up. uses buildpack
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


#Import Union supports Item object that have tags as either strings a list.
from typing import Union 

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object with its components and their type.
class InData(BaseModel):
    Age: int
    Workclass: str
    Education: str
    Marital: str
    Occupation: str
    Relationship: str
    Race: str
    Sex: str
    Weekhour:int
    Country: str


model_path = "./src/model/trainedmodel.pkl"

model = joblib.load(model_path)

#instantiate the app
app = FastAPI()

#Define a GET on the specified endpoint
@app.get("/")
async def welcome():
    return {'message': 'Welcome... !'}

# This allows sending of data (TaggedItem) via POST to the API.
@app.post("/predict")
async def predict_item(item: InData):
    
    input_data = {
        "age":item.Age, "workclass":item.Workclass,
        "education":item.Education, "marital-status":item.Marital,
        "occupation":item.Occupation, "relationship":item.Relationship,
        "race": item.Race, "sex":item.Sex,
        "hours-per-week": item.Weekhour ,"native-country": item.Country
        }
    
    input_df = pd.DataFrame(input_data,index=[0,])

    preds_vals = model.predict(input_df)
    
    if(preds_vals[0]==0):
        output_msg = "salary is prabaly is less than 50k"
    
    else:
        output_msg = "salary is prabaly is more than 50k"
    
    return output_msg




