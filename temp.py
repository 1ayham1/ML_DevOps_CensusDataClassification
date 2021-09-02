"""quick test for deployed ml model"""
import joblib
import pandas as pd

model_path = "./src/model/trainedmodel.pkl"
model = joblib.load(model_path)

"""
col_names = [
    "age","workclass","education","marital-status",
    "occupation","relationship","race","sex",
    "hours-per-week","native-country"]

input_df = pd.DataFrame([input_data])

input_df = pd.DataFrame(lst, columns =col_names)
"""

input_data = {
    "age": 40, "workclass": 'State-gov',
    "education": 'Doctorate', "marital-status": 'Never-married',
    "occupation": 'Farming-fishing', "relationship": 'Unmarried',
    "race": 'White', "sex": 'Female',
    "hours-per-week": 40, "native-country": 'United-States'
}

input_df = pd.DataFrame(input_data, index=[0, ])

preds_vals = model.predict(input_df)
if(preds_vals[0] == 0):
    print("salary is prabaly is less than 50k")
else:
    print("salary is prabaly is more than 50k")
