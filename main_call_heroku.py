"""Call remote app deployed on heroku locally"""

import requests


input_below = {
    "age": 40, "workclass": "State-gov",
    "education": "Doctorate", "marital-status": "Never-married",
    "occupation": "Farming-fishing", "relationship": "Unmarried",
    "race": "White", "sex": "Female",
    "hours-per-week": 40, "native-country": "United-States"
}

###############################
"""Again response 422?????"""
###############################
response = requests.post(
    "https://prediction-demo.herokuapp.com/",
    json=input_below)

print(response)

assert response.status_code == 200
assert response.json() == "salary is prabaly is above than 50k"
