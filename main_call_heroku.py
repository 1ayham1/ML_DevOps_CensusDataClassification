"""Call remote app deployed on heroku locally"""

import requests


input_below = {
    "age": 40, "workclass": "State-gov",
    "education": "Doctorate", "marital-status": "Never-married",
    "occupation": "Farming-fishing", "relationship": "Unmarried",
    "race": "White", "sex": "Female",
    "hours-per-week": 40, "native-country": "United-States"
}

response = requests.post(
    "https://prediction-demo.herokuapp.com/",
    json=input_below)



assert response.status_code == 200
assert response.json() == "salary is prabaly is less than 50k"
print('-'*50)
print(f"post response: {response.status_code}")
print(f"post output:\n {response.json()}\n")
print('='*50)