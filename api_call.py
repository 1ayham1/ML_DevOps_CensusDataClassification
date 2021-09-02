import requests
import json

data = {
    "name": "I don't know",
    "tags": [
        'worst course',
        'unbelivable'],
    "item_id": 50}

r = requests.post("http://127.0.0.1:8000/items/", data=json.dumps(data))

# print returned object
print(r.json())
