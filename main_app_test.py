'''unit test class for churn library

link: https://fastapi.tiangolo.com/tutorial/testing/
'''

import time
import os
import unittest


from fastapi.testclient import TestClient

import pandas as pd
from functools import wraps

from main import app



data_folder = '../data/'
data_path = os.path.join(data_folder, 'census.csv')
save_name = os.path.join(data_folder, 'clean_census_data.csv')


class TestingAndLogging(unittest.TestCase):
    '''perform unit tests for all functions in churn library'''

    def setUp(self):
        '''prepare parameters to be used in the test'''
        self.client = TestClient(app)
        self.input_below = {
            "age":40, "workclass":"State-gov",
            "education":"Doctorate", "marital-status":"Never-married",
            "occupation":"Farming-fishing", "relationship":"Unmarried",
            "race": "White", "sex":"Female",
            "hours-per-week": 40 ,"native-country": "United-States"
            }
        
    """
    def test_read_main(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json() == {'message': 'Welcome... !'}
    """
    
    def test_below_salary(self):
        response = self.client.post("/predict",json=self.input_below)
        print("*"*80)
        print(self.input_below)
        print(response)
        assert response.status_code == 200
        assert response.json() == "salary is prabaly is less than 50k"


if __name__ == '__main__':

    unittest.main()

  