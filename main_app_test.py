'''unit test class for churn library

link: https://fastapi.tiangolo.com/tutorial/testing/
'''

import os
import unittest

from fastapi.testclient import TestClient

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
            "age": 40, "workclass": "State-gov",
            "education": "Doctorate", "marital-status": "Never-married",
            "occupation": "Farming-fishing", "relationship": "Unmarried",
            "race": "White", "sex": "Female",
            "hours-per-week": 40, "native-country": "United-States"
        }

        self.input_above = {
            "age": 40, "workclass": "Self-emp-not-inc",
            "education": "HS-grad", "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial", "relationship": "Husband",
            "race": "White", "sex": "Male",
            "hours-per-week": 45, "native-country": "United-States"
        }

    def test_read_main(self):
        """test fastapi get()"""

        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json() == {'message': 'Welcome... !'}

    ###################################################################
    """
    The program is working correctly from the web by runnin main:api via
    uvicorn and produce expected results with response 200 (see images
    in screenshots [FromWeb_Success_Test_Below.PNG,
    FromWeb_Success_Test_Above.PNG])
    but producess respose 422 here ???!!

    """
    ###################################################################

    def test_below_salary(self):
        """test fastapi post for inference below"""

        response = self.client.post("/", json=self.input_below)
        print("*" * 80)
        print(response)

        assert response.status_code == 200
        assert response.json() == "salary is prabaly is less than 50k"

    def test_above_salary(self):
        """test fastapi post for inference above"""

        response = self.client.post("/predict", json=self.input_above)
        print("*" * 80)
        print(response)

        assert response.status_code == 200
        assert response.json() == "salary is prabaly is above than 50k"


if __name__ == '__main__':

    unittest.main()
