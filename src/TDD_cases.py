'''unit test class for churn library'''
import time
import logging
import unittest

from pandas.api.types import is_string_dtype
from functools import wraps

import train_model
import data_ingestion



data_folder = '../data/'
data_path = os.path.join(data_folder, 'census.csv')
save_name = os.path.join(data_folder, 'clean_census_data.csv')

keep_cols = [
    'Customer_Age',
    ]

logging.basicConfig(
    filename='./logs/TDD_cases.log',
    level=logging.INFO,
    # filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def get_time(function):
    '''
    wrapper to return execution time of a function
    '''
    @wraps(function)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        run_fun = function(*args, **kwargs)
        t_end = time.time() - t_start
        logging.info('%s ran in %0.3f sec ', function.__name__, t_end)
        logging.info('---------------------------------------------------')
        # logging.info(f'{"-"*60}') #does not use lazy formatting

        return run_fun
    return wrapper


class TestingAndLogging(unittest.TestCase):
    '''perform unit tests for all functions in churn library'''

    def setUp(self):
        '''prepare parameters to be used in the test'''
      # needs refactoring so that churn class asserts read file correctness
        self.df = pd.read_csv(data_path, skipinitialspace=True)
    
    @get_time
    def test_import(self):
        ''' test data import'''

        try:
            df = pd.read_csv(data_path, skipinitialspace=True)
            logging.info("Testing import_data: SUCCESS")

        except FileNotFoundError as err:
            logging.error("Testing import_data: file wasn't found")
            raise err

        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
            logging.info(
                'input data has %d rows and %d columns',
                df.shape[0],
                df.shape[1])

        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    @get_time
    def test_read_and_clean(self):
        '''test read_and_clean'''

        try:
            #modify later to pass a file name
            data_ingestion.read_and_clean()
            
        except Exception as err:
            logging.error(
                "someting is wrong with read_and_clean()")
            raise err

    @get_time
    def test_train_models(self):
        '''test train_models'''

        try:
            train_model.train_models(train_from_scratch=False)
            logging.info("train_models ran successfully!")

        except Exception as err:
            logging.error("someting is wrong with train_models()")
            raise err


if __name__ == '__main__':

    unittest.main()

  