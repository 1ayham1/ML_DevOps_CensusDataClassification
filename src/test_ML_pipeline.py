'''unit test class for churn library'''

import time
import os
import logging
import unittest

import pandas as pd
from functools import wraps

import train_model
import data_ingestion


data_folder = os.path.abspath("./data/")

data_path = os.path.join(data_folder, 'census.csv')
save_name = os.path.join(data_folder, 'clean_census_data.csv')

# to suppress other logging
# https://stackoverflow.com/questions/35898160/logging-module-not-writing-to-file

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

dirname = 'ML_DevOps_CensusDataClassification/src/logs/'
log_name = os.path.join(dirname, 'TDD_cases.log')


logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    filename=log_name,
    filemode='w',
)

logger = logging.getLogger()


def get_time(function):
    '''
    wrapper to return execution time of a function
    '''
    @wraps(function)
    def wrapper(*args, **kwargs):
        t_start = time.time()
        run_fun = function(*args, **kwargs)
        t_end = time.time() - t_start
        logger.info('%s ran in %0.3f sec ', function.__name__, t_end)
        logger.info('---------------------------------------------------')
        # logger.info(f'{"-"*60}') #does not use lazy formatting

        return run_fun
    return wrapper


class TestingAndLogging(unittest.TestCase):
    '''perform unit tests for all functions in churn library'''

    def setUp(self):
        '''prepare parameters to be used in the test'''

        self.df = pd.read_csv(data_path, skipinitialspace=True)

    @get_time
    def test_import(self):
        ''' test data import and '''

        try:
            df = pd.read_csv(data_path, skipinitialspace=True)
            logger.info("Testing import_data: SUCCESS")

        except FileNotFoundError as err:
            logger.error("Testing import_data: file wasn't found")
            raise err

        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
            logger.info(
                'input data has %d rows and %d columns',
                df.shape[0],
                df.shape[1])

        except AssertionError as err:
            logger.error(
                "Test import_data:file doesn't appear to have rows & columns")
            raise err

    @get_time
    def test_read_and_clean(self):
        '''test read_and_clean'''

        try:
            # modify later to pass a file name
            data_ingestion.read_and_clean()

        except Exception as err:
            logger.error(
                "someting is wrong with read_and_clean()")
            raise err

    @get_time
    def test_train_models(self):
        '''test train_models'''

        try:
            train_model.train_models(train_from_scratch=False)
            logger.info("train_models ran successfully!")

        except Exception as err:
            logger.error("someting is wrong with train_models()")
            raise err


if __name__ == '__main__':

    unittest.main()
