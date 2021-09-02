"""read input data and save out a cleaned version"""

import logging
import os
import pandas as pd
import numpy as np


data_folder = '../data/'

data_path = os.path.join(data_folder, 'census.csv')
save_name = os.path.join(data_folder, 'clean_census_data.csv')

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def read_and_clean():
    """read input csv file and save cleaned version"""

    logger.info("reading input data...")
    df = pd.read_csv(data_path, skipinitialspace=True)

    logger.info("cleaning...")
    df_clean = df.copy(deep=True)

    df_clean.drop_duplicates(inplace=True)
    df_clean.replace({'?': np.nan}, inplace=True)
    df_clean.dropna(
        axis='index',
        how='any',
        inplace=True)  # to be imputed later
    df_clean.drop("capital-gain", axis=1, inplace=True)
    df_clean.drop("capital-loss", axis=1, inplace=True)

    # to prepared data for prediction
    df_clean.salary = df_clean.salary.replace('<=50K', 0)
    df_clean.salary = df_clean.salary.replace('>50K', 1)

    # further
    df_clean.drop("education-num", axis=1, inplace=True)  # encoding eduction
    # irrelevent feture: final weight
    df_clean.drop("fnlgt", axis=1, inplace=True)

    logger.info("save output...")
    df_clean.to_csv(save_name, index=False)


if __name__ == "__main__":
    read_and_clean()
