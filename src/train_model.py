"""Script to train machine learning model.

 predict person's income [>50k or <50k] using various features
 {age, education, marital-status, relatoinship, race, sex}
"""

import os
import pickle

import logging
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

model_path = os.path.abspath('model')
model_name = os.path.join(model_path, "trainedmodel.pkl")


def data_split():
    """returns data {X,y} splitted in training and testing"""

    df_data = pd.read_csv("../data/clean_census_data.csv")

    X_data = df_data.drop('salary', axis=1)
    y_response = df_data['salary']

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_response,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y_response)

    return X_train, X_test, y_train, y_test


def get_inference_pipeline():
    """classify and preprocess data. return inference pipeline"""
    """
    ordinal_categorical = ["education","occupation"]

    ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(),
    )
    """

    non_ordinal_categorical = [
        "education", "occupation", "sex", "marital-status",
        "relationship", "race", "native-country"]

    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(),
    )

    numeric_cols = ["age", "hours-per-week"]
    numeric_cols_preproc = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )

    # put everything together
    preprocessor = ColumnTransformer(
        transformers=[
            ("non_ordinal_cat",
             non_ordinal_categorical_preproc,
             non_ordinal_categorical),
            ("impute_numer", numeric_cols_preproc, numeric_cols),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    clf_svm = SVC(random_state=42, kernel='rbf')

    logger.info("Searching Parameters............")
    param_grid = {
        'C': [1.5], 'gamma': [2, 10],
        'kernel': ['rbf']}

    grid_svm = GridSearchCV(clf_svm, param_grid, refit=True, verbose=2)

    # Use the explicit Pipeline constructor so you can assign the names to the
    # steps, do not use make_pipeline
    sk_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("random_forest", grid_svm),
    ])

    return sk_pipe


def train_models(train_from_scratch=True):
    '''Run model inferences and print out scores

    train_from_scratch: True to train a model
    '''

    logger.info("reading and processing data")
    X_train, X_test, y_train, y_test = data_split()

    if train_from_scratch:

        logger.info("Training a new model............")

        model = get_inference_pipeline()

        logger.info("fitting model..........")
        model.fit(X_train, y_train)

        logger.info("saving the model............")
        pickle.dump(model, open(model_name, 'wb'))

    else:
        logger.info("loading saved model............")

        model = joblib.load(model_name)

    logger.info("Model Evaluation............\n")

    scoring.score_model(X_train, y_train, label='training')
    scoring.score_model(X_test, y_test, label='testing')


if __name__ == "__main__":
    train_models(train_from_scratch=False)
