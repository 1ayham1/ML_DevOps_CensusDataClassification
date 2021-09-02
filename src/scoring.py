"""Calculating F1-Score of a trained model"""

import pickle
import os
import logging

from sklearn.metrics import fbeta_score, precision_score, recall_score


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

model_folder = os.path.abspath('model')


def score_model(X_in, y_val, label='testing'):
    """Function for model scoring

    Given a trained model, relevent data is loaded, and F1 score is
    then calculated.

    """

    logger.info("read and load trained model")

    model_path = os.path.join(model_folder, "trainedmodel.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    logger.info("Scoring:...........")

    preds_vals = model.predict(X_in)
    precision_tr, recall_tr, fbeta_tr = compute_model_metrics(
        y_val, preds_vals)

    print(
        f"""{label} scores....\n{'-'*50}\n
    precision: {precision_tr}\n
    recall: {recall_tr}\n
    fbeta: {fbeta_tr}\n""")

    print(f"{'='*50}")

    return precision_tr, recall_tr, fbeta_tr


def compute_model_metrics(y, preds):
    """performance evaluation!

    validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    """accuracy = sum(y == preds) / float(len(preds))"""

    return precision, recall, fbeta
