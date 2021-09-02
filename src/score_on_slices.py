"""computes performance on model slices.

for given categorical variable computes the metrics when
its value is held fixed.
"""

import pandas as pd
from pandas.api.types import is_string_dtype

import train_model
import scoring


data_path = "../data/clean_census_data.csv"
save_score_path = "./model/slice_output.txt"


def slice_scoring():
    """computes the performance metrics when the value of a feature is held fixed.

    prints out the model metrics for each value.
    Output the printout to a file named slice_output.txt in model folder.
    """

    df_data = pd.read_csv(data_path)
    categorical_features = [
        col for col in df_data if is_string_dtype(
            df_data[col])]

    # FIX: not a good design. refactor to pass data frame
    _, X_test, _, y_test = train_model.data_split()

    merg_df = X_test.copy(deep=True)
    merg_df['response'] = y_test

    all_scores = []

    for cat_feature in categorical_features:
        unique_clases = merg_df[cat_feature].unique()

        for u_class in unique_clases:

            df_of_interest = merg_df[merg_df[cat_feature] == u_class]

            # extract related X,y
            X_data = df_of_interest.drop('response', axis=1)
            y_response = df_of_interest['response']

            precision, recall, fbeta = scoring.score_model(X_data, y_response)

            saved_score = (
                f"Category: {cat_feature}\nFixed Feature: {u_class}\n"
                f"{'-'*80}\n"
                f"\tprecision:{precision: .3f}\n"
                f"\trecall:{recall: .3f}\n"
                f"\tfbeta:{fbeta: .3f}\n"
                f"{'*'*80}\n"
            )

            all_scores.append(saved_score)

    with open(save_score_path, 'w') as score_file:
        for score_output in all_scores:
            score_file.write(score_output + '\n')


if __name__ == "__main__":
    slice_scoring()
