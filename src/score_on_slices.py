"""computes performance on model slices.

for given categorical variable computes the metrics when its value is held fixed.
"""

import pandas as pd
from pandas.api.types import is_string_dtype

#ordinal_categorical = ["education","occupation"]
#categorical_features = []

def slice_scoring():
    """computes the performance metrics when the value of a feature is held fixed. 
    
    prints out the model metrics for each value.
    Output the printout to a file named slice_output.txt.
    """
    
    df_data = pd.read_csv("../data/clean_census_data.csv")
    categorical_features = [col for col in df_data if is_string_dtype(df_data[col])]

    print(categorical_features)

if __name__ == "__main__":
    slice_scoring()
