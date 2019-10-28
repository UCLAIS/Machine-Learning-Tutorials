import pandas as pd
import numpy as np


"""
You can either
1. Remove the row that has missing data OR
        -> pd.isna(data)
        
2. Fill the missing part with something meaningful
        such as, median value of neighbouring data, 
                 average value of the column (feature)
                 
        -> pd.fillna(data)
"""

DATA_PATH = '../../data/taxi_train.csv'

df = pd.read_csv(DATA_PATH)

print("Before removing missing data")
df.info()

print(df)

"""
@:param
dataframe: original dataframe
    :type -> class pandas.core.frame.DataFrame
    
@:returns -> dataframe which some rows have been removed
"""
def remove_missingData(dataframe):
    index2Remove = []
    # TODO: loop through features
    for feature in dataframe.keys():
        for i, data in enumerate(dataframe[feature]):
            # TODO: if one cell is empty or na, record the index in the list
            if pd.isna(data):
                index2Remove.append(i)

    # TODO: handle possible edge case: index values are not distinct
    index2Remove = list(set(index2Remove))

    # TODO: drop the rows from dataframe
    removed_data = dataframe.drop(index=index2Remove)

    return removed_data


# TODO: call the function and check df
removed_df = remove_missingData(df)
print("\nAfter removing missing data")
removed_df.info()
