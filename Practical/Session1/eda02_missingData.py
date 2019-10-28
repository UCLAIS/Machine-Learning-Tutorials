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
    for feature in None:
        # enumerate through rows of a certain feature
        # enumerate(data) "yields" tuple of iteration number(index) and element
        # unpacking tuple into i and data
        for i, data in enumerate(dataframe[feature]):
            # TODO: if "one cell" is empty or na, record the index in the list
            if None:
                index2Remove.append(i)

    # TODO: handle possible edge case: "index values are not distinct"
    # write your code that makes index2Remove have distinct values (classical algorithm question innit)
    # let's see who's got the best solution!!

    # TODO: drop the rows from original dataframe and store the changed df into variable 'removed_data', which you will return eventually.
    removed_data = None

    return removed_data


# calling the function and check df
removed_df = remove_missingData(df)
print("\nAfter removing missing data")
removed_df.info()
