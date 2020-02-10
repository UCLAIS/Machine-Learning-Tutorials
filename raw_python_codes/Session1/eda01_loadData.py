# new york taxi dataset EDA
# data sourced from: https://www.kaggle.com/c/nyc-taxi-trip-duration/data

import pandas as pd
import numpy as np

# Relational data path -> in this case -> two level up from current directory
# Absolute data path -> import os and get absolute path (gives the same result as 'pwd' Unix command)
TRAIN_DATA_PATH = '../../data/taxi_train.csv'

"""
@:param
path: relational or absolute data path
      can use static variable DATA_PATH
"""
def load_csv(path):
    # TODO: load taxi train dataset using pandas
    data_Frame = pd.read_csv(path)
    return data_Frame


"""
@:param
data: Target data you want to know about in detail
      type(data) -> pandas.Series (i.e. one feature)
"""
def statistical_features(data):
    # TODO: let's extract some useful statistical information about the data using numpy
    _min = np.min(data)
    _max = np.max(data)
    _mean = np.mean(data)
    _median = np.median(data)
    _var = np.var(data)
    _std = np.std(data)

    return _min, _max, _mean, _median, _var, _std


df = load_csv(TRAIN_DATA_PATH)
print(df)
print(type(df))

# TODO: Whole overview of the data (summarised information)
print("info: ")
df.info()
print('\n')

print("head: ")
print(df.head(n=10),'\n')

print("description: ")
print(df.describe(), '\n')

# TODO: statistical data about 'trip_duration'
_min, _max, _mean, _median, _var, _std = statistical_features(df['trip_duration'])
print('min:', _min, 'max:', _max, 'average:', _mean, 'median:', _median, 'variation:', _var, 'standard deviation:', _std)

# TODO: statistical data about 'passenger_count'
_min, _max, _mean, _median, _var, _std = statistical_features(df['passenger_count'])
print('min:', _min, 'max:', _max, 'average:', _mean, 'median:', _median, 'variation:', _var, 'standard deviation:', _std)
