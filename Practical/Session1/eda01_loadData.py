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
      
@:returns
dataframe
"""
def load_csv(path):
    # TODO: load taxi train dataset using pandas
    pass


"""
@:param
data: Target data you want to know about in detail
      type(data) -> pandas.Series (i.e. one feature)
      
@:returns 6
min, max, mean, median, var, std values
"""
def statistical_features(data):
    # TODO: let's extract some useful statistical information about the data using numpy
    _min = None
    _max = None
    _mean = None
    _median = None
    _var = None
    _std = None

    return _min, _max, _mean, _median, _var, _std


df = load_csv(TRAIN_DATA_PATH)
print(df)
print(type(df))

# TODO: Whole overview of the data (summarised information)
print("df info: ")
# get info
print('\n')

print("df's first 10 heads: ")
print(None,'\n')

print("description of df: ")
print(None, '\n')

# TODO: statistical data about 'trip_duration'
_min, _max, _mean, _median, _var, _std = statistical_features(None)
print('min:', _min, 'max:', _max, 'average:', _mean, 'median:', _median, 'variation:', _var, 'standard deviation:', _std)

# TODO: statistical data about 'passenger_count'
_min, _max, _mean, _median, _var, _std = statistical_features(None)
print('min:', _min, 'max:', _max, 'average:', _mean, 'median:', _median, 'variation:', _var, 'standard deviation:', _std)
