"""
utils.py

Helper utilities for the repo.
"""
import numpy as np
import pandas as pd

def train_test_split_by_day(df, day_col='day', test_day=None):
    \"\"\"If the dataset contains a 'day' column, split by day to simulate unseen-day generalization.\"\"\"
    if day_col not in df.columns or test_day is None:
        raise ValueError('day column missing or test_day not provided')
    train = df[df[day_col] != test_day].reset_index(drop=True)
    test = df[df[day_col] == test_day].reset_index(drop=True)
    return train, test
