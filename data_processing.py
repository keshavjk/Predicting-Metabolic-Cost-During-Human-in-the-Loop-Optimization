"""
data_processing.py

Utilities to load and preprocess data for Predicting Metabolic Cost project.
Expected input: CSV files with one row per 30-second trial.
Required columns:
 - metabolic_rate : normalized metabolic rate (target)
 - Any feature columns: e.g., stride_time, stride_width, peak_force_L, peak_force_R, ankle_angle_max, emg_RF_L, emg_RF_R, ...

This script provides:
 - load_data(path)
 - standard preprocessing: drop NA, z-score normalization for features, baseline subtraction optional
 - feature extraction helpers (placeholders) if raw step/EMG signals are provided.

Note: The original study used per-step medians over 30s, EMG RMS with normalization, and baseline subtraction.
This code assumes precomputed step/EMG summary features in CSV format.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(csv_path, target_col='metabolic_rate', dropna=True):
    df = pd.read_csv(csv_path)
    if dropna:
        df = df.dropna().reset_index(drop=True)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV. Columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y, df

def zscore_normalize(X_train, X_val=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if X_val is None:
        return X_train_s, scaler
    else:
        return X_train_s, scaler.transform(X_val), scaler

def save_preprocessed(X, y, out_csv):
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df['metabolic_rate'] = y
    df.to_csv(out_csv, index=False)
    return out_csv

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='Input CSV with features + metabolic_rate column')
    p.add_argument('--out', default='preprocessed.csv', help='Output CSV path')
    args = p.parse_args()
    X, y, df = load_data(args.csv)
    # z-score in-place
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[df.columns.difference(['metabolic_rate'])] = scaler.fit_transform(df[df.columns.difference(['metabolic_rate'])])
    df.to_csv(args.out, index=False)
    print(f"Wrote preprocessed data to {args.out}")
