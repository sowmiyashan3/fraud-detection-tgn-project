import gc
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def scale_features(df, feature_cols):
    scaler = RobustScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].replace(-999, 0))
    return df, scaler
