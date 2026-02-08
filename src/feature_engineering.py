import numpy as np
import pandas as pd
from .config import FEATURE_COLS, OUT_PATH
from .utils import cleanup_memory, scale_features
import os


def engineer_features(df):

    features_file = os.path.join(OUT_PATH, "features_df.csv")

    if os.path.exists(features_file):
        print(f"Loading cached features: {features_file}")
        df = pd.read_csv(features_file)
        scaler = RobustScaler()  # Dummy scaler for compatibility
        return df, scaler

    # Fill missing
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna("Unknown", inplace=True)

    for col in df.select_dtypes(include=['number']).columns:
        if col not in ['TransactionID', 'TransactionDT', 'isFraud']:
            df[col].fillna(-999, inplace=True)

    # Device node
    df["device_node"] = "dev_" + df["DeviceType"].astype(str)

    # Sort by time
    df.sort_values("TransactionDT", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Velocity features
    df['card1_count'] = df.groupby('card1').cumcount()
    df['card1_velocity'] = df.groupby('card1')['TransactionDT'].diff().fillna(0) / 3600
    df['addr1_count'] = df.groupby('addr1').cumcount()
    df['addr1_velocity'] = df.groupby('addr1')['TransactionDT'].diff().fillna(0) / 3600
    df['device_count'] = df.groupby('device_node').cumcount()
    df['email_count'] = df.groupby('P_emaildomain').cumcount()

    # Aggregation features
    df['card_email'] = df['card1'].astype(str) + "_" + df['P_emaildomain'].astype(str)
    df['card_email_count'] = df.groupby('card_email').cumcount()
    df['card_device'] = df['card1'].astype(str) + "_" + df['device_node'].astype(str)
    df['card_device_count'] = df.groupby('card_device').cumcount()

    # Time features
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // 86400) % 7
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Amount features
    df['amt_log'] = np.log1p(df['TransactionAmt'])
    df['amt_decimal'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)
    df['amt_round'] = (df['amt_decimal'] < 0.01).astype(int)

    # D-column features
    df['D1_D2_ratio'] = np.where(df['D2'] != 0, df['D1'] / df['D2'], 0)
    df['D3_D4_ratio'] = np.where(df['D4'] != 0, df['D3'] / df['D4'], 0)

    # C-column features
    c_cols = [f'C{i}' for i in range(1, 15)]
    df['C_sum'] = df[c_cols].sum(axis=1)
    df['C_mean'] = df[c_cols].mean(axis=1)

    df, scaler = scale_features(df, FEATURE_COLS)
    df.to_csv(features_file, index=False)
    cleanup_memory()
    return df, scaler
