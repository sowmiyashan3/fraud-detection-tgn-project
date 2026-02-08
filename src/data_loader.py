import pandas as pd
from .config import DATA_PATH, TRANSACTION_COLUMNS, IDENTITY_COLUMNS, OUT_PATH
from .utils import cleanup_memory
import os


def load_transaction_data(sample_frac=0.4):
    merged_file = os.path.join(OUT_PATH, "merged_transactions.csv")
    if os.path.exists(merged_file):
        print(f"Loading cached merged transactions: {merged_file}")
        df = pd.read_csv(merged_file)
        return df

    print("\nSmart loading (sample_frac={})".format(sample_frac))
    df_list = []
    chunk_size = 100000

    for chunk in pd.read_csv(DATA_PATH + "train_transaction.csv",
                             usecols=TRANSACTION_COLUMNS,
                             chunksize=chunk_size):
        fraud = chunk[chunk['isFraud'] == 1]
        non_fraud = chunk[chunk['isFraud'] == 0].sample(frac=sample_frac, random_state=42)
        sampled = pd.concat([fraud, non_fraud])
        df_list.append(sampled)

    transaction_df = pd.concat(df_list, ignore_index=True)
    del df_list
    cleanup_memory()
    return transaction_df

def load_identity_data():
    identity_df = pd.read_csv(DATA_PATH + "train_identity.csv", usecols=IDENTITY_COLUMNS)
    return identity_df

def merge_data(transaction_df, identity_df):
    merged_file = os.path.join(OUT_PATH, "merged_transactions.csv")
    if os.path.exists(merged_file):
        print(f"Merged file already exists: {merged_file}")
        return pd.read_csv(merged_file)

    df = transaction_df.merge(identity_df, on="TransactionID", how="left")
    df.to_csv(merged_file, index=False)
    cleanup_memory()
    return df
