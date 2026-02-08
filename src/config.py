import os

# Paths
DATA_PATH = "fraud-detection-tgn-project/data/raw/"
OUT_PATH = "fraud-detection-tgn-project/data/processed/"
os.makedirs(OUT_PATH, exist_ok=True)

# Columns
TRANSACTION_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "ProductCD",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6",
    "isFraud"
]

IDENTITY_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo"]

FEATURE_COLS = [
    'TransactionAmt', 'amt_log', 'amt_decimal', 'amt_round',
    'card1_count', 'card1_velocity',
    'addr1_count', 'addr1_velocity',
    'device_count', 'email_count',
    'card_email_count', 'card_device_count',
    'hour', 'day_of_week', 'is_night', 'is_weekend',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C_sum', 'C_mean',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D1_D2_ratio', 'D3_D4_ratio'
]
