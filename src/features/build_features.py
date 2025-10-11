import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load train data
train_file = os.path.join(RAW_DIR, "train_FD001.txt")
train_df = pd.read_csv(train_file, sep=" ", header=None, engine='python')
train_df = train_df.dropna(axis=1, how='all')  # remove empty columns

# Save processed train data
train_df.to_csv(os.path.join(PROCESSED_DIR, "train_FD001_processed.csv"), index=False)

# Load test data
test_file = os.path.join(RAW_DIR, "test_FD001.txt")
test_df = pd.read_csv(test_file, sep=" ", header=None, engine='python')
test_df = test_df.dropna(axis=1, how='all')  # remove empty columns

# Save processed test data
test_df.to_csv(os.path.join(PROCESSED_DIR, "test_FD001_processed.csv"), index=False)

print("Preprocessing done! Processed files saved to data/processed/")
