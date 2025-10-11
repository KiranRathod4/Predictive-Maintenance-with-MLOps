import pandas as pd
import os

PROCESSED_DIR = "data/processed"
FEATURE_DIR = "data/features"

os.makedirs(FEATURE_DIR, exist_ok=True)

# Load processed train data
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_FD001_processed.csv"))

# Rename first two columns
train_df = train_df.rename(columns={train_df.columns[0]: "engine_id",
                                    train_df.columns[1]: "cycle"})

# Compute RUL (Remaining Useful Life)
rul = train_df.groupby("engine_id")["cycle"].max().reset_index()
rul.columns = ["engine_id", "max_cycle"]
train_df = train_df.merge(rul, on="engine_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]

# Save features
train_df.to_csv(os.path.join(FEATURE_DIR, "train_FD001_features.csv"), index=False)

print("Feature engineering done! Features saved to data/features/")
