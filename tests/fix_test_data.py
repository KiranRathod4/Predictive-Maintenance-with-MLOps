# Create fix_test_data.py
import pandas as pd
import json

# Load your actual data
df = pd.read_csv('data/features/train_FD001_features.csv')

# Get a sample row (first row, all features)
sample_row = df.iloc[0]

# Create sample data dictionary matching your model's expected input
sample_data = {
    "engine_id": 1,
    "cycle": 1,
    "sensor_2": -0.0007,
    "sensor_3": -0.0004,
    "sensor_4": 100.0,
    "sensor_5": 518.67,
    "sensor_6": 641.82,
    "sensor_7": 1589.7,
    "sensor_8": 1400.6,
    "sensor_9": 14.62,
    "sensor_10": 21.61,
    "sensor_11": 554.36,
    "sensor_12": 2388.06,
    "sensor_13": 9046.19,
    "sensor_14": 1.3,
    "sensor_15": 47.47,
    "sensor_16": 521.66,
    "sensor_17": 2388.02,
    "sensor_18": 8138.62,
    "sensor_19": 8.4195,
    "sensor_20": 0.03,
    "sensor_21": 392.0,
    "sensor_22": 2388.0,
    "sensor_23": 100.0,
    "sensor_24": 39.06,
    "sensor_25": 23.419,
    "max_cycle": 192
}

print("Correct sample data for tests:")
print(json.dumps(sample_data, indent=2))

# Also print with friendly names for documentation
friendly_sample = {
    "engine_id": int(sample_row['engine_id']),
    "cycle": int(sample_row['cycle']),
    "sensor_2": float(sample_row['2']),   # These map to your numeric columns
    "sensor_3": float(sample_row['3']),
    "sensor_4": float(sample_row['4']),
    "sensor_5": float(sample_row['5']),
    "sensor_6": float(sample_row['6']),
    "sensor_7": float(sample_row['7']),
    "sensor_8": float(sample_row['8']),
    "sensor_9": float(sample_row['9']),
    "sensor_10": float(sample_row['10']),
    "sensor_11": float(sample_row['11']),
    "sensor_12": float(sample_row['12']),
    "sensor_13": float(sample_row['13']),
    "sensor_14": float(sample_row['14']),
    "sensor_15": float(sample_row['15']),
    "sensor_16": float(sample_row['16']),
    "sensor_17": float(sample_row['17']),
    "sensor_18": float(sample_row['18']),
    "sensor_19": float(sample_row['19']),
    "sensor_20": float(sample_row['20']),
    "sensor_21": float(sample_row['21']),
    "sensor_22": float(sample_row['22']),
    "sensor_23": float(sample_row['23']),
    "sensor_24": float(sample_row['24']),
    "sensor_25": float(sample_row['25']),
    "max_cycle": int(sample_row['max_cycle'])
}

print("\nFriendly named sample data (for API calls):")
print(json.dumps(friendly_sample, indent=2))

with open('correct_sample_data.json', 'w') as f:
    json.dump(friendly_sample, f, indent=2)