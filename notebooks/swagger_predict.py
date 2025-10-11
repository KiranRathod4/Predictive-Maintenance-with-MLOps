import joblib
model = joblib.load("models/model.pkl")
print(model.feature_names_in_)
