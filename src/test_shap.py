import pandas as pd
import joblib
import shap

print("Loading SharkBait model...")

model = joblib.load(
    "models/sharkbait_time_model.pkl"
)

training_data = pd.read_csv(
    "data/training_data.csv"
)

features = [
    "month",
    "day",
    "hour",
    "latitude",
    "longitude",
    "sea_surface_temp",
    "wind_speed",
    "wave_height",
    "wave_period",
    "swell_height",
    "swell_period"
]

# Use one real row from the training data
sample = training_data[
    features
].iloc[[0]]

print()
print("Testing SHAP...")
print()
print("Input:")
print(sample)

explainer = shap.TreeExplainer(model)

shap_values = explainer(
    sample
)

print()
print("SHAP test successful!")
print()

print("SHAP values:")

values = shap_values.values

# Random Forest classification returns
# values for both classes.
# Class 1 = attack.
if values.ndim == 3:
    values = values[:, :, 1]

for feature, value in zip(
    features,
    values[0]
):
    print(
        f"{feature}: {float(value):.4f}"
    )