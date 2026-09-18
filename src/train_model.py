import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.impute import SimpleImputer
import joblib


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("data/training_data.csv")

print("Loaded training data:")
print("Rows:", len(df))
print()


# =========================
# 2. SELECT FEATURES
# =========================

features = [
    "year",
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

X = df[features]
y = df["attack"]


# =========================
# 3. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print()


# =========================
# 4. TRAIN RANDOM FOREST
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")
print()


# =========================
# 5. MAKE PREDICTIONS
# =========================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# =========================
# 6. EVALUATE MODEL
# =========================

print("=========================")
print("MODEL RESULTS")
print("=========================")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=3
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print()

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("ROC-AUC:", round(roc_auc, 3))


# =========================
# 7. FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("Feature Importance:")
print(importance.to_string(index=False))


# =========================
# 8. SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/sharkbait_model.pkl"
)

print()
print("Model saved:")
print("models/sharkbait_model.pkl")