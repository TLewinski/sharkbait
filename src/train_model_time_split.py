import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("data/training_data.csv")

print("Loaded training data:")
print("Rows:", len(df))
print()


# =========================
# 2. REMOVE YEAR
# =========================

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


# =========================
# 3. TIME-BASED SPLIT
# =========================

train_df = df[df["year"] <= 2016].copy()
test_df = df[df["year"] == 2017].copy()

X_train = train_df[features]
y_train = train_df["attack"]

X_test = test_df[features]
y_test = test_df["attack"]

print("Training years: 2011–2016")
print("Testing year: 2017")
print()

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print()
print("Training class distribution:")
print(y_train.value_counts())

print()
print("Testing class distribution:")
print(y_test.value_counts())


# =========================
# 4. TRAIN MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print()
print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# =========================
# 5. PREDICTIONS
# =========================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# =========================
# 6. EVALUATION
# =========================

print()
print("=========================")
print("TIME-BASED MODEL RESULTS")
print("=========================")

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=3,
        zero_division=0
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

# ROC-AUC requires both classes in the test set
if y_test.nunique() == 2:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print(
        "ROC-AUC:",
        round(roc_auc, 3)
    )


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

print(
    importance.to_string(
        index=False
    )
)


# =========================
# 8. SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/sharkbait_time_model.pkl"
)

print()
print("Model saved:")
print("models/sharkbait_time_model.pkl")