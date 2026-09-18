import pandas as pd
import joblib


# =========================
# 1. LOAD MODEL
# =========================

model = joblib.load(
    "models/sharkbait_time_model.pkl"
)


# =========================
# 2. GET USER INPUT
# =========================

print("==============================")
print("       SHARKBAIT")
print("   Shark Risk Predictor")
print("==============================")
print()

latitude = float(input("Latitude: "))
longitude = float(input("Longitude: "))

month = int(input("Month (1-12): "))
day = int(input("Day (1-31): "))
hour = int(input("Hour (0-23): "))

sea_surface_temp = float(
    input("Sea surface temperature (°C): ")
)

wind_speed = float(
    input("Wind speed: ")
)

wave_height = float(
    input("Wave height: ")
)

wave_period = float(
    input("Wave period: ")
)

swell_height = float(
    input("Swell height: ")
)

swell_period = float(
    input("Swell period: ")
)


# =========================
# 3. CREATE INPUT DATA
# =========================

data = pd.DataFrame([{
    "month": month,
    "day": day,
    "hour": hour,
    "latitude": latitude,
    "longitude": longitude,
    "sea_surface_temp": sea_surface_temp,
    "wind_speed": wind_speed,
    "wave_height": wave_height,
    "wave_period": wave_period,
    "swell_height": swell_height,
    "swell_period": swell_period
}])


# =========================
# 4. PREDICT
# =========================

probability = model.predict_proba(data)[0][1]

risk_score = round(probability * 100)


# =========================
# 5. RISK LEVEL
# =========================

if risk_score < 25:
    risk_level = "LOW"

elif risk_score < 50:
    risk_level = "MODERATE"

elif risk_score < 75:
    risk_level = "HIGH"

else:
    risk_level = "VERY HIGH"


# =========================
# 6. DISPLAY RESULT
# =========================

print()
print("==============================")
print("       SHARKBAIT RESULT")
print("==============================")
print()

print(
    f"Risk Score: {risk_score} / 100"
)

print(
    f"Risk Level: {risk_level}"
)

print()

print(
    "Model attack score:",
    round(probability, 3)
)

print()
print("==============================")