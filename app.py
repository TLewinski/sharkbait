import streamlit as st
import pandas as pd
import joblib
import sys
import requests

# Allow importing files from src/
sys.path.append("src")

from realtime_predict import (
    get_stations,
    find_best_station,
    get_value
)

st.set_page_config(
    page_title="SharkBait",
    page_icon="🦈",
    layout="centered"
)

# Load model and training data
model = joblib.load(
    "models/sharkbait_time_model.pkl"
)

training_data = pd.read_csv(
    "data/training_data.csv"
)


# -----------------------------
# GEOCODE LOCATION
# -----------------------------

def geocode_location(location):
    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "SharkBait/1.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        return None

    return {
        "latitude": float(results[0]["lat"]),
        "longitude": float(results[0]["lon"]),
        "display_name": results[0]["display_name"]
    }


# -----------------------------
# APP
# -----------------------------

st.title("🦈 SharkBait")

st.subheader(
    "Real-Time Shark Risk Predictor"
)

st.write(
    "Enter a beach, city, or coastal location "
    "to analyze current marine conditions and "
    "estimate shark risk."
)


# -----------------------------
# LOCATION INPUT
# -----------------------------

st.markdown("### Location")

location = st.text_input(
    "Enter a location",
    placeholder="Example: Daytona Beach, Florida"
)

predict_button = st.button(
    "Analyze Shark Risk",
    type="primary"
)


# -----------------------------
# RUN PREDICTION
# -----------------------------

if predict_button:

    if not location.strip():
        st.warning(
            "Please enter a location."
        )
        st.stop()

    # Geocode
    with st.spinner(
        "Finding your location..."
    ):
        try:
            coordinates = geocode_location(
                location
            )

        except Exception as error:
            st.error(
                f"Unable to find location: {error}"
            )
            st.stop()

    if coordinates is None:
        st.error(
            "Location not found. "
            "Try including the city and state/country."
        )
        st.stop()

    latitude = coordinates["latitude"]
    longitude = coordinates["longitude"]

    st.success(
        f"Location found: "
        f"{coordinates['display_name']}"
    )

    st.markdown(
        f"**Coordinates:** "
        f"{latitude:.4f}, {longitude:.4f}"
    )


    # -----------------------------
    # NOAA STATION
    # -----------------------------

    with st.spinner(
        "Finding the nearest NOAA station..."
    ):
        try:
            stations = get_stations()

            station, observation = find_best_station(
                latitude,
                longitude,
                stations
            )

        except Exception as error:
            st.error(
                f"Unable to retrieve NOAA data: {error}"
            )
            st.stop()


    if station is None or observation is None:
        st.error(
            "No NOAA station with available "
            "real-time data was found."
        )
        st.stop()


    station_id = station["station"]


    # -----------------------------
    # LIVE NOAA CONDITIONS
    # -----------------------------

    sea_surface_temp = get_value(
        observation,
        "WTMP"
    )

    wind_speed = get_value(
        observation,
        "WSPD"
    )

    wave_height = get_value(
        observation,
        "WVHT"
    )

    wave_period = get_value(
        observation,
        "DPD"
    )


    # -----------------------------
    # HANDLE MISSING DATA
    # -----------------------------

    values = {
        "sea_surface_temp": sea_surface_temp,
        "wind_speed": wind_speed,
        "wave_height": wave_height,
        "wave_period": wave_period,
        "swell_height": None,
        "swell_period": None
    }

    fallback_values = []

    for column in values:

        if values[column] is None:

            values[column] = training_data[
                column
            ].median()

            fallback_values.append(column)


    # -----------------------------
    # CURRENT TIME
    # -----------------------------

    now = pd.Timestamp.now()

    month = now.month
    day = now.day
    hour = now.hour


    # -----------------------------
    # MODEL INPUT
    # -----------------------------

    model_input = pd.DataFrame([{

        "month": month,

        "day": day,

        "hour": hour,

        "latitude": latitude,

        "longitude": longitude,

        "sea_surface_temp":
            values["sea_surface_temp"],

        "wind_speed":
            values["wind_speed"],

        "wave_height":
            values["wave_height"],

        "wave_period":
            values["wave_period"],

        "swell_height":
            values["swell_height"],

        "swell_period":
            values["swell_period"]

    }])


    # -----------------------------
    # PREDICTION
    # -----------------------------

    model_score = model.predict_proba(
        model_input
    )[0][1]

    risk_score = round(
        model_score * 100
    )


    if risk_score < 25:

        risk_level = "LOW"

    elif risk_score < 50:

        risk_level = "MODERATE"

    elif risk_score < 75:

        risk_level = "HIGH"

    else:

        risk_level = "VERY HIGH"


    # -----------------------------
    # RESULTS
    # -----------------------------

    st.markdown("---")

    st.markdown(
        "## 🦈 SharkBait Risk Assessment"
    )


    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Risk Score",
            f"{risk_score} / 100"
        )

    with result_col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    # -----------------------------
    # NOAA STATION
    # -----------------------------

    st.markdown("---")

    st.markdown(
        "### NOAA Station"
    )

    st.write(
        f"**Station:** {station_id}"
    )

    st.write(
        f"**Distance:** "
        f"{station['distance_km']:.1f} km"
    )


    # -----------------------------
    # MARINE CONDITIONS
    # -----------------------------

    st.markdown(
        "### Current Marine Conditions"
    )

    c1, c2 = st.columns(2)


    with c1:

        if sea_surface_temp is not None:

            st.metric(
                "Water Temperature",
                f"{sea_surface_temp:.1f} °C"
            )

        else:

            st.metric(
                "Water Temperature",
                "Unavailable"
            )


        if wind_speed is not None:

            st.metric(
                "Wind Speed",
                f"{wind_speed:.1f} m/s"
            )

        else:

            st.metric(
                "Wind Speed",
                "Unavailable"
            )


    with c2:

        if wave_height is not None:

            st.metric(
                "Wave Height",
                f"{wave_height:.1f} m"
            )

        else:

            st.metric(
                "Wave Height",
                "Unavailable"
            )


        if wave_period is not None:

            st.metric(
                "Wave Period",
                f"{wave_period:.1f} s"
            )

        else:

            st.metric(
                "Wave Period",
                "Unavailable"
            )


    # -----------------------------
    # FALLBACK WARNING
    # -----------------------------

    if fallback_values:

        st.warning(
            "Some NOAA measurements were unavailable "
            "and were replaced with training-data "
            "median values: "
            + ", ".join(fallback_values)
        )


    # -----------------------------
    # MODEL INFORMATION
    # -----------------------------

    st.markdown("---")

    st.caption(
        f"Model attack score: "
        f"{model_score:.3f}"
    )

    st.caption(
        "Risk score is a machine-learning model "
        "score and is not a literal probability "
        "of a shark attack."
    )