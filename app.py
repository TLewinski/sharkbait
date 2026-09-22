import sys

import joblib
import pandas as pd
import requests
import streamlit as st

# Allow importing files from src/
sys.path.append("src")

from realtime_predict import (  # noqa: E402
    find_best_station,
    get_stations,
    get_value,
)

st.set_page_config(page_title="SharkBait", page_icon="🦈", layout="centered")


# -----------------------------
# STYLES
# -----------------------------

st.markdown(
    """
<style>
.block-container { padding-top: 2.5rem; max-width: 780px; }

.hero {
    background: linear-gradient(135deg, #0b3d5c 0%, #0e6b8c 55%, #1ba3b8 100%);
    border-radius: 18px;
    padding: 2rem 2rem 1.6rem;
    color: #fff;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(11, 61, 92, 0.25);
}
.hero h1 { color: #fff; font-size: 2.4rem; margin: 0; padding: 0; }
.hero p  { color: #d6f1f6; margin: .35rem 0 0; font-size: 1.02rem; }

.card {
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    background: rgba(128,128,128,0.05);
    height: 100%;
}
.card .label { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65; }
.card .value { font-size: 1.55rem; font-weight: 700; margin-top: .2rem; }
.card .sub   { font-size: .8rem; opacity: .6; margin-top: .15rem; }

.risk {
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    color: #fff;
    margin: .5rem 0 1.2rem;
}
.risk .top { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: .5rem; }
.risk .score { font-size: 3.2rem; font-weight: 800; line-height: 1; }
.risk .score span { font-size: 1.1rem; font-weight: 500; opacity: .8; }
.risk .level { font-size: 1.1rem; font-weight: 700; letter-spacing: .08em;
               background: rgba(255,255,255,.2); padding: .35rem .8rem; border-radius: 999px; }
.risk .bar  { height: 10px; background: rgba(255,255,255,.25); border-radius: 999px; margin-top: 1rem; overflow: hidden; }
.risk .fill { height: 100%; background: #fff; border-radius: 999px; }
.risk .loc  { margin-top: .8rem; font-size: .9rem; opacity: .9; }

.section { font-size: 1.05rem; font-weight: 700; margin: 1.4rem 0 .6rem; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# DATA / MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/sharkbait_time_model.pkl")


@st.cache_data
def load_training_data():
    return pd.read_csv("data/training_data.csv")


def _nominatim(location):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": "SharkBait/1.1 (github.com/TLewinski/sharkbait)"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    return {
        "latitude": float(results[0]["lat"]),
        "longitude": float(results[0]["lon"]),
        "display_name": results[0]["display_name"],
    }


def _open_meteo(location):
    parts = [p.strip() for p in location.split(",") if p.strip()]
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": parts[0], "count": 10, "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    best = results[0]
    if len(parts) > 1:
        hint = parts[1].lower()
        for r in results:
            fields = [str(r.get(k, "")).lower() for k in ("admin1", "country", "country_code")]
            if any(hint in f or f == hint for f in fields if f):
                best = r
                break
    name = ", ".join(x for x in (best.get("name"), best.get("admin1"), best.get("country")) if x)
    return {"latitude": best["latitude"], "longitude": best["longitude"], "display_name": name}


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_location(location):
    try:
        result = _nominatim(location)
        if result:
            return result
    except requests.RequestException:
        pass
    return _open_meteo(location)


model = load_model()
training_data = load_training_data()

RISK_LEVELS = [
    (25, "LOW", "linear-gradient(135deg,#1e8e5a,#34b37a)"),
    (50, "MODERATE", "linear-gradient(135deg,#c98a0b,#e6ad2e)"),
    (75, "HIGH", "linear-gradient(135deg,#d0561b,#ee7a3b)"),
    (101, "VERY HIGH", "linear-gradient(135deg,#a31d2b,#d63447)"),
]


def classify(score):
    for limit, name, color in RISK_LEVELS:
        if score < limit:
            return name, color
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]


def card(label, value, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def fmt(value, unit):
    return f"{value:.1f} {unit}" if value is not None else "—"


# -----------------------------
# HEADER + INPUT
# -----------------------------

st.markdown(
    """
<div class="hero">
  <h1>🦈 SharkBait</h1>
  <p>Real-time shark risk from live NOAA marine conditions.
  Enter a beach, city, or coastal location to get started.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.form("search", border=False):
    col_in, col_btn = st.columns([4, 1.3], vertical_alignment="bottom")
    with col_in:
        location = st.text_input(
            "Location",
            placeholder="e.g. Daytona Beach, Florida",
            label_visibility="collapsed",
        )
    with col_btn:
        predict_button = st.form_submit_button(
            "Analyze", type="primary", use_container_width=True
        )

st.caption("Try: New Smyrna Beach, FL · Myrtle Beach, SC · Maui, Hawaii · Cape Cod, MA")


# -----------------------------
# RUN PREDICTION
# -----------------------------

if predict_button:
    if not location.strip():
        st.warning("Please enter a location.")
        st.stop()

    with st.status("Analyzing conditions...", expanded=False) as status:
        status.update(label="Finding your location...")
        try:
            coordinates = geocode_location(location)
        except Exception as error:
            status.update(label="Lookup failed", state="error")
            st.error(f"Unable to find location: {error}")
            st.stop()

        if coordinates is None:
            status.update(label="Location not found", state="error")
            st.error("Location not found. Try including the city and state/country.")
            st.stop()

        latitude = coordinates["latitude"]
        longitude = coordinates["longitude"]

        status.update(label="Finding the nearest NOAA station...")
        try:
            stations = get_stations()
            station, observation = find_best_station(latitude, longitude, stations)
        except Exception as error:
            status.update(label="NOAA request failed", state="error")
            st.error(f"Unable to retrieve NOAA data: {error}")
            st.stop()

        if station is None or observation is None:
            status.update(label="No station found", state="error")
            st.error("No NOAA station with available real-time data was found.")
            st.stop()

        status.update(label="Running the model...")

        sea_surface_temp = get_value(observation, "WTMP")
        wind_speed = get_value(observation, "WSPD")
        wave_height = get_value(observation, "WVHT")
        wave_period = get_value(observation, "DPD")

        values = {
            "sea_surface_temp": sea_surface_temp,
            "wind_speed": wind_speed,
            "wave_height": wave_height,
            "wave_period": wave_period,
            "swell_height": None,
            "swell_period": None,
        }

        fallback_values = []
        for column in values:
            if values[column] is None:
                values[column] = training_data[column].median()
                fallback_values.append(column)

        now = pd.Timestamp.now()
        model_input = pd.DataFrame([{
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "latitude": latitude,
            "longitude": longitude,
            **values,
        }])

        model_score = model.predict_proba(model_input)[0][1]
        risk_score = round(model_score * 100)
        risk_level, risk_color = classify(risk_score)

        status.update(label="Analysis complete", state="complete")

    # -----------------------------
    # RISK CARD
    # -----------------------------

    short_name = ", ".join(coordinates["display_name"].split(", ")[:3])

    st.markdown(
        f"""
<div class="risk" style="background:{risk_color}">
  <div class="top">
    <div class="score">{risk_score}<span> / 100</span></div>
    <div class="level">{risk_level} RISK</div>
  </div>
  <div class="bar"><div class="fill" style="width:{max(risk_score, 2)}%"></div></div>
  <div class="loc">📍 {short_name} &nbsp;·&nbsp; {latitude:.4f}, {longitude:.4f}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # -----------------------------
    # CONDITIONS
    # -----------------------------

    st.markdown('<div class="section">🌊 Current marine conditions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Water temp", fmt(sea_surface_temp, "°C")), unsafe_allow_html=True)
    c2.markdown(card("Wind", fmt(wind_speed, "m/s")), unsafe_allow_html=True)
    c3.markdown(card("Wave height", fmt(wave_height, "m")), unsafe_allow_html=True)
    c4.markdown(card("Wave period", fmt(wave_period, "s")), unsafe_allow_html=True)

    measured = [c for c in fallback_values if c not in ("swell_height", "swell_period")]
    if measured:
        st.info(
            "Some NOAA readings were unavailable and were filled with "
            "training-data medians: " + ", ".join(m.replace("_", " ") for m in measured)
        )

    # -----------------------------
    # STATION + MAP
    # -----------------------------

    st.markdown('<div class="section">📡 Data source</div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    s1.markdown(card("NOAA station", station["station"]), unsafe_allow_html=True)
    s2.markdown(card("Distance", f"{station['distance_km']:.1f} km", "from your location"),
                unsafe_allow_html=True)

    map_points = [{"lat": latitude, "lon": longitude, "color": "#e6532e", "size": 400}]
    if "lat" in station and "lon" in station:
        map_points.append({"lat": float(station["lat"]), "lon": float(station["lon"]),
                           "color": "#1ba3b8", "size": 250})
    elif "latitude" in station and "longitude" in station:
        map_points.append({"lat": float(station["latitude"]), "lon": float(station["longitude"]),
                           "color": "#1ba3b8", "size": 250})
    st.write("")
    st.map(pd.DataFrame(map_points), color="color", size="size", zoom=7)
    st.caption("🟠 Your location   🔵 NOAA station")

    # -----------------------------
    # FOOTER
    # -----------------------------

    with st.expander("About this score"):
        st.write(
            f"Raw model score: **{model_score:.3f}**. The risk score is a "
            "machine-learning model output, not a literal probability of a shark "
            "attack. It combines time of year, time of day, location, and live "
            "NOAA buoy conditions."
        )
