import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autumn Air Quality & Health Risk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CUSTOM CSS ---
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- CONSTANTS & CONFIG ---
CITY_COORDS = {
    "Delhi 🇮🇳": (28.6139, 77.2090),
    "Helsinki 🇫🇮": (60.1699, 24.9384),
    "Amsterdam 🇳🇱": (52.3676, 4.9041),
    "New York 🇺🇸": (40.7128, -74.0060),
    "Beijing 🇨🇳": (39.9042, 116.4074),
    "London 🇬🇧": (51.5074, -0.1278),
    "Sydney 🇦🇺": (-33.8688, 151.2093)
}

# --- DATA FETCHING (Cached for performance) ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_air_quality_data(city, lat, lon, days=30):
    """Fetches historical air quality and weather data."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_url = "https://api.open-meteo.com/v1/forecast"
    
    # Air Quality Params
    params_aq = {
        "latitude": lat, "longitude": lon,
        "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide"],
        "past_days": days, "timezone": "auto"
    }
    
    # Weather Params (Wind, Temp, Humidity)
    params_weather = {
        "latitude": lat, "longitude": lon,
        "daily": ["temperature_2m_mean", "wind_speed_10m_max", "relative_humidity_2m_mean"],
        "past_days": days, "timezone": "auto"
    }

    try:
        r_aq = requests.get(url, params=params_aq)
        r_w = requests.get(weather_url, params=params_weather)
        
        aq_data = r_aq.json()
        w_data = r_w.json()

        # Process Air Quality
        df_aq = pd.DataFrame({
            "date": pd.to_datetime(aq_data["hourly"]["time"]),
            "pm25": aq_data["hourly"]["pm2_5"],
            "pm10": aq_data["hourly"]["pm10"],
            "no2": aq_data["hourly"]["nitrogen_dioxide"],
            "co": aq_data["hourly"]["carbon_monoxide"]
        })

        # Process Weather (Resample to hourly to match AQ)
        df_w = pd.DataFrame({
            "date": pd.to_datetime(w_data["hourly"]["time"] if "hourly" in w_data else w_data["daily"]["time"]),
            "temp": w_data["hourly"]["temperature_2m"] if "hourly" in w_data else np.nan, # Fallback logic if API changes
            "wind": w_data["hourly"]["wind_speed_10m"] if "hourly" in w_data else np.nan
        })
        
        # Simple fallback for daily data if hourly weather isn't fully aligned
        if "daily" in w_data:
             df_w_daily = pd.DataFrame({
                "date": pd.to_datetime(w_data["daily"]["time"]),
                "temp": w_data["daily"]["temperature_2m_mean"],
                "wind": w_data["daily"]["wind_speed_10m_max"]
            })
             # Merge daily weather forward fill to hourly
             df_w = pd.merge_asof(df_aq, df_w_daily, on="date")

        # Merge and Resample to Daily Averages
        df = pd.merge_asof(df_aq, df_w, on="date")
        df.set_index("date", inplace=True)
        df_daily = df.resample("D").mean().dropna()
        df_daily.reset_index(inplace=True)
        
        return df_daily

    except Exception as e:
        st.error(f"Error fetching data for {city}: {e}")
        return pd.DataFrame()

# --- LOGIC: RISK CALCULATION ---
def calculate_health_risk(row, vulnerability_multiplier=1.0):
    """
    Calculates a normalized health risk score (0-200+).
    vulnerability_multiplier: 1.0 (Normal), 1.5 (Sensitive Groups)
    """
    # Weights based on health impact
    score = (
        (row["pm25"] / 60) * 50 +    # PM2.5 is most dangerous
        (row["pm10"] / 100) * 30 +   # PM10
        (row["no2"] / 200) * 15 +    # NO2
        (row["co"] / 10000) * 5      # CO
    )
    return round(score * vulnerability_multiplier, 1)

def get_risk_level(score):
    if score <= 50: return "Low", "#00e400"
    elif score <= 100: return "Moderate", "#ffff00"
    elif score <= 150: return "Unhealthy for Sensitive", "#ff7e00"
    elif score <= 200: return "Unhealthy", "#ff0000"
    else: return "Very Unhealthy", "#8f3f97"

def generate_forecast(df, days=7):
    """Simple Linear Regression Forecast"""
    if len(df) < 5: return None
    
    model = LinearRegression()
    # Use last 14 days to predict next 7
    recent = df.tail(14)
    X = np.arange(len(recent)).reshape(-1, 1)
    y = recent["health_risk_score"].values
    
    model.fit(X, y)
    
    future_X = np.arange(len(recent), len(recent) + days).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    future_dates = [df["date"].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
    
    return pd.DataFrame({"date": future_dates, "forecast": forecast})

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. City Selection
    selected_city = st.selectbox("Select City", list(CITY_COORDS.keys()))
    
    # 2. Date Range
    days_range = st.slider("Historical Data (Days)", 7, 60, 30)
    
    st.markdown("---")
    
    # 3. User Personalization
    st.subheader("👤 Personalization")
    user_profile = st.radio("Who is this for?", 
                            ["General Public", "Children / Elderly", "Athletes / Outdoor Workers"])
    
    # Set Multiplier
    if user_profile == "General Public": multiplier = 1.0
    elif user_profile == "Children / Elderly": multiplier = 1.5
    else: multiplier = 1.2 # Athletes breathe in more air
    
    st.info(f"Vulnerability Multiplier: x{multiplier}")

# --- MAIN APPLICATION ---
st.title("🍂 Advanced Air Quality & Health Risk Analysis")
st.markdown(f"### Real-time Analysis for **{selected_city}**")

# Fetch Data
lat, lon = CITY_COORDS[selected_city]
df = fetch_air_quality_data(selected_city, lat, lon, days=days_range)

if not df.empty:
    # Calculate Risk
    df["health_risk_score"] = df.apply(lambda row: calculate_health_risk(row, multiplier), axis=1)
    df["risk_level"], df["color_code"] = zip(*df["health_risk_score"].apply(get_risk_level))
    
    latest = df.iloc[-1]
    level_text, color_hex = get_risk_level(latest["health_risk_score"])

    # --- TOP SECTION: METRICS & MAP ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current AQI (PM2.5)", f"{latest['pm25']:.1f} µg/m³")
    
    with col2:
        st.metric("Personalized Risk Score", f"{latest['health_risk_score']:.1f}")
        
    with col3:
        st.metric("Risk Level", level_text)
    
    with col4:
        # Trend calculation
        trend = latest['health_risk_score'] - df.iloc[-2]['health_risk_score']
        st.metric("7-Day Trend", f"{trend:+.1f}", delta_color="inverse")

    # Map View (All Cities Comparison)
    st.subheader("🌍 Global Snapshot (Real-time PM2.5)")
    map_data = []
    for city, (c_lat, c_lon) in CITY_COORDS.items():
        # Quick fetch for map (or cache it properly in production)
        # For speed here, we just simulate a quick fetch or use current data if match
        if city == selected_city:
            pm25_val = latest["pm25"]
        else:
            # In a real app, you'd loop fetch or have a cached dict. 
            # Here we just put a placeholder or 0 for others to avoid API timeouts in this demo.
            # Ideally, load all city data in background on start.
            pm25_val = 0 
            
        if pm25_val > 0:
            map_data.append({"lat": c_lat, "lon": c_lon, "city": city, "pm25": pm25_val})

    if map_data:
        map_df = pd.DataFrame(map_data)
        st.map(map_df, size="pm25", color=px.colors.sequential.Reds, zoom=4, use_container_width=True)
        st.caption("Bubble size represents PM2.5 concentration (Only showing detailed data for selected city in this demo).")

    # --- TABS SECTION ---
    tab1, tab2, tab3 = st.tabs(["📈 Historical Trends", "🧪 Pollutant Breakdown", "🔮 Forecast & Advice"])

    with tab1:
        # Interactive Line Chart
        fig = px.line(df, x="date", y="health_risk_score", 
                      title="Health Risk Over Time",
                      labels={"health_risk_score": "Risk Score", "date": "Date"},
                      color_discrete_sequence=["#FF4B4B"])
        
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Low Risk")
        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, annotation_text="Moderate")
        fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, annotation_text="High")
        
        fig.update_layout(hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather Correlation
        st.subheader("🌬️ Impact of Wind on Pollution")
        fig_wind = px.scatter(df, x="wind", y="pm25", trendline="ols",
                              title="Wind Speed vs. PM2.5 Concentration",
                              labels={"wind": "Wind Speed (km/h)", "pm25": "PM2.5 (µg/m³)"},
                              template="plotly_dark")
        st.plotly_chart(fig_wind, use_container_width=True)

    with tab2:
        st.subheader("Pollutant Components")
        pollutant = st.selectbox("Select Pollutant", ["pm25", "pm10", "no2", "co"])
        
        fig_p = px.area(df, x="date", y=pollutant, 
                        title=f"{pollutant.upper()} Concentration",
                        template="plotly_dark")
        st.plotly_chart(fig_p, use_container_width=True)
        
        # Bar chart comparison of latest reading
        st.subheader("Latest Composition")
        comp_data = {
            "Pollutant": ["PM2.5", "PM10", "NO2", "CO"],
            "Concentration": [latest['pm25'], latest['pm10'], latest['no2'], latest['co']]
        }
        st.bar_chart(pd.DataFrame(comp_data).set_index("Pollutant"))

    with tab3:
        # Forecast Logic
        forecast_df = generate_forecast(df)
        if forecast_df is not None:
            st.subheader("7-Day Health Risk Forecast")
            
            # Combine historical and forecast for visual continuity
            forecast_df["type"] = "Forecast"
            hist_df = df[["date", "health_risk_score"]].copy()
            hist_df["type"] = "Historical"
            
            combined = pd.concat([hist_df, forecast_df.rename(columns={"forecast": "health_risk_score"})])
            
            fig_f = px.line(combined, x="date", y="health_risk_score", color="type",
                            line_dash="type", title="Projected Health Risk",
                            template="plotly_dark")
            st.plotly_chart(fig_f, use_container_width=True)
        
        # Health Advisory
        st.subheader("🩺 Personalized Health Advisory")
        st.markdown(f"<div class='card' style='border-left: 5px solid {color_hex}'>", unsafe_allow_html=True)
        
        if latest["health_risk_score"] < 50:
            st.success("**Air Quality is Good.**")
            st.write("It's a great day to be outside! Enjoy your run or walk.")
        elif latest["health_risk_score"] < 100:
            st.warning("**Moderate Concern.**")
            if multiplier > 1.0:
                st.write("While acceptable for most, **you are in a sensitive group.** Consider reducing prolonged outdoor exertion.")
            else:
                st.write("Unusually sensitive people should consider reducing prolonged outdoor exertion.")
        else:
            st.error("**Health Alert: Unhealthy Air.**)
            if multiplier > 1.0:
                st.write("**High Risk for You:** Avoid all outdoor physical activity. If you must go out, wear an N95 mask.")
            else:
                st.write("Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.")
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.info("💡 **Insight:** Autumn often sees higher pollution due to temperature inversions (trapping cold air and pollutants near the ground) and crop residue burning in agricultural regions.")

else:
    st.error("Could not load data. Please check your internet connection.")

# --- FOOTER ---
st.markdown("---")
st.caption("Data provided by Open-Meteo API | Analysis by Creative ML Projects")