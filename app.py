import streamlit as st
import pandas as pd

st.set_page_config(page_title="Autumn Air Quality & Health Risk", layout="wide")

st.title("🍂 Autumn Air Quality → Health Risk Analysis")
st.caption("Seasonal comparison of air pollution impact on human health")

# Load data
cities = {
    "Delhi 🇮🇳": (
        pd.read_csv("delhi_data.csv", parse_dates=["date"]),
        pd.read_csv("delhi_forecast.csv", parse_dates=["date"])
    ),
    "Helsinki 🇫🇮": (
        pd.read_csv("helsinki_data.csv", parse_dates=["date"]),
        pd.read_csv("helsinki_forecast.csv", parse_dates=["date"])
    ),
    "Amsterdam 🇳🇱": (
        pd.read_csv("amsterdam_data.csv", parse_dates=["date"]),
        pd.read_csv("amsterdam_forecast.csv", parse_dates=["date"])
    )
}

# --- METRICS ---
st.subheader("📊 City Comparison (Autumn Averages)")
cols = st.columns(3)

for col, (city, (df, _)) in zip(cols, cities.items()):
    with col:
        st.markdown(f"### {city}")
        st.metric("Avg PM2.5", round(df["pm25"].mean(), 1))
        st.metric("Avg Health Risk", round(df["health_risk_score"].mean(), 1))
        st.metric("Dominant Risk", df["risk_level"].mode()[0])

# --- HISTORICAL TRENDS ---
st.subheader("📈 Historical Health Risk Trends")
trend_cols = st.columns(3)

for col, (city, (df, _)) in zip(trend_cols, cities.items()):
    with col:
        st.markdown(f"#### {city}")
        st.line_chart(df.set_index("date")["health_risk_score"])

# --- FORECAST ---
st.subheader("🔮 7-Day Health Risk Forecast")
forecast_cols = st.columns(3)

for col, (city, (_, forecast)) in zip(forecast_cols, cities.items()):
    with col:
        st.markdown(f"#### {city}")
        st.line_chart(forecast.set_index("date")["forecast_health_risk"])

# --- ADVISORY ---
st.subheader("🩺 Latest Health Advisory")
adv_cols = st.columns(3)

for col, (city, (df, _)) in zip(adv_cols, cities.items()):
    latest = df.iloc[-1]
    with col:
        st.markdown(f"#### {city}")
        if latest["risk_level"] == "High":
            st.error("High risk: Avoid outdoor activity. Vulnerable groups at risk.")
        elif latest["risk_level"] == "Moderate":
            st.warning("Moderate risk: Limit prolonged outdoor exposure.")
        else:
            st.success("Low risk: Safe for outdoor activities.")

st.markdown("---")
st.caption("Built using real-world air quality data and a custom health risk index.")
