import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🍂 Autumn Air Quality → Health Risk Comparison")

city = st.selectbox("Select City", ["Delhi", "Helsinki"])

df = pd.read_csv(f"{city.lower()}_data.csv", parse_dates=["date"])

st.subheader(f"Health Risk Overview: {city}")

st.metric("Average Health Risk Score", round(df["health_risk_score"].mean(), 2))
st.metric("Highest Risk Level", df["risk_level"].mode()[0])

st.line_chart(df.set_index("date")[["pm25", "pm10", "no2"]])

st.subheader("Health Risk Timeline")
st.line_chart(df.set_index("date")["health_risk_score"])

st.subheader("🩺 Health Advisory")
latest = df.iloc[-1]
if latest["risk_level"] == "High":
    st.error("High risk: Elderly, children, and asthma patients should avoid outdoor exposure.")
elif latest["risk_level"] == "Moderate":
    st.warning("Moderate risk: Limit prolonged outdoor activity.")
else:
    st.success("Low risk: Safe for most outdoor activities.")