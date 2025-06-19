import streamlit as st
import requests
import pandas as pd
import altair as alt
import time
from kafka import KafkaConsumer
import json

# === CONFIG ===
FASTAPI_URL = "http://fastapi_app:8000"
KAFKA_BROKER = "kafka:29092"
ALERT_TOPIC = "alerts"
PREDICTION_TOPIC = "predictions"
REFRESH_INTERVAL = 15  # seconds
MAX_KAFKA_MESSAGES = 50

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title(" Predictive Maintenance: Real-Time Monitoring")

st.markdown(
    f"**Overview**  \nLive telemetry → violation check → RUL prediction → Kafka → Dashboard  \nRefreshes every **{REFRESH_INTERVAL} seconds**"
)

@st.cache_data(ttl=10)
def fetch_devices():
    try:
        resp = requests.get(f"{FASTAPI_URL}/status")
        resp.raise_for_status()
        devices = list(resp.json().keys())
        return devices
    except Exception as e:
        st.error(f"Error fetching devices: {e}")
        return []

@st.cache_data(ttl=10)
def fetch_status(device_id):
    try:
        resp = requests.get(f"{FASTAPI_URL}/status", params={"device_id": device_id})
        resp.raise_for_status()
        return resp.json().get(device_id, {})
    except Exception as e:
        st.error(f"Error fetching status: {e}")
        return {}

@st.cache_data(ttl=10)
def fetch_alerts():
    try:
        resp = requests.get(f"{FASTAPI_URL}/alerts")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        return {}

@st.cache_data(ttl=10)
def fetch_history(device_id):
    try:
        resp = requests.get(f"{FASTAPI_URL}/history", params={"device_id": device_id})
        resp.raise_for_status()
        return resp.json().get("history", [])
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return []

def consume_kafka_topic(topic, max_messages=MAX_KAFKA_MESSAGES):
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[KAFKA_BROKER],
            auto_offset_reset="earliest",  # 
            enable_auto_commit=False,
            group_id="streamlit-group",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=5000  # 
        )
        messages = []
        for message in consumer:
            messages.append(message.value)
            if len(messages) >= max_messages:
                break
        consumer.close()
        return messages
    except Exception as e:
        st.error(f"Kafka connection error: {e}")
        return []

# --- Sidebar: Device selection & refresh ---
device_ids = fetch_devices()
if not device_ids:
    st.warning("No devices found.")
    st.stop()

selected_device = st.sidebar.selectbox("Select Device", device_ids)
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", min_value=5, max_value=60, value=REFRESH_INTERVAL, step=5)

# --- Fetch data ---
status = fetch_status(selected_device)
alerts = fetch_alerts()
history = fetch_history(selected_device)

# Dummy predicted RUL for demonstration (since your FastAPI provides rul directly)
predicted_rul = status.get("rul", None)

# --- Tabs ---
tabs = st.tabs([" Overview", " Violations", " Telemetry", " Alerts (Kafka)"])

# Overview tab
with tabs[0]:
    st.metric("Health Score", status.get("health_score", "N/A"))
    st.metric("Violation Count", status.get("violation_count", 0))
    st.metric("Remaining Useful Life (hours)", predicted_rul if predicted_rul is not None else "N/A")
    st.write(f"Last updated: {status.get('timestamp', 'N/A')}")

# Violations tab
with tabs[1]:
    st.subheader("Current Parameter Violations")
    active_alerts = status.get("alerts", [])
    if active_alerts:
        for alert in active_alerts:
            severity = alert.get("severity", "LOW").upper() if "severity" in alert else "LOW"
            color = {"HIGH": "red", "MODERATE": "orange ", "LOW": "pink"}.get(severity, "white")
            st.markdown(f"""
                <div style='background-color:{color}; padding:10px; border-radius:5px; margin-bottom:10px'>
                    <strong>{alert.get('parameter', '')}</strong>: {alert.get('value', '')} (Expected: {alert.get('expected_range', ['N/A','N/A'])[0]} - {alert.get('expected_range', ['N/A','N/A'])[1]})<br>
                    <em>Reason: {alert.get('reason', 'N/A')}</em>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No current violations.")

# Telemetry tab
with tabs[2]:
    # --- Enhanced Telemetry Tab ---

    st.subheader(" Telemetry History & Trends")

# Human-readable parameter labels
PARAMETER_LABELS = {
    "rated_output_p_n": " Rated Power Output (W)",
    "rated_speed_n_n": " Rated Speed (RPM)",
    "rated_current_i_n": " Rated Current (A)",
    "value_std_5": " Short-term Variability",
    "value_mean_5": " 5-Point Rolling Avg",
    "value_mean_10": " 10-Point Rolling Avg",
    # Add more mappings as needed
}

# Convert to DataFrame
df_history = pd.DataFrame(history)

if not df_history.empty:
    # Ensure timestamp is datetime
    if "timestamp" in df_history.columns:
        df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])

    # Identify numeric columns
    numeric_cols = df_history.select_dtypes(include='number').columns.tolist()

    # Let user select parameters
    selected_params = st.multiselect(
        " Parameters to plot",
        numeric_cols,
        default=numeric_cols[:3]
    )

    if selected_params:
        # Show latest readings as summary metrics
        st.markdown("###  Latest Sensor Readings")
        latest_row = df_history.sort_values("timestamp").iloc[-1]
        cols = st.columns(len(selected_params))
        for i, param in enumerate(selected_params):
            label = PARAMETER_LABELS.get(param, param.replace("_", " ").title())
            value = latest_row.get(param, "N/A")
            display_value = f"{value:.2f}" if isinstance(value, (float, int)) else str(value)
            cols[i].metric(label, display_value)

        # Plot telemetry trends
        df_melted = df_history.melt(
            id_vars=["timestamp"],
            value_vars=selected_params,
            var_name="Parameter",
            value_name="Value"
        )
        df_melted["Parameter"] = df_melted["Parameter"].map(
            PARAMETER_LABELS).fillna(df_melted["Parameter"])

        chart = alt.Chart(df_melted).mark_line().encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("Value:Q", title="Sensor Reading"),
            color=alt.Color("Parameter:N", title="Parameter"),
            tooltip=["timestamp:T", "Parameter:N", "Value:Q"]
        ).properties(
            title=" Telemetry Parameter Trends Over Time",
            background="white"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # Caption for layman understanding
        st.caption("""
        ℹ **What you're seeing**: These trends show how equipment sensors behave over time.
        Sudden spikes or drops may indicate anomalies or early warnings. Hover over points for values.
        """)

        # Downloadable CSV
        st.download_button(
            "⬇ Download CSV",
            df_history.to_csv(index=False),
            file_name=f"{selected_device}_telemetry.csv"
        )

    else:
        st.info(" Select at least one parameter to visualize.")
else:
    st.warning(" No telemetry data available for this device.")


# Alerts from Kafka tab
with tabs[3]:
    st.subheader(" Alerts from Kafka (`alerts` topic)`")

    # Fetch and show raw alert count
    kafka_alerts = consume_kafka_topic(ALERT_TOPIC)
    st.text(f" Raw Kafka alerts received: {len(kafka_alerts)}")

    if kafka_alerts:
        try:
            # Convert to DataFrame
            df_alerts = pd.DataFrame(kafka_alerts)

            # Optional: Rename columns for clarity
            df_alerts.rename(columns={
                "timestamp": " Timestamp",
                "device_id": " Device",
                "parameter": " Parameter",
                "value": " Value",
                "reason": " Reason"
            }, inplace=True)

            # Convert to datetime if not already
            df_alerts[" Timestamp"] = pd.to_datetime(df_alerts[" Timestamp"])

            # Sort newest first
            df_alerts = df_alerts.sort_values(" Timestamp", ascending=False)

            # Show in clean table
            st.dataframe(df_alerts, use_container_width=True)

        except Exception as e:
            st.error(f" Failed to parse alerts: {e}")
            st.json(kafka_alerts)  # Show raw if DataFrame parsing fails
    else:
        st.info("No alerts received from Kafka.")

# --- Auto-refresh ---
st.markdown("---")
st.caption(f" Auto-refreshing every {refresh_rate} seconds")
time.sleep(refresh_rate)
st.experimental_rerun()
