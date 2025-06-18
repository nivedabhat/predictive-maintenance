import os
import re
import time
import json
import traceback
from datetime import datetime
from threading import Thread


import joblib
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware 
from kafka import KafkaConsumer, KafkaProducer
from pydantic import BaseModel
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from kafka.errors import NoBrokersAvailable
from fastapi.responses import JSONResponse

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.metrics import mean_squared_error
from typing import Dict, Union

# === CONFIG ===
def clean_param(name):
    name = name.lower()
    name = re.sub(r"[^a-z0-9_ ]+", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name

KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
TOPIC = "telemetry"
ALERT_TOPIC = "alerts"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./telemetry.db")

producer = None

app = FastAPI(
    title="RUL Prediction API",
    description="Predicts RUL, detects failures, and streams real-time alerts for equipment health.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DATABASE SETUP ===
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class Telemetry(Base):
    __tablename__ = "telemetry"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    equipment_id = Column(String)
    model_id = Column(String)
    parameter = Column(String)
    value = Column(Float)
    unit = Column(String)

class TelemetryRecord(BaseModel):
    timestamp: str
    equipment_id: str
    model_id: str
    readings: Dict[str, Union[float, int]]    

Base.metadata.create_all(bind=engine)

device_buffer = defaultdict(dict)
alert_buffer = defaultdict(list)

# === MODEL & SCALER PATHS ===
MODEL_PATH = os.getenv("MODEL_PATH")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH")
SCALER_CLS_PATH = os.getenv("SCALER_CLS_PATH")
SCALER_REG_PATH = os.getenv("SCALER_REG_PATH")
RUL_SCALER_PATH = os.getenv("RUL_SCALER_PATH")
FEATURE_COLS_PATH = os.getenv("FEATURE_COLS_PATH")
SPECS_PATH = os.getenv("SPECS_PATH")
LE_MODEL_PATH = os.getenv("LE_MODEL_PATH",)

# === LOAD MODELS & CONFIG ===
print("\U0001F4E6 Loading models...")
rul_model = joblib.load(MODEL_PATH)
cls_model = joblib.load(CLASSIFIER_PATH)
scaler_cls = joblib.load(SCALER_CLS_PATH)
scaler_reg = joblib.load(SCALER_REG_PATH)
rul_scaler = joblib.load(RUL_SCALER_PATH)
le_model = joblib.load(LE_MODEL_PATH)

def compute_features(eq_id, features, model_id):
    row = {}

    # Encode model
    try:
        model_enc = le_model.transform([model_id])[0]
    except:
        model_enc = 0  # fallback

    row['model_enc'] = model_enc

    for param in FEATURE_COLS:
        if param.startswith('model_enc'):
            continue

    for base_param in set(k.split('_')[0] for k in FEATURE_COLS if '_' in k and not k.startswith("model")):
        val = features.get(base_param, 0)
        bounds = spec_dict.get((model_id, base_param), {})
        lower, upper = bounds.get("lower"), bounds.get("upper")

        row[f"{base_param}_norm"] = (val - lower) / (upper - lower) if lower is not None and upper is not None and upper > lower else 0
        row[f"{base_param}_dev_mid"] = val - ((lower + upper) / 2) if lower is not None and upper is not None else 0
        row[f"{base_param}_dist_lower"] = val - lower if lower is not None else 0
        row[f"{base_param}_dist_upper"] = upper - val if upper is not None else 0

        # Rolling stats approximation using device_history
        history = [h["value"] for h in device_history[eq_id] if h["parameter"] == base_param][-10:]
        row[f"{base_param}_mean_5"] = np.mean(history[-5:]) if history else 0
        row[f"{base_param}_std_5"] = np.std(history[-5:]) if history else 0
        row[f"{base_param}_mean_10"] = np.mean(history) if history else 0
        row[f"{base_param}_std_10"] = np.std(history) if history else 0

    return pd.DataFrame([row])

with open(FEATURE_COLS_PATH) as f:
    FEATURE_COLS = json.load(f)

   

spec_df = pd.read_csv(SPECS_PATH)
spec_df.columns = spec_df.columns.str.lower().str.replace(" ", "_")
spec_dict = {
    (row['model_id'], clean_param(row['parameter'])): {
        "lower": row.get("lower"),
        "upper": row.get("upper")
    }
    for _, row in spec_df.iterrows()
    if pd.notna(row.get("lower")) and pd.notna(row.get("upper"))
}

latest_data = defaultdict(dict)
device_history = defaultdict(list)


def create_producer():
    for attempt in range(10):
        try:
            return KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8")
            )
        except NoBrokersAvailable:
            print(f" Waiting for Kafka attempt {attempt+1}/10")
            time.sleep(5)
    raise SystemExit(" Kafka broker not available after 10 attempts.")

@app.on_event("startup")
def init_producer():
    global producer
    producer = create_producer()

def compute_health_score(violation_count):
    return max(0, 100 - violation_count * 5)

def send_alert(alert: dict):
    eq_id = alert.get("device_id")
    param = alert.get("parameter")
    alert_key = f"{eq_id}-{param}"

    # Save to buffer
    alert_buffer[eq_id].append(alert)

    try:
        producer.send(ALERT_TOPIC, key=alert_key, value=alert)
        producer.flush()
        print(f" Alert sent to Kafka: {alert}")
    except Exception as e:
        print(f" Failed to send alert to Kafka: {e}")

def validate_and_store(record: TelemetryRecord):
    eq_id = record.equipment_id
    model_id = record.model_id
    timestamp = record.timestamp

    latest_data[eq_id]["readings"] = record.readings
    latest_data[eq_id]["timestamp"] = timestamp
    latest_data[eq_id]["model_id"] = model_id

    db = SessionLocal()
    for param, val in record.readings.items():
        # Store in in-memory history
        device_history[eq_id].append({
            "timestamp": timestamp,
            "parameter": param,
            "value": val
        })

        # Save to DB
        db_record = Telemetry(
            timestamp=datetime.fromisoformat(timestamp),
            equipment_id=eq_id,
            model_id=model_id,
            parameter=param,
            value=val,
            unit=None  # Units not available in row format
        )
        db.add(db_record)
    db.commit()
    db.close()

def predict_status_and_rul(eq_id: str):
    state = latest_data.get(eq_id, {})
    features = state.get("readings", {})
    model_id = state.get("model_id")

    if not features or not model_id:
        return

    try:
        # Prepare input for model
        df = compute_features(eq_id, features, model_id)
        df = df.reindex(columns=FEATURE_COLS, fill_value=0)

        # --- Failure prediction ---
        x_cls = scaler_cls.transform(df)
        cls_pred = cls_model.predict(x_cls)[0]

        # --- RUL prediction ---
        x_reg = scaler_reg.transform(df)
        rul_pred_scaled = rul_model.predict(x_reg)
        rul_pred_log = rul_scaler.inverse_transform(rul_pred_scaled.reshape(-1, 1)).ravel()[0]
        rul_pred = max(np.expm1(rul_pred_log), 0)

        # --- Violation detection ---
        violations = []
        for raw_param, val in features.items():
            cleaned_param = clean_param(raw_param)
            bounds = spec_dict.get((model_id, cleaned_param), {})
            lower = bounds.get("lower")
            upper = bounds.get("upper")

            if lower is not None and val < lower:
                reason = f"Below lower bound ({lower})"
                violations.append({
                    "parameter": raw_param,
                    "current_value": val,
                    "expected_range": [lower, upper],
                    "reason": reason
                })
            elif upper is not None and val > upper:
                reason = f"Above upper bound ({upper})"
                violations.append({
                    "parameter": raw_param,
                    "current_value": val,
                    "expected_range": [lower, upper],
                    "reason": reason
                })

        # Count unique violating parameters
        violation_count = len({v["parameter"] for v in violations})

        # Compute health score
        health_score = max(0, 100 - violation_count * 5)

        # Store in-memory state
        latest_data[eq_id].update({
            "status": "Failure" if cls_pred == 1 else "Normal",
            "rul": float(round(rul_pred)),
            "violation_count": violation_count,
            "health_score": health_score,
            "violations": violations
        })
        latest_data[eq_id]["alerts"] = violations

        # Prepare Kafka message
        rul_message = {
            "timestamp": state.get("timestamp"),
            "device_id": eq_id,
            "model_id": model_id,
            "rul": float(round(rul_pred)),
            "status": "Failure" if cls_pred == 1 else "Normal",
            "violation_count": violation_count,
            "health_score": health_score,
            "violations": violations
        }

        try:
            producer.send("predictions", key=eq_id, value=rul_message)
            producer.flush()
            print(f" RUL prediction sent to Kafka: {rul_message}")
        except Exception as e:
            print(f" Failed to send RUL to Kafka: {e}")

    except Exception as e:
        print(f" Error in RUL prediction for {eq_id}: {e}")
        traceback.print_exc()



@app.on_event("startup")
def start_kafka_listener():
    thread = Thread(target=kafka_listener, daemon=True)
    thread.start()

def kafka_listener():
    print("Kafka consumer listenin")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset='earliest',
        group_id="fastapi-consumer"
    )

    for msg in consumer:
        try:
            # Copy to avoid mutation
            payload = msg.value.copy()
            print(f" Received from Kafka: {payload}")

            # Extract required fields
            timestamp = payload.get("timestamp")
            equipment_id = payload.get("equipment_id")
            model_id = payload.get("model_id")
            readings = payload.get("readings", {})

            # Validate all required fields exist
            if not (timestamp and equipment_id and model_id and isinstance(readings, dict)):
                print(" Invalid payload received. Missing required fields.")
                continue

            # Construct the Pydantic record
            record = TelemetryRecord(
                timestamp=timestamp,
                equipment_id=equipment_id,
                model_id=model_id,
                readings=readings
            )

            validate_and_store(record)
            predict_status_and_rul(equipment_id)

            # Alerting logic (unchanged)
            for param, val in readings.items():
                cleaned_param = clean_param(param)
                bounds = spec_dict.get((model_id, cleaned_param), {})
                lower = bounds.get("lower")
                upper = bounds.get("upper")

                if lower is not None and val < lower:
                    send_alert({
                        "timestamp": timestamp,
                        "device_id": equipment_id,
                        "parameter": param,
                        "value": val,
                        "reason": f"Below lower bound ({lower})"
                    })
                elif upper is not None and val > upper:
                    send_alert({
                        "timestamp": timestamp,
                        "device_id": equipment_id,
                        "parameter": param,
                        "value": val,
                        "reason": f"Above upper bound ({upper})"
                    })

        except Exception as e:
            print(" Error processing Kafka message:")
            traceback.print_exc()
@app.get("/status")
def get_device_status(device_id: str = Query(None, description="Filter by specific device ID")):
    if device_id:
        data = latest_data.get(device_id)
        if not data:
            return {"error": "Device ID not found"}
        alerts = data.get("alerts", [])
        return {
            device_id: {
                "timestamp": str(data.get("timestamp")),
                "rul": float(data.get("rul", 0)),
                "violation_count": int(data.get("violation_count", 0)),
                "health_score": int(data.get("health_score", 100)),
                "alerts": alerts,
                "status": data.get("status", "Unknown")
            }
        }

    return {
        device_id: {
            "timestamp": str(data.get("timestamp")),
            "rul": float(data.get("rul", 0)),
            "violation_count": int(data.get("violation_count", 0)),
            "health_score": int(data.get("health_score", 100)),
            "alerts": data.get("alerts", []),
            "status": data.get("status", "Unknown")
        }
        for device_id, data in latest_data.items()
    }

def get_device_history(device_id: str):
    db = SessionLocal()
    records = db.query(Telemetry).filter(Telemetry.equipment_id == device_id).order_by(Telemetry.timestamp).all()
    db.close()

    history_dict = defaultdict(dict)
    for r in records:
        timestamp = r.timestamp.isoformat()
        if "timestamp" not in history_dict[timestamp]:
            history_dict[timestamp]["timestamp"] = timestamp
        history_dict[timestamp][r.parameter] = r.value
    
    return {
        "device_id": device_id,
        "history": list(history_dict.values())
    }
@app.get("/history")
def history(device_id: str = Query(..., description="Device ID to fetch history for")):
    try:
        return get_device_history(device_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/trends")
def get_trends():
    return device_history

@app.get("/alerts")
def get_alerts():
    return alert_buffer

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
