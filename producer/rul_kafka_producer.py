"""Kafka producer script to stream telemetry data from CSV to Kafka topic."""

import os
import json
import time
import traceback
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import pandas as pd

# === CONFIGURATION ===
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
TOPIC = os.getenv("KAFKA_TOPIC", "telemetry")
CSV_PATH = os.getenv("CSV_PATH", "/app/pdf_spec_parser/output/prod copy.csv")
SEND_INTERVAL = float(os.getenv("KAFKA_SEND_INTERVAL", 0.5))  # in seconds

# === LOAD DATASET ===
try:
    df = pd.read_csv(CSV_PATH)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("=", "")
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(by=["equipment_id", "timestamp"])

except Exception as e:
    raise SystemExit(f"Failed to load or parse CSV file: {e}") from e

# === KAFKA PRODUCER SETUP ===
print(f"🔌 Connecting to Kafka at {KAFKA_BROKER}...")
PRODUCER = None
for attempt in range(10):
    try:
        PRODUCER = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8")
        )
        print("Connected to Kafka.")
        break
    except NoBrokersAvailable:
        print(f"Attempt {attempt + 1}/10 failed. Retrying in 5 seconds...")
        time.sleep(5)
else:
    raise SystemExit("Kafka connection failed after 10 attempts.")

# === STREAM TO KAFKA ===
print(f"Streaming {len(df)} rows to topic '{TOPIC}'...")

try:
    for _, row in df.iterrows():
        try:
            payload = {
                "timestamp": row["timestamp"].isoformat() if not pd.isnull(row["timestamp"]) else None,
                "equipment_id": row["equipment_id"],
                "model_id": row["model_id"],
                "readings": {}
            }

            for col in df.columns:
                if col not in ["timestamp", "equipment_id", "model_id"]:
                    val = row[col]
                    if pd.notnull(val):
                        try:
                            payload["readings"][col] = float(val)
                        except (ValueError, TypeError):
                            continue

            if not payload["readings"]:
                print(f"Skipping row with no valid readings for {payload['equipment_id']} at {payload['timestamp']}")
                continue

            future = PRODUCER.send(TOPIC, key=str(payload["equipment_id"]), value=payload)
            try:
               record_metadata = future.get(timeout=10)
               print(f"Sent to partition {record_metadata.partition} offset {record_metadata.offset}")
            except Exception as e:
               print(f"Failed to send message: {e}")
            print(f"Sent: {payload['timestamp']} | {payload['equipment_id']} | {len(payload['readings'])} readings")

        except Exception:
            print("Error while sending row:")
            traceback.print_exc()

        time.sleep(SEND_INTERVAL)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    PRODUCER.flush()
    PRODUCER.close()
    print("Kafka producer shutdown complete.")

