#  Predictive Maintenance via Spec-Aware Real-Time Monitoring

> End-to-end system for real-time detection of spec violations and prediction of Remaining Useful Life (RUL) in industrial equipment.

---

##  Problem Statement

Industrial equipment often operates outside of nominal parameters before failing. The challenge is to:
- Parse technical specification sheets for acceptable parameter ranges
- Monitor real-time telemetry for violations
- Predict failures before they happen

---

##  Objective

Build a scalable, real-time predictive maintenance pipeline that:
- Extracts parameters from spec-sheet PDFs (e.g., temperature, vibration thresholds)
- Streams sensor data via Kafka
- Flags violations in real-time
- Predicts Remaining Useful Life (RUL) using ML models
- Visualizes results and exposes health status via REST API

---

## 🛠️ Tech Stack

| Layer              | Tools Used                              |
|-------------------|------------------------------------------|
| Spec Parsing       | `pdfminer`, `pytesseract`, `re` (Regex) |
| Data Streaming     | `Kafka`, `JSON`, `time` (simulator)     |
| Real-Time Engine   | `Python`, `Pandas`, `Unit Conversion`   |
| ML Model           | `XGBoost`, `Scikit-learn`               |
| Dashboard/API      | `Streamlit`, `FastAPI`                  |
| Deployment         | `Docker`, `Docker Compose`              |

---

##  Architecture

```text
    ┌─────────────┐         ┌──────────────┐       ┌──────────────┐
    │ Spec Parser │ ──────▶ │ Rule Store   │◀────┐ │ PDF/HTML Docs│
    └─────────────┘         └────┬─────────┘     └──────────────┘
                                  │
                                  ▼
                           ┌─────────────┐
                           │ Kafka Stream│◀────────────┐
                           └────┬────────┘             │
                                ▼                      │
                          ┌──────────────┐      ┌────────────┐
                          │ Compliance   │────▶ │ Alerts Topic│
                          │ Engine       │      └────────────┘
                          └────┬─────────┘
                               ▼
                        ┌───────────────┐
                        │ RUL Predictor │────▶ `/status/{device_id}`
                        └────┬──────────┘
                             ▼
                       ┌────────────┐
                       │ Dashboard  │
                       └────────────┘

 Future Enhancements
- Add Prometheus + Grafana for monitoring
- Add authentication for endpoints
- Adaptive thresholds based on degradation
- Multilingual spec-sheet support
- Unit tests and CI/CD integration



##  Setup Instructions

### 1 Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2️ Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scriptsactivate
pip install -r requirements.txt
```

---

### 3️ Parse Spec PDF to Threshold Files

```bash
python "data/Spec parser .py"
```

This script parses a PDF to generate threshold `.csv` and `.json` files.

---

### 4️ Generate Synthetic Telemetry Data

```bash
python notebook/"Sythetic data gereation_time_series_model.py"
```

This produces `synthetic_timeseries_realistic_wide.csv` used to train models.

---

### 5️ Train the ML Models

```bash
python notebook/Model.py
```

Outputs:
- failure_classifier.pkl, rul_regressor.pkl, le_model.pkl, rul_scaler.pkl
- feature_config.json
- Scalers and encoders

These are used by the FastAPI service.

---

### 6️ Stream Data to Kafka

```bash
python notebook/rul_prod_data.py
```

This streams telemetry data to Kafka topic prod copy.csv

---

### 7️ Run the Entire System via Docker

```bash
bash start.sh

NOTE: Ensure the name 'predictive-maintenance' matches your image/tag prefix
```

> This starts:
> - Kafka & Zookeeper
> - FastAPI app
> - Kafka RUL producer
> - Streamlit dashboard

---

##  Access the Application

| Service              | URL                            |
|----------------------|--------------------------------|
| FastAPI API Docs     | http://localhost:8000/docs     |
| Streamlit Dashboard  | http://localhost:8501          |

---

##  Kafka Topics Used

| Topic              | Purpose                        |
|---------------     |--------------------------------|
| `producer data`    | Streaming input telemetry data |
| `predictions`      | RUL + health predictions       |
| `alerts`           | Parameter violation alerts     |

---

##  Notes

- Make sure you run steps 3–6 before launching Docker.
- `.pkl` files and feature configs must exist before Docker builds.
- Kafka must be running before starting the Kafka data producer.

---
