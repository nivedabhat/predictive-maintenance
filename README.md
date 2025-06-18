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

##  Tech Stack

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



 
 ```

---

#   Setup Instructions


### To begin the initial setup and start Docker, run the following script in your terminal

```bash
bash start.sh
```

---

### Verify Docker Startup
### Once Docker is successfully running, you should see a message like:

```bash
fastapi_app          |  Alert sent to Kafka: {'timestamp': '2025-06-09T11:28:49', 'device_id': '3GBA_112_410-ADDIN_Calc_UNIT5', 'parameter': 'rated_output_p_n', 'value': -0.18, 'reason': 'Below lower bound (0.7000000000000002)'}python pdf_spec_parser/spec_parser.py
```

> This starts:
> - Kafka & Zookeeper
> - FastAPI app
> - Kafka RUL producer
> - Streamlit dashboard


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

- `.pkl` files and feature configs must exist before Docker builds.
- Kafka must be running before starting the Kafka data producer.

---
