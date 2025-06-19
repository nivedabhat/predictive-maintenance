# kafka_producer/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rul_kafka_producer.py .
# COPY ../data ./data

CMD ["python", "rul_kafka_producer.py"]
