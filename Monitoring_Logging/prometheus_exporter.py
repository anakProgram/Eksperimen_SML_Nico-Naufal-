from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram
)

import time
import psutil
import joblib
import pandas as pd

from sklearn.metrics import mean_squared_error

# ======================
# LOAD MODEL & DATA
# ======================

model = joblib.load("mlartifacts\\1\\models\\m-f1a2ec09749c499f88d3aca8621c536f\\artifacts\\model.pkl")

df = pd.read_csv(
    "Membangun_model/preprocessing/dataset_preprocessing/hasil_preprocessing.csv"
)

X = df.drop("LastPrice", axis=1)
y = df["LastPrice"]

# ======================
# PROMETHEUS METRICS
# ======================

REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total prediction requests'
)

MODEL_MSE = Gauge(
    'model_mse',
    'Current model mse'
)

PREDICTION_VALUE = Gauge(
    'prediction_value',
    'Latest prediction value'
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percent'
)

INFERENCE_TIME = Histogram(
    'inference_time_seconds',
    'Inference latency'
)

# ======================
# START EXPORTER
# ======================

start_http_server(8000)

print("Prometheus Exporter running on port 8000")

# ======================
# MONITORING LOOP
# ======================

while True:

    REQUEST_COUNT.inc()

    # real CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    CPU_USAGE.set(cpu_percent)

    # sample inference
    sample_data = X.iloc[[0]]

    start_time = time.time()

    prediction = model.predict(sample_data)[0]

    inference_duration = time.time() - start_time

    INFERENCE_TIME.observe(inference_duration)

    # real prediction value
    PREDICTION_VALUE.set(float(prediction))

    # real mse
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)

    MODEL_MSE.set(float(mse))

    print(
        f"Prediction={prediction:.2f} | "
        f"MSE={mse:.4f} | "
        f"CPU={cpu_percent}%"
    )

    time.sleep(5)