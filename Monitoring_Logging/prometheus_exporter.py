from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram
)

import random
import time

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

start_http_server(8000)

print("Prometheus Exporter running on port 8000")

while True:

    REQUEST_COUNT.inc()

    MODEL_MSE.set(random.uniform(0.1, 1.0))

    PREDICTION_VALUE.set(random.uniform(100, 500))

    CPU_USAGE.set(random.uniform(20, 95))

    with INFERENCE_TIME.time():
        time.sleep(random.uniform(0.1, 0.5))

    time.sleep(5)