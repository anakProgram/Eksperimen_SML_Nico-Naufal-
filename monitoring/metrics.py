from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram
)

import time
import random

# Counter metric
REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total prediction requests'
)

# Gauge metrics
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

MEMORY_USAGE = Gauge(
    'memory_usage_percent',
    'Memory usage percent'
)

# Histogram metric
INFERENCE_TIME = Histogram(
    'inference_time_seconds',
    'Inference latency'
)

# Start Prometheus server
start_http_server(8000)

print("Prometheus metrics running on port 8000")

while True:

    REQUEST_COUNT.inc()

    MODEL_MSE.set(random.uniform(0.1, 1.0))

    PREDICTION_VALUE.set(random.uniform(100, 500))

    CPU_USAGE.set(random.uniform(20, 95))

    MEMORY_USAGE.set(random.uniform(30, 90))

    with INFERENCE_TIME.time():
        time.sleep(random.uniform(0.1, 0.5))

    time.sleep(5)