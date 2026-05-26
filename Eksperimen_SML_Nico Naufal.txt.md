# Tutorial Menjalankan Aplikasi MLOps

## 1. Aktifkan Virtual Environment

### Windows CMD

```bash
.venv\Scripts\activate
```

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

---

# 2. Install Dependencies

Pastikan seluruh library sudah terinstall.

```bash

pip install -r requirements.txt
```

---

# 3. Jalankan Preprocessing Dataset

File preprocessing akan membuat dataset bersih bernama `clean_data.csv`.

```bash
cd ..
pip install mlflow prometheus-client requests
python membangun_model/automate_Nico.py
```

Output akan tersimpan di:

```plaintext
preprocessing/dataset_preprocessing/clean_data.csv
```

---

# 4. Jalankan Training Model

Untuk melakukan training model machine learning:

```bash
python Membangun_model/modelling.py
```

Atau untuk hyperparameter tuning:

```bash
 mlflow server --host 127.0.0.1 --port 5000


'''lalu buka diterminal lain atau New Terminal Window '''

python Membangun_model/modelling_tuning.py
```

---

# 5. Jalankan MLflow UI

Buka terminal baru lalu jalankan:

```bash
mlflow ui
```

MLflow UI akan berjalan pada:

```plaintext
http://127.0.0.1:5001
```

Di dalam MLflow UI terdapat:

- Metrics
- Parameters
- Artifacts
- Model
- Experiment Tracking

---

---

# 6. Menjalankan Prometheus

Masuk ke folder Prometheus:

```powershell
cd "C:\prometheus\prometheus-3.11.3"
```

Jalankan Prometheus:

```powershell
.\prometheus.exe --config.file=prometheus.yml
```

Jika berhasil:

```text
Server is ready to receive web requests.
```

```buka dibrowser
http://localhost:9090
```

---

# 7. Monitoring di Prometheus

Buka:

```bash
python Monitoring_Logging/monitoring/metrics.py
'''lalu buka dibrowser''
http://localhost:9090
```

Query metrics berikut:

```text
 isi query  model_requests_total dan klik execute
```

```text
isi query model_mse dan klik execute
```

```text
isi query prediction_value dan klik execute
```

```text
isi query cpu_usage_percent dan klik execute
```

```text
isi query memory_usage_percent dan klik execute
```

---

# 8. Menjalankan Grafana

Instal Grafana
Kemudian
cd cd C:\Grafana\grafana\bin
lalu
.\grafana-server.exe

Buka browser:

```plaintext
http://localhost:3000
```

Login default:

```plaintext
Username : admin
Password : admin
```

Tambahkan Prometheus sebagai Data Source:

```plaintext
http://localhost:9090
```

---

# 11. Menjalankan GitHub Actions Workflow

1. Push project ke GitHub
2. Buka tab Actions
3. Pilih workflow
4. Klik Run Workflow

Workflow akan:

- install dependencies
- menjalankan preprocessing
- training model
- menyimpan artifact

---

# 12. Membuat Alerting Grafana

Masuk ke:

```text
Alerting
→ Alert Rules
→ New Alert Rule
```

Konfigurasi:

## Query

```text
cpu_usage_percent
```

## Condition

```text
IS ABOVE 80
```

## Evaluation

```text
Evaluate every: 5s
For: 5s
```

## Folder

```text
ML Monitoring
```

## Group

```text
ml_monitoring_group
```

Klik:

```text
Save rule and exit
```

Jika metric CPU lebih dari 80 maka status alert menjadi:

```text
Firing
```

---

# 13. Menjalankan Inference

Gunakan file inference yang sudah tersedia:

```bash
python Monitoring_Logging/inference.py
```

atau gunakan script berikut:

Buat file inference.py:

```python
import requests

url = "http://127.0.0.1:5001/invocations"

sample_data = {
    "dataframe_split": {
        "columns": [f"f{i}" for i in range(3078)],
        "data": [[0]*3078]
    }
}

response = requests.post(url, json=sample_data)

print("Status Code:", response.status_code)
print("Response:", response.text)
```

Jalankan:

```bash
python inference.py
```

Jika berhasil:

```text
Status Code: 200
```

dan:

```json
{"predictions":[...]}
```

---

# 13. Struktur Submission

```text
Monitoring dan Logging
├── 1.bukti_serving
├── 2.prometheus.yml
├── 3.prometheus_exporter.py
├── 4.bukti monitoring Prometheus
├── 5.bukti monitoring Grafana
├── 6.bukti alerting Grafana
├── 7.inference.py
├── folder/file tambahan
```
