import requests
import pandas as pd

url = "http://127.0.0.1:5001/invocations"

sample = [[0]*3078]

data = {
    "inputs": sample
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.text)