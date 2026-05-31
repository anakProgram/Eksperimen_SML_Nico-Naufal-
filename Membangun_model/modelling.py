import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Experiment
mlflow.set_experiment("Stock_Prediction")

# WAJIB (kriteria Basic)
mlflow.sklearn.autolog(log_models=True)

# Load dataset
df = pd.read_csv(
    "Membangun_model/dataset_preprocessing/hasil_preprocessing.csv"
)

# Features & target
X = pd.get_dummies(
    df.drop("LastPrice", axis=1),
    drop_first=True
)

y = df["LastPrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metric
    mse = mean_squared_error(y_test, y_pred)

    print("MSE:", mse)

    # Log metric
    mlflow.log_metric("mse", mse)

    # Save model artifact
    mlflow.sklearn.save_model(
        sk_model=model,
        path="saved_model"
    )

    # Log artifacts manually
    mlflow.log_artifacts("saved_model", artifact_path="model")

    print("Model artifact berhasil disimpan")