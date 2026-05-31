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

# Enable autolog
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv(
    "Membangun_model/dataset_preprocessing/clean_data.csv"
)

# Feature & target
X = df.drop("LastPrice", axis=1)
y = df["LastPrice"]

# Split dataset
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

    # Training
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)

    print("MSE:", mse)

    # Explicit model logging
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )