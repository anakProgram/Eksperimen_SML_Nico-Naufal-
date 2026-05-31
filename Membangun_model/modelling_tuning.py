import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Disable system metrics
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Experiment
mlflow.set_experiment("Stock_Prediction")

# Load dataset
df = pd.read_csv(
    "Membangun_model/dataset_preprocessing/hasil_preprocessing.csv"
)

# Features & target
X = df.drop("LastPrice", axis=1)
y = df["LastPrice"]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Convert bool -> int
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)

# Validation
print("Data types:")
print(X.dtypes.unique())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Hyperparameter tuning
for n in [50, 100]:
    for depth in [5, 10]:

        with mlflow.start_run(
            run_name=f"RF_n{n}_d{depth}"
        ):

            # Model
            model = RandomForestRegressor(
                n_estimators=n,
                max_depth=depth,
                random_state=42,
                n_jobs=-1
            )

            # Training
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # Evaluation
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log parameters
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)

            # Log metrics
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)

            # Save model locally first
            mlflow.sklearn.save_model(
                sk_model=model,
                path="saved_model"
            )

            # Log artifact manually
            mlflow.log_artifacts(
                "saved_model",
                artifact_path="model"
            )

            print(
                f"Run success | "
                f"n={n}, depth={depth}, "
                f"MSE={mse:.4f}, "
                f"MAE={mae:.4f}, "
                f"R2={r2:.4f}"
            )