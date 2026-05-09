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

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# tracking uri
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# experiment
mlflow.set_experiment("Stock_Prediction")

# load data
df = pd.read_csv(
    "preprocessing/dataset_preprocessing/clean_data.csv"
)

# features & target
X = df.drop("LastPrice", axis=1)
y = df["LastPrice"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# tuning
for n in [50, 100]:
    for depth in [5, 10]:

        with mlflow.start_run(
            run_name=f"RF_n{n}_d{depth}"
        ):

            model = RandomForestRegressor(
                n_estimators=n,
                max_depth=depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # logging params
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", depth)

            # logging metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            # logging model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            print(
                f"Run success | "
                f"n={n}, depth={depth}, mse={mse}"
            )