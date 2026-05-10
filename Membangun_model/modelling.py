import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# local mlflow database
mlflow.set_tracking_uri("file:./mlruns")

# create experiment manually
experiment_name = "Stock_Prediction"

try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

# load dataset
df = pd.read_csv(
    "preprocessing/dataset_preprocessing/clean_data.csv"
)

X = df.drop("LastPrice", axis=1)
y = df["LastPrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run(experiment_id=experiment_id):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("MSE:", mse)