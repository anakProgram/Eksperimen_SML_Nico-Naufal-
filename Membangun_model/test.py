# load dataset
import pandas as pd


df = pd.read_csv(
    "Membangun_model/dataset_preprocessing/hasil_preprocessing.csv"
)

# ubah categorical jadi numeric
df = pd.get_dummies(df, drop_first=True)

# feature target
X = df.drop("LastPrice", axis=1)
y = df["LastPrice"]

print(X.dtypes)
print(X.select_dtypes(include='object').columns)