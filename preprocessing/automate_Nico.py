import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def handle_missing(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def encoding(df, target_col):
    categorical_cols = df.select_dtypes(include=['object']).columns

    # jangan encode target
    categorical_cols = categorical_cols.drop(target_col, errors='ignore')

    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

def scaling(df, target_col):
    scaler = StandardScaler()

    df = df.copy()

    # pisahkan fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # scaling hanya fitur numerik
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # gabungkan kembali
    df_scaled = pd.concat([X, y], axis=1)

    return df_scaled

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def preprocess_pipeline(input_path, output_path, target_col):
    df = load_data(input_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    df = handle_missing(df)
    df = encoding(df, target_col)
    df = scaling(df, target_col)

    save_data(df, output_path)

    return df

if __name__ == "__main__":
    preprocess_pipeline(
        "dataset_raw/DaftarSaham.csv",
        "preprocessing/dataset_preprocessing/clean_data.csv",
        target_col="LastPrice"
    )