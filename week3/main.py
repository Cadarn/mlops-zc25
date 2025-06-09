from pathlib import Path
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error

DATA_ROOT = Path("/Users/adam.hill/scratch/mlops-zoomcamp/cohorts/2025/02-experiment-tracking/raw_data")

DATA_FILE =  DATA_ROOT / "yellow_tripdata_2023-03.parquet"

df = pd.read_parquet(DATA_FILE)

# Q3
print(f"Number of rows: {df.shape[0]}")

# Q4
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

df = read_dataframe(DATA_FILE)
print(f"Number of rows: {df.shape[0]}")


# Q5
def generate_feature_dicts(df: pd.DataFrame, feature_cols: list[str])->list[dict]:
    if feature_cols is None:
        feature_cols = df.columns
    return df[feature_cols].to_dict(orient="records")

categorical = ["PULocationID", "DOLocationID"]
numerical = []

target = "duration"

train_df = df

dv = DictVectorizer()

X_train = dv.fit_transform(generate_feature_dicts(train_df, categorical+numerical))
y_train = train_df[target].values

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print(regr)