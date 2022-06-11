from asyncio import Task
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
import pickle
import os

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
#from prefect.backend.artifacts import create_link_artifact


my_logger = get_run_logger

from mlflow.tracking import MlflowClient
client = MlflowClient()

# MLFLOW_TRACKING_URI = "http://52.18.246.39:5000"
# client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# EXPERIMENT_NAME = "home_work_w3"
# experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

# @task
# def make_artifact():
#     create_link_artifact("https://www.prefect.io/")

@task
def dump_pickle(obj, filename):    
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

from prefect import flow, task, get_run_logger


@task
def get_paths(date=None):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    
    if date is None:
        current_date = datetime.today()
        train_date = (current_date - relativedelta(months=2)).strftime('%Y-%m')
        validate_date = (current_date - relativedelta(months=1)).strftime('%Y-%m')
    else:
        custom_date = datetime.fromisoformat(f"{date}")
        train_date = (custom_date-relativedelta(months=2)).strftime('%Y-%m')
        validate_date = (custom_date-relativedelta(months=1)).strftime('%Y-%m')

    
    data_url = "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_"

    train_path = f"{data_url}{train_date}.parquet"
    val_path = f"{data_url}{validate_date}.parquet"
    
    return train_path, val_path 



@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    logger = get_run_logger()
    
    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger = get_run_logger()


    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values
    
    logger = get_run_logger()

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):

    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    
    # save dictvectorizer
    dump_pickle(dv, os.path.join("./outbut", "dv.pkl"))
    run_model(df_val_processed, categorical, dv, lr)



from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="deployment_week3",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["mlops"]
)







