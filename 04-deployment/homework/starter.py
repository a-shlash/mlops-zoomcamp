#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Predict The Ride Of Duration')
parser.add_argument('month', type=str)
parser.add_argument('year', type=str)

args = parser.parse_args()

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


month = args.month  
year = args.year  
file_path = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-{month}.parquet'

df = read_data(file_path)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


print(" Mean Predicted Duration is : ", round(y_pred.mean(), 3))

def prepare_features(df):
    df['ride_id'] = f'{year}/{month}_' + df.index.astype('str')
    df['predicted_duration'] = y_pred


    return df

df_result = prepare_features(df)

output_file = f'pred_result_{year}-{month}.parquet'
print(output_file)

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


f_size = os.path.getsize(output_file)
print('Size of the output file: ', round(f_size/(1024*1024)))

