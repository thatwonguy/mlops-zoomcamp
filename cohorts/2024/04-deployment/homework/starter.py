#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


year = 2023
month = 4
taxi_type = 'yellow'

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{taxi_type}/{taxi_type}_tripdata_{year}-{month}.parquet'

df = read_data(input_file)


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


import numpy as np
std_dev = np.std(y_pred)
mean_duration = np.mean(y_pred)
print('std_dev is: ', std_dev)
print('mean predicted duration is: ', mean_duration)



import uuid


str(uuid.uuid4())


n = len(df)
ride_ids = []
for i in range(n):
    ride_ids.append(str(uuid.uuid4()))


ride_ids[:10]


df['ride_id'] = ride_ids


df.head()


df_result = pd.DataFrame()



df_result['ride_id'] = df['ride_id']
# df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
# df_result['PULocationID'] = df['PULocationID']
# df_result['DOLocationID'] = df['DOLocationID']
# df_result['actual_duration'] = df['duration']
df_result['predicted_duration'] = y_pred
# df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


import os
file_size = os.path.getsize(output_file)  # Size in bytes
file_size_mb = file_size / (1024 * 1024)  # Convert to megabytes

print(f"Size of the output file: {file_size_mb:.2f} MB")






