import pandas as pd

def read_dataframe(filename, categorical):
    # Downloading the data
    df = pd.read_parquet(filename)

    # Computing duration
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Remove outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Handel missing values
    df['PUlocationID'].fillna(value=-1, inplace=True)
    df['DOlocationID'].fillna(value=-1, inplace=True)

    # 
    df[categorical] = df[categorical].astype(str)

    return df

