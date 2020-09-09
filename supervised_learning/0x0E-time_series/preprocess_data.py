#!/usr/bin/env python3
"""contains the pre-processing function"""


# import pandas as pd


def preprocessing():
    """
    Pre-processing function, carries out the loading, cleaning,
    cropping, sub-sampling, splitting and normalization
    operations on the raw data
    :return: train_df, val_df, test_df
        train_df is the training dataset
        val_df is the validation dataset
        test_df is the test dataset
    """
    # DATASET LOADING
    filename = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    raw_data = pd.read_csv(filename)

    # CLEANING, CROPPING AND SUB-SAMPLING
    # Cleaning and reindexing
    df = raw_data.dropna()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.reset_index(inplace=True, drop=True)

    # Cropping
    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(inplace=True, drop=True)

    # subsample every hour
    df = df[0::60]

    # Auxiliary variables
    date_time = pd.to_datetime(df.pop('Timestamp'))
    data_cols = ['Open', 'High', 'Low', 'Close',
                 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']

    # Dataset description
    print(df.describe().transpose())

    # SANITY CHECK
    # Plotting

    # Complete time_series

    plot_features = df[data_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    # First month (720 hours)
    plot_features = df[data_cols][:720]
    plot_features.index = date_time[:720]
    _ = plot_features.plot(subplots=True)

    # DATASET SPLIT
    # Dictionary with name, index pairs for the data set columns
    column_indices = {name: i for i, name in enumerate(df.columns)}

    # Data set split (Train 70%, validation 20%, test 10%)
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    # Check
    # Description
    num_features = df.shape[1]
    print('Number of features:', num_features)
    print(train_df.describe().transpose())

    # Plotting
    plot_features = train_df[data_cols]
    plot_features.index = date_time[0:int(n * 0.7)]
    _ = plot_features.plot(subplots=True)

    # DATASET NORMALIZATION
    # Parameters calculation (train)
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)

    print('MEAN:\n', train_mean, '\n', sep='')
    print('STD:\n', train_std, '\n', sep='')

    # Normalization operation
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std

    # Normalization check
    print(df_std.head())

    return train_df, val_df, test_df
