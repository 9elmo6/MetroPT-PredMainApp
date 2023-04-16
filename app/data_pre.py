import pandas as pd
import numpy as np

def preprocess_data(raw_data):
    df = raw_data.copy()

    # Compute the difference between consecutive timestamps
    df['TimeDelta'] = df['timestamp'].diff()

    # Find the index of the first timestamp for each train
    train_start_index = df[df['TimeDelta'] > pd.Timedelta(hours=1)].index

    # Compute the compressor run time and idle time for each train
    T_run_list = []
    T_idle_list = []
    for i in range(len(train_start_index)):
        if i < len(train_start_index) - 1:
            # For trains that are not the last one
            T_run = (df.iloc[train_start_index[i]+1:train_start_index[i+1]]['COMP'] == 1).sum()
            T_idle = (df.iloc[train_start_index[i]+1:train_start_index[i+1]]['COMP'] == 0).sum()
            T_run_list.append(T_run)
            T_idle_list.append(T_idle)
        else:
            # For the last train
            T_run = (df.iloc[train_start_index[i]+1:]['COMP'] == 1).sum()
            T_idle = (df.iloc[train_start_index[i]+1:]['COMP'] == 0).sum()
            T_run_list.append(T_run)
            T_idle_list.append(T_idle)

    # Add the T_run and T_idle values to the DataFrame
    df.loc[train_start_index, 'T_run'] = T_run_list
    df.loc[train_start_index, 'T_idle'] = T_idle_list

    # Drop unnecessary columns
    columns_to_drop = ['gpsLong', 'gpsLat', 'gpsSpeed', 'gpsQuality']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns_to_drop, axis=1)

    # Extract features from analog sensors
    num_bins = 7
    num_analog_sensors = 8
    num_features_per_sensor = num_bins + 2  # num_bins plus T_run and T_idle
    features = np.zeros((len(df), num_analog_sensors * num_features_per_sensor))

    # Calculate bins for each analog sensor
    for sensor_idx in range(num_analog_sensors):
        sensor_data = df.iloc[:, sensor_idx + 1]  # Skip the timestamp column

        for idx, (T_run, T_idle) in enumerate(zip(df['T_run'], df['T_idle'])):
            # Check for NaN values in T_run and T_idle
            if pd.isna(T_run) or pd.isna(T_idle):
                continue

            cycle_duration = T_run + T_idle
            T_run_bins = np.array_split(sensor_data[:int(T_run)], 2)
            T_idle_bins = np.array_split(sensor_data[int(T_run):int(cycle_duration)], 5)

            # Calculate the mean values of each bin
            feature_idx = sensor_idx * num_features_per_sensor
            features[idx, feature_idx:feature_idx + 2] = [np.mean(bin) for bin in T_run_bins]
            features[idx, feature_idx + 2:feature_idx + 7] = [np.mean(bin) for bin in T_idle_bins] 

            # Add the T_run and T_idle values to the features
            features[idx, feature_idx + 7] = T_run
            features[idx, feature_idx + 8] = T_idle

    return features
