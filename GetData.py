import xarray as xr
import numpy as np
import random
import torch

def get_features_at(ds, X, time, latitude, longitude):

    desired_time = np.datetime64(time)
    # Find time index
    try: time_idx = np.where(ds['valid_time'].values == desired_time)[0][0]
    except IndexError:
        #print(f"Time {time} not found in dataset.")
        return False

    

    # Find point index where both latitude and longitude match
    latitudes = ds['latitude'].values  # shape: (point,)
    longitudes = ds['longitude'].values  # shape: (point,)
    try: point_idx = np.where((latitudes == latitude) & (longitudes == longitude))[0][0]
    except IndexError:
        #print(f"Point ({latitude}, {longitude}) not found in dataset.")
        return "edge"

    # Get the features as a list for that point and time
    return X[:, point_idx, time_idx].tolist()
'''
time = '2025-01-01T10:00:00'
latitude = 36
longitude = -28
features_list = get_features_at(ds, X, time, latitude, longitude)
print(features_list)
#'''



def check_small_grid(ds, X, latitude, longitude, neigh_lat, neigh_lon, start_time, time_horizon=2):
    list_res = []
    start_time = np.datetime64(start_time)
    for t in range(time_horizon):
        current_time = start_time + np.timedelta64(t, 'h')
        for i in range(-neigh_lat, neigh_lat + 1):
            for j in range(-neigh_lon, neigh_lon + 1):
                lat = latitude + i * 0.25
                lon = longitude + j * 0.25
                features = get_features_at(ds, X, current_time, lat, lon)
                if features is False:
                    return False
                if features == "edge":
                    return "edge"
                list_res.append(features)
                #print(len(features), 'features at', lat, lon, current_time)
    return np.concatenate(list_res).tolist()

'''
# Example usage:
# result = check_small_grid(ds, X, 35.25, -26, 1, 1, '2025-01-01T10:00:00', time_horizon=2)

#check if neighboors exist

time = '2025-01-01T23:00:00'
latitude = 36.25
longitude = -28.25
features_list = get_features_at(ds, X, time, latitude, longitude)
neigh_lat = 1
neigh_lon = 1

a = check_small_grid(ds, X, latitude, longitude, neigh_lat, neigh_lon, time, time_horizon=2)
print('shape a:', np.array(a).flatten().shape)  # should be (num_points, 11)'''




def get_train_data(ds, X, n_data=1, neigh_lat=1, neigh_lon=1, time_horizon=2, seed=40):
    list_data = []
    used_samples = set()
    edge_set = set()
    random.seed(seed)
    lat_vals = ds['latitude'].values
    lon_vals = ds['longitude'].values
    time_vals = ds['valid_time'].values

    while len(list_data) < n_data:
        latitude = random.choice(lat_vals)
        longitude = random.choice(lon_vals)
        start_time = random.choice(time_vals)
        sample_key = (latitude, longitude, start_time)

        if sample_key in used_samples:
            continue  # Skip if already used
        if (latitude, longitude) in edge_set:
            continue  # Skip if previously found to be an edge

        data = check_small_grid(ds, X, latitude, longitude, neigh_lat, neigh_lon, start_time, time_horizon)
        if data is False:
            #print("invalid point:", latitude, longitude, start_time)
            continue
        if data == "edge":
            if sample_key not in edge_set:
                edge_set.add(sample_key[0:2])
            continue


        used_samples.add(sample_key)
        list_data.append(data)
        # Simple progress bar
        progress = len(list_data) / n_data
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f"\rProgress: |{bar}| {len(list_data)}/{n_data} ({progress*100:.1f}%)", end='', flush=True)

    return list_data, list(used_samples)
'''
a = get_train_data(ds, X, n_data=10, neigh_lat=1, neigh_lon=1, time_horizon=2)
print('shape a:', a.shape)  # should be (num_points, 11)
#'''