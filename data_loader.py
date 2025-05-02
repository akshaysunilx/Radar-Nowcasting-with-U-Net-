# data_loader.py

import os
import numpy as np
import xarray as xr
import pandas as pd
from skimage.transform import resize
import gc
import psutil

def process_reflectivity(data, max_value=70.0):
    return np.clip(np.nan_to_num(data, nan=0.0), 0, max_value)

def process_velocity(data):
    return np.nan_to_num(data, nan=0.0)

def compute_magnitude(vel):
    return np.abs(vel)

def compute_divergence(vel):
    grad_x = np.gradient(vel, axis=-2)
    grad_y = np.gradient(vel, axis=-1)
    return grad_x + grad_y

def compute_direction(vel):
    grad_x = np.gradient(vel, axis=-2)
    grad_y = np.gradient(vel, axis=-1)
    return np.arctan2(grad_y, grad_x) / np.pi

def compute_vorticity(vel):
    grad_y = np.gradient(vel, axis=-2)
    grad_x = np.gradient(vel, axis=-1)
    return grad_y - grad_x

def print_ram_usage(tag=""):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[RAM Check] {tag} - RAM Usage: {ram_mb:.2f} MB")

def load_full_dataset(folder_paths, target_shape, time_horizon, sequence_length, reflectivity_max, velocity_scale):
    X_list, Y_list, input_times, future_times = [], [], [], []

    for folder_name, path in folder_paths.items():
        for file in sorted(os.listdir(path)):
            if file.endswith("_combined.nc"):
                try:
                    ds = xr.open_dataset(os.path.join(path, file))
                    ref = process_reflectivity(ds['Reflectivity_Horizontal'].values, reflectivity_max)
                    vel = process_velocity(ds['Horizontal_Velocity'].values)
                    times = pd.to_datetime(ds['time'].values)
                    num_samples = ref.shape[0] - sequence_length - time_horizon + 1
                    num_elev = ref.shape[1]

                    for i in range(num_samples):
                        X_sample = np.zeros((*target_shape, 6 * num_elev), dtype=np.float32)
                        Y_sample = np.zeros((*target_shape, time_horizon), dtype=np.float32)

                        for elev in range(num_elev):
                            ref_input = resize(ref[i, elev], target_shape, preserve_range=True)
                            vel_input = resize(vel[i, elev], target_shape, preserve_range=True)
                            X_sample[..., elev] = ref_input
                            X_sample[..., num_elev + elev] = vel_input
                            X_sample[..., 2*num_elev + elev] = compute_magnitude(vel_input)
                            X_sample[..., 3*num_elev + elev] = compute_divergence(vel_input)
                            X_sample[..., 4*num_elev + elev] = compute_direction(vel_input)
                            X_sample[..., 5*num_elev + elev] = compute_vorticity(vel_input)

                        for t in range(time_horizon):
                            max_future = np.max(ref[i + sequence_length + t], axis=0)
                            Y_sample[..., t] = resize(max_future, target_shape, preserve_range=True)

                        X_list.append(X_sample)
                        Y_list.append(Y_sample)
                        input_times.append(times[i])
                        future_times.append(times[i + sequence_length + time_horizon - 1])

                    ds.close()

                except Exception as e:
                    print(f"Error processing {file}: {e}")
                finally:
                    gc.collect()
                    print_ram_usage(f"Loaded {file}")

    X = np.array(X_list)
    Y = np.array(Y_list)
    input_times = pd.to_datetime(input_times)
    future_times = pd.to_datetime(future_times)

    num_elev = X.shape[-1] // 6
    X[..., :num_elev] = (X[..., :num_elev] - reflectivity_max/2) / (reflectivity_max/3)
    X[..., num_elev:2*num_elev] /= velocity_scale
    X[..., 2*num_elev:3*num_elev] /= velocity_scale
    X[..., 3*num_elev:] = np.clip(X[..., 3*num_elev:], -1, 1)
    Y = (Y - reflectivity_max/2) / (reflectivity_max/3)

    return X, Y, input_times, future_times
