##### This code was used when only using single sensors. The abstraction between this and and combine_csv.py did not make logical sense as more sensors was added.
##### The code was refactored to combine the two scripts into one.

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample
import argparse
import sys
from tqdm import tqdm

# This code is based off the Axivity data processing pipeline described in the paper: 
# https://findresearcher.sdu.dk/ws/files/145634666/Generating_ActiGraph_Counts_from_Raw_Acceleration_Recorded_by_an_Alternative_Monitor.pdf
# The purpose of this file is to generate activity counts from raw Axivity data. 
# The pipeline consists of the following steps (as described in comments):

def butter_lowpass(cutoff, fs, order=4):
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def process_axivity_data(input_csv, output_csv):
    # We'll track each major step as one progress increment
    total_steps = 12
    pbar = tqdm(total=total_steps, desc="Processing Axivity Data")

    # Step 1: Load data
    pbar.set_postfix_str("Loading data...")
    data = pd.read_csv(input_csv)
    timestamps = data['dataTimestamp']
    x, y, z = data['axis1'], data['axis2'], data['axis3']
    pbar.update(1)

    # Step 2: Resample to 30 Hz
    pbar.set_postfix_str("Resampling to 30 Hz...")
    original_rate = 100  # Adjust if your input sampling rate is different
    target_rate = 30
    resample_factor = target_rate / original_rate
    num_samples = int(len(x) * resample_factor)
    x_resampled = resample(x, num_samples)
    y_resampled = resample(y, num_samples)
    z_resampled = resample(z, num_samples)
    pbar.update(1)

    # Step 3: Apply aliasing low-pass filter (cutoff ~15 Hz)
    pbar.set_postfix_str("Applying low-pass filter...")
    b, a = butter_lowpass(cutoff=14.9, fs=target_rate)  # Slightly less than 15
    x_filtered = filtfilt(b, a, x_resampled)
    y_filtered = filtfilt(b, a, y_resampled)
    z_filtered = filtfilt(b, a, z_resampled)
    pbar.update(1)

    # Step 4: Apply band-pass filter (0.29–1.63 Hz)
    pbar.set_postfix_str("Applying band-pass filter...")
    b, a = butter_bandpass(lowcut=0.29, highcut=1.63, fs=target_rate)
    x_bandpassed = filtfilt(b, a, x_filtered)
    y_bandpassed = filtfilt(b, a, y_filtered)
    z_bandpassed = filtfilt(b, a, z_filtered)
    pbar.update(1)

    # Step 5: Resample to 10 Hz
    pbar.set_postfix_str("Resampling to 10 Hz...")
    target_rate = 10
    resample_factor = target_rate / 30  # From 30 Hz to 10 Hz
    num_samples = int(len(x_bandpassed) * resample_factor)
    x_downsampled = resample(x_bandpassed, num_samples)
    y_downsampled = resample(y_bandpassed, num_samples)
    z_downsampled = resample(z_bandpassed, num_samples)
    pbar.update(1)

    # Step 6: Calculate vector magnitude
    pbar.set_postfix_str("Calculating vector magnitude...")
    vm = np.sqrt(x_downsampled**2 + y_downsampled**2 + z_downsampled**2)
    pbar.update(1)

    # Step 7: Apply dead-band threshold
    pbar.set_postfix_str("Applying dead-band threshold...")
    x_downsampled[x_downsampled < 0.068] = 0
    y_downsampled[y_downsampled < 0.068] = 0
    z_downsampled[z_downsampled < 0.068] = 0
    vm[vm < 0.068] = 0
    pbar.update(1)

    # Step 8: Cap at 2.13 g
    pbar.set_postfix_str("Capping at 2.13 g...")
    x_downsampled[x_downsampled > 2.13] = 2.13
    y_downsampled[y_downsampled > 2.13] = 2.13
    z_downsampled[z_downsampled > 2.13] = 2.13
    vm[vm > 2.13] = 2.13
    pbar.update(1)

    # Step 9: Convert to 8-bit resolution
    pbar.set_postfix_str("Converting to 8-bit resolution...")
    x_scaled = (x_downsampled / 2.13) * 128
    y_scaled = (y_downsampled / 2.13) * 128
    z_scaled = (z_downsampled / 2.13) * 128
    vm_scaled = (vm / 2.13) * 128

    x_scaled = np.round(x_scaled).astype(int)
    y_scaled = np.round(y_scaled).astype(int)
    z_scaled = np.round(z_scaled).astype(int)
    vm_scaled = np.round(vm_scaled).astype(int)
    pbar.update(1)

    # Step 10: Aggregate into 60-second epochs (600 samples at 10 Hz)
    pbar.set_postfix_str("Aggregating into 60-second epochs...")
    num_epochs = len(vm_scaled) // 600
    x_scaled = x_scaled[:num_epochs * 600].reshape(-1, 600)
    y_scaled = y_scaled[:num_epochs * 600].reshape(-1, 600)
    z_scaled = z_scaled[:num_epochs * 600].reshape(-1, 600)
    vm_scaled = vm_scaled[:num_epochs * 600].reshape(-1, 600)

    x_epoch_counts = np.sum(x_scaled, axis=1)
    y_epoch_counts = np.sum(y_scaled, axis=1)
    z_epoch_counts = np.sum(z_scaled, axis=1)
    # vm_epoch_counts = np.sum(vm_scaled, axis=1)
    pbar.update(1)

    # Step 11: Match timestamps to 60-second epochs
    pbar.set_postfix_str("Matching timestamps to epochs...")
    initial_timestamp = pd.to_datetime(timestamps.iloc[0])  # Start time
    epoch_duration = pd.Timedelta(seconds=60)
    epoch_timestamps = [initial_timestamp + i * epoch_duration for i in range(num_epochs)]
    pbar.update(1)

    # Step 12: Save results to CSV
    pbar.set_postfix_str("Saving final CSV...")
    output_df = pd.DataFrame({
        'dataTimestamp': epoch_timestamps,
        'axis1': x_epoch_counts,
        'axis2': y_epoch_counts,
        'axis3': z_epoch_counts,
        # 'vm_epoch_counts': vm_epoch_counts
    })
    output_df.to_csv(output_csv, index=False)
    pbar.update(1)

    pbar.close()
    print(f"Processing complete. Output saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Process raw Axivity data and generate activity counts with a progress bar."
    )
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
    args = parser.parse_args()

    try:
        process_axivity_data(args.input_csv, args.output_csv)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()