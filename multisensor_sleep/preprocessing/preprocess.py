import argparse
import os
import sys
import re
import math
import datetime
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

##############################################################################
# 1. HELPER FUNCTIONS
##############################################################################

### Raw Data Processing Functions ###

# Precompile the regex pattern so it isn't compiled for every row. This results in a timeout otherwise.
TIME_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:)(\d{2})(\.\d+)?$")

def parse_time_column(time_str, baseline=None):
    """
    Parse a datetime string of the form '2025-02-03 21:13:10.260' into a pandas.Timestamp.
    If the seconds value is 60 or more, it is adjusted to '59.999' so that the string can be parsed.
    If a baseline is provided, returns the elapsed seconds (as a float) since that baseline. 
    The purpose of the baseline is to synchonize multiple sensors to the same time reference.
    
    Parameters:
      time_str (str): The timestamp string.
      baseline (pd.Timestamp, optional): A reference timestamp.
      
    Returns:
      pd.Timestamp or float: The parsed timestamp or the elapsed seconds since baseline.
    """
    match = TIME_PATTERN.match(time_str)
    if match:
        prefix, sec_str, frac = match.groups()
        frac = frac if frac is not None else ""
        sec_val = float(sec_str + frac)
        if sec_val >= 60:
            # Adjust seconds to 59.999 if they are 60 or greater
            time_str = prefix + "59" + ".999"
    # Explicitly specify the expected format to speed up parsing.
    dt = pd.to_datetime(time_str, format='%Y-%m-%d %H:%M:%S.%f', errors='raise')
    if baseline is not None:
        return (dt - baseline).total_seconds()
    return dt

def round_to_nearest_second(seconds):
    return int(round(seconds))


### Filter functions for signal processing in Actigraphy Count Processing ###

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

##############################################################################
# 2. ACTIGRAPHY COUNT PROCESSING FUNCTION
##############################################################################

def process_axivity_data(df, sampling_rate=100):
    """
    This function returns a new DataFrame with epoch-level actigraphy counts.
    - We round the initial timestamp to the nearest second to align with baseline.
    - The final output has columns: dataTimestamp, axis1, axis2, axis3.
    """
    
    # Use a progress bar for major steps
    total_steps = 12
    pbar = tqdm(total=total_steps, desc="Processing Axivity Data")

    # Step 1: Extract time and axes
    pbar.set_postfix_str("Extracting data...")
    timestamps = df['dataTimestamp'].values  # now numeric seconds relative to start
    x, y, z = df['axis1'].values, df['axis2'].values, df['axis3'].values
    pbar.update(1)

    # Step 2: Resample from original_rate to 30 Hz
    pbar.set_postfix_str("Resampling to 30 Hz...")
    original_rate = sampling_rate
    target_rate = 30
    resample_factor = target_rate / original_rate
    num_samples = int(len(x) * resample_factor)
    x_resampled = resample(x, num_samples)
    y_resampled = resample(y, num_samples)
    z_resampled = resample(z, num_samples)
    pbar.update(1)

    # Step 3: Apply low-pass filter (~15 Hz)
    pbar.set_postfix_str("Applying low-pass filter...")
    b, a = butter_lowpass(cutoff=14.9, fs=target_rate)  # slightly below 15 Hz
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
    resample_factor = target_rate / 30
    num_samples = int(len(x_bandpassed) * resample_factor)
    x_downsampled = resample(x_bandpassed, num_samples)
    y_downsampled = resample(y_bandpassed, num_samples)
    z_downsampled = resample(z_bandpassed, num_samples)
    pbar.update(1)

    # Step 6: Calculate vector magnitude (optional if needed)
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
    x_scaled = np.round((x_downsampled / 2.13) * 128).astype(int)
    y_scaled = np.round((y_downsampled / 2.13) * 128).astype(int)
    z_scaled = np.round((z_downsampled / 2.13) * 128).astype(int)
    vm_scaled = np.round((vm / 2.13) * 128).astype(int)
    pbar.update(1)

    # Step 10: Aggregate into 60-second epochs
    # At 10 Hz, 60 seconds = 600 samples
    pbar.set_postfix_str("Aggregating into 60-second epochs...")
    samples_per_epoch = 600
    num_epochs = len(vm_scaled) // samples_per_epoch

    x_scaled = x_scaled[:num_epochs * samples_per_epoch].reshape(-1, samples_per_epoch)
    y_scaled = y_scaled[:num_epochs * samples_per_epoch].reshape(-1, samples_per_epoch)
    z_scaled = z_scaled[:num_epochs * samples_per_epoch].reshape(-1, samples_per_epoch)

    x_epoch_counts = np.sum(x_scaled, axis=1)
    y_epoch_counts = np.sum(y_scaled, axis=1)
    z_epoch_counts = np.sum(z_scaled, axis=1)
    pbar.update(1)

    # Step 11: Create epoch timestamps
    # We'll treat the very first second in the raw data as our "start_time".
    # Then each epoch is offset by 60-second increments.
    pbar.set_postfix_str("Creating epoch timestamps...")
    start_time_seconds = timestamps[0]  # numeric seconds
    start_time_rounded = round_to_nearest_second(start_time_seconds)

    epoch_times = [start_time_rounded + 60 * i for i in range(num_epochs)]
    pbar.update(1)

    # Step 12: Build the output DataFrame
    pbar.set_postfix_str("Building output DataFrame...")
    output_df = pd.DataFrame({
        'dataTimestamp': epoch_times,
        'axis1': x_epoch_counts,
        'axis2': y_epoch_counts,
        'axis3': z_epoch_counts
    })
    pbar.update(1)

    pbar.close()
    return output_df

##############################################################################
# 3. MAIN PIPELINE
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: read 1–4 raw CSV files (headerless), process "
            "them into actigraphy counts, and optionally combine them."
        )
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to 1–4 raw CSV files (no headers)."
    )
    parser.add_argument(
        "-r", "--raw_rate",
        type=int,
        default=100,
        help="Sampling rate of raw files (default=100 Hz)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=None,
        help="Output directory name. If not provided, will create a timestamped folder."
    )

    args = parser.parse_args()
    raw_csv_files = args.input_files
    if len(raw_csv_files) < 1 or len(raw_csv_files) > 4:
        print("Error: You must provide between 1 and 4 raw CSV files.")
        sys.exit(1)

    # Create output directory in the ../test_data folder (See specified project file structure in README.md)
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../test_data")
    if args.output_dir is None:
        now_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"{now_str}")
    else:
        output_dir = os.path.join(base_output_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 3.1 Process each raw CSV => produce actigraphy counts
    processed_dataframes = []
    for i, file_path in enumerate(raw_csv_files, start=1):
        sensor_id = i
        # Read raw CSV (headers havent been added yet)
        try:
            df_raw = pd.read_csv(
                file_path,
                header=None,
                names=["dataTimestamp", "axis1", "axis2", "axis3"]
            )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            sys.exit(1)

        # Convert the first column to pd.Timestamp using the helper,
        # then convert to numeric seconds relative to the first timestamp.
        df_raw["dataTimestamp"] = df_raw["dataTimestamp"].apply(parse_time_column)
        baseline = df_raw["dataTimestamp"].iloc[0]
        df_raw["dataTimestamp"] = df_raw["dataTimestamp"].apply(lambda ts: (ts - baseline).total_seconds())

        # Run the pipeline
        processed_df = process_axivity_data(df_raw, sampling_rate=args.raw_rate)

        # Save the individual output
        sensor_output_path = os.path.join(output_dir, f"sensor_{sensor_id}_counts.csv")
        processed_df.to_csv(sensor_output_path, index=False)
        print(f"Sensor {sensor_id} processed counts => {sensor_output_path}")

        # Keep it for later combining
        # Rename axis columns to axis1_i, axis2_i, axis3_i
        rename_map = {
            "axis1": f"axis1_{sensor_id}",
            "axis2": f"axis2_{sensor_id}",
            "axis3": f"axis3_{sensor_id}"
        }
        df_for_merge = processed_df.rename(columns=rename_map)
        processed_dataframes.append(df_for_merge)

    # 3.2 If more than one CSV provided, merge them
    if len(processed_dataframes) > 1:
        merged_df = processed_dataframes[0]
        for df in processed_dataframes[1:]:
            merged_df = pd.merge(
                merged_df,
                df,
                on="dataTimestamp",
                how="inner"
            )

        # We only want columns for the sensors that exist.
        num_files = len(processed_dataframes)
        col_order = ["dataTimestamp"]
        for axis in ["axis1", "axis2", "axis3"]:
            for i in range(1, num_files + 1):
                col_order.append(f"{axis}_{i}")

        # Filter columns that actually exist
        col_order = [c for c in col_order if c in merged_df.columns]
        merged_df = merged_df[col_order]

        # Write the combined CSV
        combined_path = os.path.join(output_dir, "combined_counts.csv")
        merged_df.to_csv(combined_path, index=False)
        print(f"Combined actigraphy counts => {combined_path}")

    print("Done. All outputs are in:", output_dir)

if __name__ == "__main__":
    main()
