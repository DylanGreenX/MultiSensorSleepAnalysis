import pandas as pd
import numpy as np
import os

def ensure_output_dir(filepath):
    """Ensure output directory exists for the given filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def format_time_column(time, baseline=None):
    """
    Convert an integer or float timestamp (elapsed seconds since baseline) into a formatted time string:
    'YYYY-MM-DD HH:MM:SS.fff' given a baseline
        
    Parameters:
        time (int or float): seconds since baseline.
        baseline (int or float): Reference time.
        
    Returns:
        str: Formatted timestamp string.
    """
    if isinstance(time, float) or isinstance(time, int):
        if baseline is None:
            raise ValueError("must provide baseline.")
        dt = baseline + pd.to_timedelta(time, unit='s')
    else:
        raise TypeError("time must be int or float (seconds since baseline).")
    
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Trim to milliseconds


# Actigraph adjustment function for each limb and axis
def actigraph_adjustment_sing(data):
    data['count'] = np.minimum(data['axis1'] / 100, 300)
    return data

def actigraph_adjustment_mult(data, column):
    # Scale and cap activity counts
    data[f'{column}_adjusted'] = np.minimum(data[column] / 100, 300)
    return data

def apply_cole_kripke_1min_sing(data):
    # Apply the sleep index formula using shift for lag and lead
    data['sleep_index'] = 0.001 * (
        106 * data['count'].shift(4, fill_value=0) +
        54 * data['count'].shift(3, fill_value=0) +
        58 * data['count'].shift(2, fill_value=0) +
        76 * data['count'].shift(1, fill_value=0) +
        230 * data['count'] +
        74 * data['count'].shift(-1, fill_value=0) +
        67 * data['count'].shift(-2, fill_value=0)
    )

    # Assign sleep state based on the sleep index
    data['sleep'] = np.where(data['sleep_index'] < 1, 'S', 'W')
    return data

def apply_cole_kripke_1min_mult(data, column):
    # Calculate sleep index using shifted activity counts
    data[f'{column}_sleep_index'] = 0.001 * (
        106 * data[f'{column}_adjusted'].shift(4, fill_value=0) +
        54 * data[f'{column}_adjusted'].shift(3, fill_value=0) +
        58 * data[f'{column}_adjusted'].shift(2, fill_value=0) +
        76 * data[f'{column}_adjusted'].shift(1, fill_value=0) +
        230 * data[f'{column}_adjusted'] +
        74 * data[f'{column}_adjusted'].shift(-1, fill_value=0) +
        67 * data[f'{column}_adjusted'].shift(-2, fill_value=0)
    )
    return data

def format_cole_kripke_output(data, num_limbs=4):
    """
    Build an output DataFrame with per‐limb indices + sleep,
    then append a weighted consensus sleep label.
    """
    output_data = pd.DataFrame()
    output_data['dataTimestamp'] = data['dataTimestamp']

    # 1) copy over each limb's score + label
    for limb in range(1, num_limbs + 1):
        output_data[f'Limb {limb} sleep_index'] = data[f'limb_{limb}_sleep_index']
        output_data[f'Limb {limb} sleep']        = data[f'limb_{limb}_sleep']

    # 2) weighted consensus vote
    #    limbs 2 & 4 (wrists) get weight=2; limbs 1 & 3 (ankles) get weight=1
    weights = {1: 1, 2: 2, 3: 1, 4: 2}

    def vote(row):
        w_sleep = sum(weights[i]
                      for i in range(1, num_limbs+1)
                      if row[f'Limb {i} sleep'] == 'S')
        w_wake  = sum(weights[i]
                      for i in range(1, num_limbs+1)
                      if row[f'Limb {i} sleep'] == 'W')
        return 'S' if w_sleep >= w_wake else 'W'

    output_data['sleep'] = output_data.apply(vote, axis=1)

    return output_data


def apply_cole_kripke_mult(data, num_limbs=4, output_file="results/algorithm_outputs/cole_mult_results.csv"):
    axes = ['axis1', 'axis2', 'axis3'] # x, y, z axes

    for limb in range(1, num_limbs + 1):
        limb_sleep_indices = []

        for axis in axes:
            column = f'{axis}_{limb}'
            # Apply the adjustment and Cole-Kripke for each axis of each limb
            data = actigraph_adjustment_mult(data, column)
            data = apply_cole_kripke_1min_mult(data, column)
            limb_sleep_indices.append(data[f'{column}_sleep_index'])

        # Combine sleep indices for the limb by averaging the values across axes
        data[f'limb_{limb}_sleep_index'] = sum(limb_sleep_indices) / len(limb_sleep_indices)

        # Assign sleep state for the limb based on the combined sleep index
        data[f'limb_{limb}_sleep'] = np.where(data[f'limb_{limb}_sleep_index'] < 1, 'S', 'W')
    
    # Convert timestamps back to the original
    baseline = pd.Timestamp("2025-02-03 21:00:00")
    if 'dataTimestamp' in data.columns:
        data['dataTimestamp'] = data['dataTimestamp'].apply(lambda sec: format_time_column(sec, baseline=baseline))

    # Format the output and save to a CSV file
    output_data = format_cole_kripke_output(data, num_limbs)
    ensure_output_dir(output_file)
    output_data.to_csv(output_file, index=False)
    print(f"Multi-sensor results saved to {output_file} (using {num_limbs} limbs)")
    return output_data
        

def apply_cole_kripke_single(data, output_file="results/algorithm_outputs/cole_single_results.csv"):
    data = actigraph_adjustment_sing(data)
    data = apply_cole_kripke_1min_sing(data)
    
    # Convert timestamps back to the original
    baseline = pd.Timestamp("2025-02-03 21:00:00")
    if 'dataTimestamp' in data.columns:
        data['dataTimestamp'] = data['dataTimestamp'].apply(lambda sec: format_time_column(sec, baseline=baseline))

    # Ensure `dataTimestamp` is retained
    output_columns = ['dataTimestamp', 'sleep_index', 'sleep']
    ensure_output_dir(output_file)
    data[output_columns].to_csv(output_file, index=False)
    print(f"Single-sensor results saved to {output_file}")
    return data

def apply_cole_kripke_mult_weighted(data, num_limbs=4, output_file="results/algorithm_outputs/cole_weighted_mult_results.csv"):
    axes = ['axis1', 'axis2', 'axis3']  # x, y, z axes

    for limb in range(1, num_limbs + 1):
        limb_sleep_indices = []

        for axis in axes:
            column = f'{axis}_{limb}'
            # Apply the adjustment and Cole-Kripke for each axis of each limb
            data = actigraph_adjustment_mult(data, column)
            data = apply_cole_kripke_1min_mult(data, column)
            limb_sleep_indices.append(data[f'{column}_sleep_index'])

        # Combine sleep indices for the limb by averaging the values across axes
        data[f'limb_{limb}_sleep_index'] = sum(limb_sleep_indices) / len(limb_sleep_indices)

        # Assign sleep state for the limb based on the combined sleep index
        data[f'limb_{limb}_sleep'] = np.where(data[f'limb_{limb}_sleep_index'] < 1, 'S', 'W')

    # Convert timestamps back to the original
    baseline = pd.Timestamp("2025-02-03 21:00:00")
    if 'dataTimestamp' in data.columns:
        data['dataTimestamp'] = data['dataTimestamp'].apply(lambda sec: format_time_column(sec, baseline=baseline))

    # Build per-limb + weighted consensus outputs
    output_data = pd.DataFrame()
    output_data['dataTimestamp'] = data['dataTimestamp']

    # 1) Copy limb sleep indices & labels
    for limb in range(1, num_limbs + 1):
        output_data[f'Limb {limb} sleep_index'] = data[f'limb_{limb}_sleep_index']
        output_data[f'Limb {limb} sleep'] = data[f'limb_{limb}_sleep']

    # 2) Compute a weighted consensus index
    #    ankles (1 & 3) get weight=0.5, wrists (2 & 4) = 1.0
    weights = {1: 0.5, 2: 2.0, 3: 0.5, 4: 2.0}
    total_w = sum(weights.values())
    output_data['consensus_index'] = sum(
        weights[i] * output_data[f'Limb {i} sleep_index'] for i in weights
    ) / total_w

    # 3) Threshold for a binary consensus_sleep ('S' if below 1, else 'W')
    thresh = 0.7
    output_data['consensus_sleep'] = np.where(
        output_data['consensus_index'] < thresh, 'S', 'W'
    )

    # Save and return
    ensure_output_dir(output_file)
    output_data.to_csv(output_file, index=False)
    print(f"Multi-sensor results saved to {output_file} (using {num_limbs} limbs)")
    return output_data

def apply_cole_kripke_mult_majority(
    data,
    num_limbs=4,
    output_file="results/algorithm_outputs/cole_majority_mult_results.csv",
):
    """
    Run the standard multi‐limb Cole–Kripke
    and then aggregate with a simple majority vote
    across limbs.
    """
    axes = ['axis1', 'axis2', 'axis3']  # x, y, z axes

    # 1) per‐limb Cole–Kripke
    for limb in range(1, num_limbs + 1):
        limb_sleep_indices = []
        for axis in axes:
            col = f'{axis}_{limb}'
            data = actigraph_adjustment_mult(data, col)
            data = apply_cole_kripke_1min_mult(data, col)
            limb_sleep_indices.append(data[f'{col}_sleep_index'])

        data[f'limb_{limb}_sleep_index'] = sum(limb_sleep_indices) / len(limb_sleep_indices)
        data[f'limb_{limb}_sleep'] = np.where(
            data[f'limb_{limb}_sleep_index'] < 1, 'S', 'W'
        )

    # 2) rewrite timestamps
    baseline = pd.Timestamp("2025-02-03 21:00:00")
    if 'dataTimestamp' in data.columns:
        data['dataTimestamp'] = data['dataTimestamp'].apply(
            lambda sec: format_time_column(sec, baseline=baseline)
        )

    # 3) build output + majority vote
    output_data = pd.DataFrame({
        'dataTimestamp': data['dataTimestamp']
    })
    for limb in range(1, num_limbs + 1):
        output_data[f'Limb {limb} sleep_index'] = data[f'limb_{limb}_sleep_index']
        output_data[f'Limb {limb} sleep']        = data[f'limb_{limb}_sleep']

    # majority‐vote consensus: S if ≥ half limbs say S
    label_cols = [f'Limb {i} sleep' for i in range(1, num_limbs+1)]
    threshold = num_limbs // 2 + 1
    output_data['consensus_majority'] = output_data[label_cols].apply(
        lambda row: 'S' if (row == 'S').sum() >= threshold else 'W',
        axis=1
    )

    # 4) save & return
    ensure_output_dir(output_file)
    output_data.to_csv(output_file, index=False)
    print(f"Multi-sensor majority‐vote results saved to {output_file}")
    return output_data
