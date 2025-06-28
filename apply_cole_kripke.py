import pandas as pd
import numpy as np

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
    # Collect only the limb-level sleep classifications in the output
    output_data = pd.DataFrame()
    output_data['dataTimestamp'] = data['dataTimestamp']  # Keep timestamps

    for limb in range(1, num_limbs + 1):
        output_data[f'Limb {limb} sleep_index'] = data[f'limb_{limb}_sleep_index']
        output_data[f'Limb {limb} sleep'] = data[f'limb_{limb}_sleep']

    return output_data


def apply_cole_kripke_mult(data, num_limbs=4, output_file="cole_mult_results.csv"):
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
    output_data.to_csv(output_file, index=False)
    print(f"Multi-sensor results saved to {output_file} (using {num_limbs} limbs)")
    return output_data
        

def apply_cole_kripke_single(data, output_file="cole_single_results.csv"):
    data = actigraph_adjustment_sing(data)
    data = apply_cole_kripke_1min_sing(data)
    
    # Convert timestamps back to the original
    baseline = pd.Timestamp("2025-02-03 21:00:00")
    if 'dataTimestamp' in data.columns:
        data['dataTimestamp'] = data['dataTimestamp'].apply(lambda sec: format_time_column(sec, baseline=baseline))

    # Ensure `dataTimestamp` is retained
    output_columns = ['dataTimestamp', 'sleep_index', 'sleep']
    data[output_columns].to_csv(output_file, index=False)
    print(f"Single-sensor results saved to {output_file}")
    return data
