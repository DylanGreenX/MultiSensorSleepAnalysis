##### This algorithm is unlikely to be used for our study. Initially adapted it from the R.file in case it would be needed.


import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d

# Helper functions for rolling windows
def roll_mean(x, window):
    """Compute the rolling mean with a specified window size, using padding."""
    return uniform_filter1d(x, size=window, mode='constant', origin=-(window//2))

def roll_std(x, window):
    """Compute the rolling standard deviation with a specified window size, using padding."""
    padded_x = np.pad(x, (window, 0), 'constant', constant_values=0)
    return pd.Series(padded_x).rolling(window=window+1, min_periods=1).std().values[window:]

def roll_nats(x, window):
    """Count the number of epochs with activity between 50 and 100 in a rolling window."""
    y = np.where((x >= 50) & (x < 100), 1, 0)
    return uniform_filter1d(y, size=window, mode='constant', origin=-(window//2))

# Main function to apply the Sadeh algorithm
def apply_sadeh_single(data):
    half_window = 5  # Window size of 11 epochs (5 preceding, 5 following)

    # Adjust counts: cap axis1 values at 300
    data['count'] = np.minimum(data['axis1'], 300)

    # Compute the rolling features
    data['roll_avg'] = roll_mean(data['count'], window=2 * half_window + 1)
    data['roll_std'] = roll_std(data['count'], window=half_window)
    data['roll_nats'] = roll_nats(data['count'], window=2 * half_window + 1)

    # Compute the sleep index (SI) based on the Sadeh formula
    data['sleep_index'] = (
        7.601 
        - 0.065 * data['roll_avg']
        - 1.08 * data['roll_nats']
        - 0.056 * data['roll_std']
        - 0.703 * np.log(data['count'] + 1)
    )

    # Assign sleep state based on the sleep index: S = Sleep, W = Wake
    data['sleep'] = np.where(data['sleep_index'] > -4, 'S', 'W')

    return data


def format_sadeh_output(data):
    # Collect only the limb-level sleep classifications in the output
    output_data = pd.DataFrame()
    num_limbs = 4  # Assuming 4 limbs

    # For each limb, gather combined sleep index and sleep state
    for limb in range(1, num_limbs + 1):
        output_data[f'Limb {limb} sleep_index'] = data[f'limb_{limb}_sleep_index']
        output_data[f'Limb {limb} sleep'] = data[f'limb_{limb}_sleep']

    return output_data

def apply_sadeh_mult(data):
    half_window = 5  # Window size of 11 epochs (5 preceding, 5 following)
    axes = ['axis1', 'axis2', 'axis3']  # x, y, z axes
    num_limbs = 4  # Assuming 4 limbs

    for limb in range(1, num_limbs + 1):
        limb_sleep_indices = []  # To store sleep indices for each axis of the limb

        for axis in axes:
            column = f'{axis}_{limb}'
            
            # Adjust counts: cap values at 300
            data[f'{column}_count'] = np.minimum(data[column], 300)

            # Compute the rolling features
            data[f'{column}_roll_avg'] = roll_mean(data[f'{column}_count'], window=2 * half_window + 1)
            data[f'{column}_roll_std'] = roll_std(data[f'{column}_count'], window=half_window)
            data[f'{column}_roll_nats'] = roll_nats(data[f'{column}_count'], window=2 * half_window + 1)

            # Compute the sleep index (SI) based on the Sadeh formula
            data[f'{column}_sleep_index'] = (
                7.601 
                - 0.065 * data[f'{column}_roll_avg']
                - 1.08 * data[f'{column}_roll_nats']
                - 0.056 * data[f'{column}_roll_std']
                - 0.703 * np.log(data[f'{column}_count'] + 1)
            )

            # Append each axis's sleep index to the list for the limb
            limb_sleep_indices.append(data[f'{column}_sleep_index'])

        # Combine sleep indices for the limb by averaging the values across axes
        data[f'limb_{limb}_sleep_index'] = sum(limb_sleep_indices) / len(limb_sleep_indices)

        # Assign sleep state for the limb based on the combined sleep index
        data[f'limb_{limb}_sleep'] = np.where(data[f'limb_{limb}_sleep_index'] > -4, 'S', 'W')

    return format_sadeh_output(data)