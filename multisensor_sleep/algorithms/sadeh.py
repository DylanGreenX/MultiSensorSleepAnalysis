import pandas as pd
import numpy as np
import os
from scipy.ndimage import uniform_filter1d

def ensure_output_dir(filepath):
    """Ensure output directory exists for the given filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

#Sadeh Index calculation helper functions
def roll_mean(x, window):
    """Compute the rolling mean with a specified window size, using padding."""
    return uniform_filter1d(x, size=window, mode='constant', origin=-(window // 2))

def roll_std(x, window):
    """Compute the rolling standard deviation over a trailing window.

    For Sadeh (1994), SD is computed over the 6 epochs ending at
    the current epoch (current + 5 preceding).
    """
    padded_x = np.pad(x, (window, 0), 'constant', constant_values=0)
    return pd.Series(padded_x).rolling(window=window + 1, min_periods=1).std().values[window:]

def roll_nats(x, window):
    """Count epochs with activity between 50 and 100 in a rolling window."""
    y = np.where((x >= 50) & (x < 100), 1, 0)
    return uniform_filter1d(y, size=window, mode='constant', origin=-(window // 2))

#Sadeh Index calculation function
def sadeh_index(counts, half_window=5):
    """Compute the Sadeh (1994) sleep index for a single count series.

    Features (per the original paper, p. 203):
        MEAN — rolling mean over an 11-epoch centered window
        NAT  — number of epochs with activity in [50, 100) in the same window
        SD   — standard deviation over a 6-epoch trailing window
        LG   — natural log of the current epoch's activity + 1

    Formula:
        PS = 7.601 - 0.065*MEAN - 1.08*NAT - 0.056*SD - 0.703*LG

    Reference:
        Sadeh, A., Sharkey, K. M., & Carskadon, M. A. (1994).
        Activity-based sleep-wake identification: An empirical test
        of methodological issues. Sleep, 17(3), 201-207.
    """
    full_window = 2 * half_window + 1  # 11 epochs

    avg = roll_mean(counts, window=full_window)
    sd = roll_std(counts, window=half_window)
    nats = roll_nats(counts, window=full_window)

    ps = (7.601
          - 0.065 * avg
          - 1.08 * nats
          - 0.056 * sd
          - 0.703 * np.log(counts + 1))

    return ps

def compute_per_limb_sadeh(data, num_limbs=4):
    axes = ['axis1', 'axis2', 'axis3']

    for limb in range(1, num_limbs + 1):
        limb_sleep_indices = []

        for axis in axes:
            column = f'{axis}_{limb}'
            counts = np.minimum(data[column].values.astype(float), 300)
            data[f'{column}_sleep_index'] = sadeh_index(counts)
            limb_sleep_indices.append(data[f'{column}_sleep_index'])

        # Average across axes for this limb
        data[f'limb_{limb}_sleep_index'] = sum(limb_sleep_indices) / len(limb_sleep_indices)
        data[f'limb_{limb}_sleep'] = np.where(
            data[f'limb_{limb}_sleep_index'] >= 0, 'S', 'W'
        )

    return data

def format_per_limb_output(data, num_limbs=4):
    output_data = pd.DataFrame()

    if 'dataTimestamp' in data.columns:
        output_data['dataTimestamp'] = data['dataTimestamp']

    for limb in range(1, num_limbs + 1):
        output_data[f'Limb {limb} sleep_index'] = data[f'limb_{limb}_sleep_index']
        output_data[f'Limb {limb} sleep'] = data[f'limb_{limb}_sleep']

    return output_data

def load_combined_counts_as_single(df_combined):
    timestamp_col = df_combined.columns[0]
    count_cols = [c for c in df_combined.columns if c != timestamp_col]
    if not count_cols:
        raise ValueError("No count columns found (need at least one column beside the timestamp).")

    out = pd.DataFrame()
    out["dataTimestamp"] = df_combined[timestamp_col]
    out["count"] = df_combined[count_cols].mean(axis=1)
    return out


def apply_sadeh_single(df, output_file="results/algorithm_outputs/sadeh_results.csv"):
    """
    Sadeh (1994) single-sensor sleep/wake algorithm.

    Computes the Sadeh sleep index (PS) from four statistical features
    of a single activity count series and classifies each epoch:
        Sleep if PS >= 0, Wake if PS < 0.
    """
    if 'count' not in df.columns and 'axis1' in df.columns:
        df = df.copy()
        df['count'] = np.minimum(df['axis1'], 300)
    elif 'count' not in df.columns:
        raise ValueError("DataFrame must contain a 'count' or 'axis1' column.")

    counts = df['count'].values.astype(float)

    result = df.copy()
    result['sleep_index'] = sadeh_index(counts)
    result['sleep'] = np.where(result['sleep_index'] >= 0, 'S', 'W')

    ensure_output_dir(output_file)
    result.to_csv(output_file, index=False)
    print(f"Sadeh results saved to {output_file}")

    return result


def apply_sadeh_combined(df_combined):
    """Average the multi-limb CSV into one 'count' series and apply Sadeh."""
    df_single = load_combined_counts_as_single(df_combined)
    return apply_sadeh_single(df_single)

def apply_sadeh_vote(data, num_limbs=4,
                     output_file="results/algorithm_outputs/sadeh_vote_results.csv"):
    """
    Per-limb Sadeh (1994) with anatomically-weighted vote on binary labels.

    Each limb is classified independently (Sleep if index >= 0), then
    binary S/W labels are combined via weighted vote (wrists=1.0, ankles=0.5).
    Sleep if weighted sleep votes >= weighted wake votes.

    Note: information is lost at the binarization step — a limb with
    PS = 0.01 and PS = 5.0 both vote 'S'. Compare with apply_sadeh_weighted
    which preserves index magnitude.
    """
    data = compute_per_limb_sadeh(data, num_limbs)
    output_data = format_per_limb_output(data, num_limbs)

    # Weighted vote on per-limb binary labels
    weights = {1: 0.5, 2: 1, 3: 0.5, 4: 1}

    def vote(row):
        w_sleep = sum(weights[i] for i in range(1, num_limbs + 1)
                      if row[f'Limb {i} sleep'] == 'S')
        w_wake = sum(weights[i] for i in range(1, num_limbs + 1)
                     if row[f'Limb {i} sleep'] == 'W')
        return 'S' if w_sleep >= w_wake else 'W'

    output_data['sleep'] = output_data.apply(vote, axis=1)

    ensure_output_dir(output_file)
    output_data.to_csv(output_file, index=False)
    print(f"Sadeh weighted vote results saved to {output_file}")

    return output_data


def apply_sadeh_weighted(data, num_limbs=4,
                         output_file="results/algorithm_outputs/sadeh_weighted_results.csv"):
    """
    Per-limb Sadeh (1994) with anatomically-weighted consensus index.

    Per-limb continuous Sadeh indices are combined via weighted average
    (wrists=1.0, ankles=0.5), preserving magnitude information.
    Sleep if consensus index >= 0.
    """
    data = compute_per_limb_sadeh(data, num_limbs)
    output_data = format_per_limb_output(data, num_limbs)

    # Weighted consensus on continuous indices
    weights = {1: 0.5, 2: 1, 3: 0.5, 4: 1}
    total_w = sum(weights[i] for i in range(1, num_limbs + 1))
    output_data['consensus_index'] = sum(
        weights[i] * output_data[f'Limb {i} sleep_index'] for i in range(1, num_limbs + 1)
    ) / total_w

    output_data['consensus_sleep'] = np.where(
        output_data['consensus_index'] >= 0, 'S', 'W'
    )

    ensure_output_dir(output_file)
    output_data.to_csv(output_file, index=False)
    print(f"Sadeh weighted consensus results saved to {output_file}")

    return output_data


def apply_sadeh_majority(data, num_limbs=4,
                         output_file="results/algorithm_outputs/sadeh_majority_results.csv"):
    """
    Per-limb Sadeh (1994) with unweighted majority vote.

    Each limb votes independently (Sleep if index >= 0).
    Final classification: Sleep if >= 3 of 4 limbs vote Sleep.
    All limbs weighted equally regardless of anatomy.
    """
    data = compute_per_limb_sadeh(data, num_limbs)
    output_data = format_per_limb_output(data, num_limbs)

    # Majority vote: Sleep if >= 3/4 limbs say Sleep
    label_cols = [f'Limb {i} sleep' for i in range(1, num_limbs + 1)]
    threshold = num_limbs // 2 + 1
    output_data['consensus_majority'] = output_data[label_cols].apply(
        lambda row: 'S' if (row == 'S').sum() >= threshold else 'W',
        axis=1
    )

    ensure_output_dir(output_file)
    output_data.to_csv(output_file, index=False)
    print(f"Sadeh majority vote results saved to {output_file}")

    return output_data
