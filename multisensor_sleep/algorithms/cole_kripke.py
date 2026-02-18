import pandas as pd
import numpy as np
import os


def ensure_output_dir(filepath):
    """Ensure output directory exists for the given filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Timestamp conversion helper function
def format_timestamp(seconds, baseline):
    dt = baseline + pd.to_timedelta(seconds, unit='s')
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def _restore_timestamps(data, baseline):
    """Convert the dataTimestamp column from elapsed seconds to real timestamps.

    Modifies data in-place and returns it.
    """
    if 'dataTimestamp' in data.columns:
        data['dataTimestamp'] = data['dataTimestamp'].apply(
            lambda sec: format_timestamp(sec, baseline)
        )
    return data

# Cole-Kripke Index calculation function
def cole_kripke_index(counts):
    """Compute the Cole-Kripke (1992) sleep index for a single count series.

    Preprocessing:
        Activity counts are scaled by 1/100 and capped at 300, consistent
        with the ActiGraph count normalization used in the original study.

    Algorithm:
        A weighted linear combination over a 7-epoch window centered on
        the current epoch (4 preceding, current, 2 following):

        SI = 0.001 * (106*C[t-4] + 54*C[t-3] + 58*C[t-2] +
                       76*C[t-1] + 230*C[t] + 74*C[t+1] + 67*C[t+2])

        Sleep if SI < 1.0, Wake otherwise.

    Reference:
        Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., & Gillin, J. C.
        (1992). Automatic sleep/wake identification from wrist activity.
        Sleep, 15(5), 461-469.
    """
    scaled = np.minimum(np.asarray(counts, dtype=float) / 100, 300)
    s = pd.Series(scaled)

    si = 0.001 * (
        106 * s.shift(4, fill_value=0) +
         54 * s.shift(3, fill_value=0) +
         58 * s.shift(2, fill_value=0) +
         76 * s.shift(1, fill_value=0) +
        230 * s +
         74 * s.shift(-1, fill_value=0) +
         67 * s.shift(-2, fill_value=0)
    )

    return si.values


# Shared per-limb computation helper function
def compute_per_limb(data, num_limbs=4):
    axes = ['axis1', 'axis2', 'axis3']

    for limb in range(1, num_limbs + 1):
        limb_sleep_indices = []

        for axis in axes:
            column = f'{axis}_{limb}'
            data[f'{column}_sleep_index'] = cole_kripke_index(data[column])
            limb_sleep_indices.append(data[f'{column}_sleep_index'])

        # Average across axes for this limb
        data[f'limb_{limb}_sleep_index'] = sum(limb_sleep_indices) / len(limb_sleep_indices)
        data[f'limb_{limb}_sleep'] = np.where(
            data[f'limb_{limb}_sleep_index'] < 1, 'S', 'W'
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

def apply_cole_kripke_single(data, baseline,
                             output_file="results/algorithm_outputs/cole_single_results.csv"):
    """
    Cole-Kripke (1992) single-sensor sleep/wake algorithm.

    Computes the sleep index from the 'axis1' column and classifies
    each epoch: Sleep if SI < 1.0, Wake otherwise.
    """
    data = data.copy()
    data['sleep_index'] = cole_kripke_index(data['axis1'])
    data['sleep'] = np.where(data['sleep_index'] < 1, 'S', 'W')

    _restore_timestamps(data, baseline)

    output_columns = ['dataTimestamp', 'sleep_index', 'sleep']
    ensure_output_dir(output_file)
    data[output_columns].to_csv(output_file, index=False)
    print(f"Single-sensor results saved to {output_file}")
    return data


# Multi-sensor Cole-Kripke variants helper functions

def apply_cole_kripke_vote(data, baseline, num_limbs=4,
                           output_file="results/algorithm_outputs/cole_vote_results.csv"):
    """
    Per-limb Cole-Kripke with anatomically-weighted vote on binary labels.

    Each limb is classified independently (Sleep if index < 1.0), then
    binary S/W labels are combined via weighted vote (wrists=1.0, ankles=0.5).
    Sleep if weighted sleep votes >= weighted wake votes.

    Note: information is lost at the binarization step — a limb barely
    sleeping (index 0.99) and deeply sleeping (index 0.01) both vote 'S'.
    Compare with apply_cole_kripke_weighted which preserves index magnitude.
    """
    data = compute_per_limb(data, num_limbs)
    _restore_timestamps(data, baseline)

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
    print(f"Cole-Kripke weighted vote results saved to {output_file}")
    return output_data


def apply_cole_kripke_weighted(data, baseline, num_limbs=4,
                               output_file="results/algorithm_outputs/cole_weighted_results.csv"):
    """
    Per-limb Cole-Kripke with anatomically-weighted consensus index.

    Per-limb continuous sleep indices are combined via weighted average
    (wrists=1.0, ankles=0.5), preserving magnitude information.
    Sleep if consensus index < 0.7.
    """
    data = compute_per_limb(data, num_limbs)
    _restore_timestamps(data, baseline)

    output_data = format_per_limb_output(data, num_limbs)

    # Weighted consensus on continuous indices
    weights = {1: 0.5, 2: 1, 3: 0.5, 4: 1}
    total_w = sum(weights[i] for i in range(1, num_limbs + 1))
    output_data['consensus_index'] = sum(
        weights[i] * output_data[f'Limb {i} sleep_index'] for i in range(1, num_limbs + 1)
    ) / total_w

    output_data['consensus_sleep'] = np.where(
        output_data['consensus_index'] < 0.7, 'S', 'W'
    )

    ensure_output_dir(output_file)
    output_data.to_csv(output_file, index=False)
    print(f"Cole-Kripke weighted consensus results saved to {output_file}")
    return output_data


def apply_cole_kripke_majority(data, baseline, num_limbs=4,
                               output_file="results/algorithm_outputs/cole_majority_results.csv"):
    """
    Per-limb Cole-Kripke with unweighted majority vote.

    Each limb votes independently (Sleep if index < 1.0).
    Final classification: Sleep if >= 3 of 4 limbs vote Sleep.
    All limbs weighted equally regardless of anatomy.
    """
    data = compute_per_limb(data, num_limbs)
    _restore_timestamps(data, baseline)

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
    print(f"Cole-Kripke majority vote results saved to {output_file}")
    return output_data
