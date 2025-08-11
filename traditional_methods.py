import pandas as pd
import numpy as np

def load_combined_counts_as_single(df_combined):
    """
    Given a DataFrame whose first column is a timestamp and
    whose remaining columns are per‐limb epoch counts,
    average those count columns into a single 'count' column.
    Returns a new DataFrame with ['dataTimestamp','count'].
    """
    # Assume first column is the timestamp
    timestamp_col = df_combined.columns[0]
    count_cols = [c for c in df_combined.columns if c != timestamp_col]
    if not count_cols:
        raise ValueError("No count columns found (need at least one column beside the timestamp).")

    out = pd.DataFrame()
    out["dataTimestamp"] = df_combined[timestamp_col]
    out["count"] = df_combined[count_cols].mean(axis=1)
    return out

def apply_sadeh(df, output_file="sadeh_results.csv"):
    """
    11-minute Sadeh sleep/wake algorithm:
    WI = sum_{j=-5..+5} w_j * count.shift(j)
    Sleep if WI < 1, else Wake.
    Writes results to `output_file`.
    """
    weights = {
        -5:  0.020, -4: -0.004, -3: 0.034, -2: 0.468,
        -1:  0.345,  0:  0.568,  1: 0.321,  2: 0.035,
         3: -0.001,  4: -0.021,  5: 0.007
    }
    wi = np.zeros(len(df))
    for shift, w in weights.items():
        wi += w * df["count"].shift(shift, fill_value=0)

    result = df.copy()
    result["sleep_index"] = wi
    result["sleep"] = np.where(wi < 1.0, "S", "W")

    # Save to CSV
    result.to_csv(output_file, index=False)
    print(f"Sadeh results saved to {output_file}")

    return result

def apply_troiano(df, zero_thresh=60, spike_tol=2):
    """
    Troiano non-wear detection on 'count':
    Finds runs of ≥ zero_thresh consecutive zeros,
    allowing up to spike_tol non-zero epochs interspersed.
    Returns a list of (start, end) timestamps.
    """
    is_zero = (df["count"] == 0).astype(int)
    periods = []
    start = None
    spike_count = 0
    length = 0

    for t, z in zip(df["dataTimestamp"], is_zero):
        if z == 1 or spike_count < spike_tol:
            if start is None:
                start = t
            length += 1
            if z == 0:
                spike_count += 1
        else:
            if length >= zero_thresh:
                periods.append((start, t))
            start, length, spike_count = None, 0, 0

    # Final flush
    if start is not None and length >= zero_thresh:
        periods.append((start, df["dataTimestamp"].iloc[-1]))

    return periods

def apply_choi(df, zero_thresh=90, inner_tol=2):
    """
    Choi improved non-wear detection:
    Like Troiano but with stricter requirement around spikes.
    """
    return apply_troiano(df, zero_thresh=zero_thresh, spike_tol=inner_tol)


# ─── Combined wrappers ───────────────────────────────────────────────────────

def apply_sadeh_combined(df_combined):
    """
    Average the multi‐limb CSV into one 'count' series and apply Sadeh.
    """
    df_single = load_combined_counts_as_single(df_combined)
    return apply_sadeh(df_single)

def apply_troiano_combined(df_combined, zero_thresh=60, spike_tol=2):
    """
    Average the CSV, then detect non-wear with Troiano.
    """
    df_single = load_combined_counts_as_single(df_combined)
    return apply_troiano(df_single, zero_thresh=zero_thresh, spike_tol=spike_tol)

def apply_choi_combined(df_combined, zero_thresh=90, inner_tol=2):
    """
    Average the CSV, then detect non-wear with Choi.
    """
    df_single = load_combined_counts_as_single(df_combined)
    return apply_choi(df_single, zero_thresh=zero_thresh, inner_tol=inner_tol)
