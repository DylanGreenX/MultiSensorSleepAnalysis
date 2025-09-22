import pandas as pd
import numpy as np

# Helper function to calculate run-length encoding (RLE) for non-wear periods
def rleid(series):
    """Run Length Encoding ID - assigns a unique id for consecutive values in a series."""
    return (series != series.shift()).cumsum()

# Helper function to add magnitude if needed (use axis values to calculate magnitude)
def add_magnitude(data):
    data['magnitude'] = np.sqrt(data['axis1']**2 + data['axis2']**2 + data['axis3']**2)
    return data

# Main Choi algorithm to detect non-wear periods
def apply_choi(data, min_period_len=90, min_window_len=30, spike_tolerance=2, use_magnitude=False):
    # Add magnitude if required
    data = add_magnitude(data) if use_magnitude else data
    data['count'] = data['magnitude'] if use_magnitude else data['axis1']
    
    # Mark wear based on counts
    data['wear'] = (data['count'] > 0).astype(int)
    
    # Group by run-length encoding (rleid) to find consecutive wear/non-wear periods
    data['rleid'] = rleid(data['wear'])
    
    # Summarize periods
    summary = data.groupby('rleid').agg(
        wear=('wear', 'first'),
        timestamp=('dataTimestamp', 'first'),
        length=('wear', 'size')
    ).reset_index(drop=True)
    
    # Step: Remove small spikes of non-wear (adjusting spikes using spike tolerance)
    summary['wear'] = np.where(
        (summary['wear'] == 0) & (summary['length'] < spike_tolerance), 1, summary['wear']
    )
    
    # Recalculate the run-length encoding after spike adjustment
    summary['rleid'] = rleid(summary['wear'])
    
    # Summarize periods again after spike adjustment
    summary = summary.groupby('rleid').agg(
        wear=('wear', 'first'),
        timestamp=('timestamp', 'first'),
        length=('length', 'sum')
    ).reset_index(drop=True)
    
    # Filter for non-wear periods that meet the minimum length requirement
    nonwear = summary[(summary['wear'] == 0) & (summary['length'] >= min_period_len)].copy()
    
    # Calculate period end time
    nonwear['period_end'] = nonwear['timestamp'] + nonwear['length'] * 60  # Assuming length is in minutes
    
    # Select relevant columns
    nonwear = nonwear[['timestamp', 'period_end', 'length']]
    
    return nonwear
