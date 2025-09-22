"""Validation utilities for PSG-actigraphy comparison."""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys

def extract_time_window(df, lights_out, lights_on, timestamp_col='timestamp'):
    """Filter data to lights-out to lights-on time window."""
    # Parse timestamps if they're strings
    lights_out = pd.to_datetime(lights_out)
    lights_on = pd.to_datetime(lights_on)

    # Filter data
    mask = (df[timestamp_col] >= lights_out) & (df[timestamp_col] <= lights_on)
    windowed_df = df[mask].copy()

    if len(windowed_df) == 0:
        sys.exit(f"Error: No data found in time window {lights_out} to {lights_on}")

    return windowed_df

def align_timestamps(actigraphy_df, psg_df):
    """Synchronize timestamp formats between actigraphy and PSG data."""
    # Ensure both timestamp columns are datetime
    actigraphy_df['dataTimestamp'] = pd.to_datetime(actigraphy_df['dataTimestamp'])
    psg_df['timestamp'] = pd.to_datetime(psg_df['timestamp'])

    return actigraphy_df, psg_df

def calculate_validation_metrics(y_true, y_pred):
    """Compute sensitivity, specificity, F1 score and other metrics."""
    # Convert to binary (0=Sleep, 1=Wake)
    y_true_binary = (y_true == 'Wake').astype(int)
    y_pred_binary = (y_pred == 'W').astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }

    return metrics, cm

def generate_confusion_matrix(y_true, y_pred):
    """Create confusion matrix for sleep/wake classification."""
    y_true_binary = (y_true == 'Wake').astype(int)
    y_pred_binary = (y_pred == 'W').astype(int)

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    return cm

def epoch_by_epoch_comparison(actigraphy_df, psg_df):
    """Match corresponding epochs between actigraphy and PSG data."""
    # This function will handle the 60s -> 30s epoch splitting logic
    # For now, we'll implement simple duplication approach

    # Create expanded actigraphy data (each 60s epoch becomes two 30s epochs)
    expanded_acti = []

    for _, row in actigraphy_df.iterrows():
        timestamp = row['dataTimestamp']
        prediction = row['sleep']

        # Create two 30-second epochs from each 60-second epoch
        epoch1_time = timestamp
        epoch2_time = timestamp + pd.Timedelta(seconds=30)

        expanded_acti.append({
            'timestamp': epoch1_time,
            'actigraphy_prediction': prediction
        })
        expanded_acti.append({
            'timestamp': epoch2_time,
            'actigraphy_prediction': prediction
        })

    expanded_acti_df = pd.DataFrame(expanded_acti)

    # Merge with PSG data on timestamp
    comparison_df = pd.merge(
        expanded_acti_df,
        psg_df[['timestamp', 'binary_sleep']],
        on='timestamp',
        how='inner'
    )

    # Add match column
    comparison_df['match'] = (
        (comparison_df['actigraphy_prediction'] == 'S') & (comparison_df['binary_sleep'] == 'Sleep')
    ) | (
        (comparison_df['actigraphy_prediction'] == 'W') & (comparison_df['binary_sleep'] == 'Wake')
    )

    return comparison_df