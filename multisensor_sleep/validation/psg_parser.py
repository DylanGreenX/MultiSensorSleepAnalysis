"""PSG data parser module for handling polysomnography data."""

import pandas as pd
import sys
import os

def ensure_output_dir(filepath):
    """Ensure output directory exists for the given filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def load_psg_csv(filepath):
    """Load PSG CSV file with error handling."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        sys.exit(f"Error: PSG file '{filepath}' not found.")
    except pd.errors.EmptyDataError:
        sys.exit(f"Error: PSG file '{filepath}' is empty.")
    except Exception as e:
        sys.exit(f"Error reading PSG file '{filepath}': {e}")

def validate_psg_format(df):
    """Ensure PSG data has proper timestamp and sleep stage columns."""
    required_columns = ['timestamp', 'sleep_stage']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        sys.exit(f"Error: PSG data missing required columns: {missing_columns}")

    return True

def map_sleep_stages(stage):
    """Convert PSG stages to binary Sleep/Wake classification."""
    stage_mapping = {
        'N1': 'Sleep',
        'N2': 'Sleep',
        'N3': 'Sleep',
        'N4': 'Sleep',
        'REM': 'Sleep',
        'Wake': 'Wake',
        'W': 'Wake'
    }

    return stage_mapping.get(stage, 'Unknown')

def process_psg_data(filepath):
    """Load and process PSG data for validation."""
    # Load data
    df = load_psg_csv(filepath)

    # Validate format
    validate_psg_format(df)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Map sleep stages to binary classification
    df['binary_sleep'] = df['sleep_stage'].apply(map_sleep_stages)

    # Check for unknown stages
    unknown_stages = df[df['binary_sleep'] == 'Unknown']['sleep_stage'].unique()
    if len(unknown_stages) > 0:
        print(f"Warning: Unknown sleep stages found: {unknown_stages}")

    return df