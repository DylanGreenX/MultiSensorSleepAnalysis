"""PSG validation CLI script."""

import argparse
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multisensor_sleep.validation.psg_parser import process_psg_data
from multisensor_sleep.validation.validation_utils import (
    extract_time_window, align_timestamps, calculate_validation_metrics,
    epoch_by_epoch_comparison
)

def main():
    parser = argparse.ArgumentParser(
        description="Validate actigraphy results against PSG ground truth data"
    )
    parser.add_argument('-a', '--actigraphy', required=True,
                       help='Path to actigraphy results CSV file')
    parser.add_argument('-p', '--psg', required=True,
                       help='Path to PSG data CSV file')
    parser.add_argument('--lights_out', required=True,
                       help='Lights out timestamp (ISO format)')
    parser.add_argument('--lights_on', required=True,
                       help='Lights on timestamp (ISO format)')
    parser.add_argument('-o', '--output', default='results/validation/',
                       help='Output directory (default: results/validation/)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("Loading actigraphy data...")
    try:
        actigraphy_df = pd.read_csv(args.actigraphy)
    except Exception as e:
        sys.exit(f"Error loading actigraphy file: {e}")

    print("Loading and processing PSG data...")
    psg_df = process_psg_data(args.psg)

    print("Aligning timestamps...")
    actigraphy_df, psg_df = align_timestamps(actigraphy_df, psg_df)

    print("Extracting time window...")
    actigraphy_windowed = extract_time_window(
        actigraphy_df, args.lights_out, args.lights_on, 'dataTimestamp'
    )
    psg_windowed = extract_time_window(
        psg_df, args.lights_out, args.lights_on, 'timestamp'
    )

    print("Performing epoch-by-epoch comparison...")
    comparison_df = epoch_by_epoch_comparison(actigraphy_windowed, psg_windowed)

    if len(comparison_df) == 0:
        sys.exit("Error: No overlapping epochs found between actigraphy and PSG data")

    print("Calculating validation metrics...")
    metrics, cm = calculate_validation_metrics(
        comparison_df['binary_sleep'], comparison_df['actigraphy_prediction']
    )

    # Save results
    print("Saving validation results...")

    # Metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.output, 'validation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)

    # Epoch comparison
    comparison_path = os.path.join(args.output, 'epoch_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)

    # Print summary
    print("\n=== Validation Results ===")
    print(f"Sensitivity (Sleep Detection): {metrics['sensitivity']:.3f}")
    print(f"Specificity (Wake Detection): {metrics['specificity']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"\nTotal epochs compared: {len(comparison_df)}")
    print(f"Matching epochs: {comparison_df['match'].sum()}")
    print(f"Agreement rate: {comparison_df['match'].mean():.3f}")

    print(f"\nResults saved to: {args.output}")
    print(f"- Metrics: {metrics_path}")
    print(f"- Epoch comparison: {comparison_path}")

if __name__ == "__main__":
    main()