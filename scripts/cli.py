import argparse
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Run a specified algorithm on a given data file.')
    parser.add_argument('-a', '--algorithm', type=str, required=True,
                        help='Algorithm to run: C, CV, CW, CM, S, SV, SW, SM')
    parser.add_argument('-l', '--limbs', type=int, required=True,
                        help='Number of limbs (e.g., 1-4)')
    parser.add_argument('-d', '--datafile', type=str, required=True,
                        help='Path to the data file (e.g., combined_counts.csv)')
    parser.add_argument('-b', '--baseline', type=str, default=None,
                        help='Recording start timestamp, e.g. "2025-11-08 21:00:00". '
                             'Required for Cole-Kripke algorithms (C, CV, CW, CM) to '
                             'convert elapsed seconds back to real timestamps.')

    args = parser.parse_args()

    # Parse baseline if provided
    baseline = None
    if args.baseline:
        try:
            baseline = pd.Timestamp(args.baseline)
        except ValueError:
            sys.exit(f"Error: Could not parse baseline '{args.baseline}'. "
                     "Use format: YYYY-MM-DD HH:MM:SS")

    # Load the data file
    try:
        df = pd.read_csv(args.datafile)
    except FileNotFoundError:
        sys.exit(f"Error: The data file '{args.datafile}' was not found.")
    except pd.errors.EmptyDataError:
        sys.exit(f"Error: The data file '{args.datafile}' is empty.")
    except Exception as e:
        sys.exit(f"Error reading '{args.datafile}': {e}")

    # Cole-Kripke algorithms require a baseline
    cole_kripke_algos = {'C', 'CV', 'CW', 'CM'}
    if args.algorithm in cole_kripke_algos and baseline is None:
        sys.exit(f"Error: --baseline is required for Cole-Kripke algorithm '{args.algorithm}'. "
                 "Provide the recording start time, e.g. -b \"2025-11-08 21:00:00\"")

    # Run the selected algorithm
    if args.algorithm == 'C':
        from multisensor_sleep.algorithms.cole_kripke import apply_cole_kripke_single
        result = apply_cole_kripke_single(df, baseline)

    elif args.algorithm == 'CV':
        from multisensor_sleep.algorithms.cole_kripke import apply_cole_kripke_vote
        result = apply_cole_kripke_vote(df, baseline, args.limbs)

    elif args.algorithm == 'CW':
        from multisensor_sleep.algorithms.cole_kripke import apply_cole_kripke_weighted
        result = apply_cole_kripke_weighted(df, baseline, args.limbs)

    elif args.algorithm == 'CM':
        from multisensor_sleep.algorithms.cole_kripke import apply_cole_kripke_majority
        result = apply_cole_kripke_majority(df, baseline, args.limbs)

    elif args.algorithm == 'S':
        from multisensor_sleep.algorithms.sadeh import apply_sadeh_combined
        result = apply_sadeh_combined(df)

    elif args.algorithm == 'SV':
        from multisensor_sleep.algorithms.sadeh import apply_sadeh_vote
        result = apply_sadeh_vote(df, args.limbs)

    elif args.algorithm == 'SW':
        from multisensor_sleep.algorithms.sadeh import apply_sadeh_weighted
        result = apply_sadeh_weighted(df, args.limbs)

    elif args.algorithm == 'SM':
        from multisensor_sleep.algorithms.sadeh import apply_sadeh_majority
        result = apply_sadeh_majority(df, args.limbs)

    else:
        sys.exit(
            f"Error: Unknown algorithm '{args.algorithm}'. "
            "Supported values are: C, CV, CW, CM, S, SV, SW, SM."
        )

    print("Algorithm Output:")
    print(result)

if __name__ == "__main__":
    main()
