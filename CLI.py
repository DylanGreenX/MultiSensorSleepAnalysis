import argparse
import pandas as pd
import sys

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Run a specified algorithm on a given data file.')
    parser.add_argument('-a', '--algorithm', type=str, required=True,
                        help='Algorithm to run (e.g., O, S, C, SM, CM)')
    parser.add_argument('-l', '--limbs', type=int, required=True,
                        help='Number of limbs (e.g., 1-4)')
    parser.add_argument('-d', '--datafile', type=str, required=True,
                        help='Path to the data file (e.g., data_table.csv)')
    
    args = parser.parse_args()
    
    # Load the data file
    try:
        df = pd.read_csv(args.datafile)
    except FileNotFoundError:
        sys.exit(f"Error: The data file '{args.datafile}' was not found.")
    except pd.errors.EmptyDataError:
        sys.exit(f"Error: The data file '{args.datafile}' is empty.")
    except Exception as e:
        sys.exit(f"Error reading '{args.datafile}': {e}")

    # Run the selected algorithm
    if args.algorithm == 'C':
        try:
            from apply_cole_kripke import apply_cole_kripke_single
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_single' from 'apply_cole_kripke.py'. "
                    "Ensure the file exists and is in the Python path.")
        result = apply_cole_kripke_single(df)
    else:  # args.algorithm == 'CM'
        try:
            from apply_cole_kripke import apply_cole_kripke_mult
        except ImportError:
            sys.exit("Error: Could not import 'apply_cole_kripke_mult' from 'apply_cole_kripke.py'. "
                    "Ensure the file exists and is in the Python path.")
        result = apply_cole_kripke_mult(df, args.limbs)

    print("Algorithm Output:")
    print(result)

if __name__ == "__main__":
    main()
