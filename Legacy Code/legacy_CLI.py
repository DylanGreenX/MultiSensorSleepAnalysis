##### This CLI was created when using multiple algorithms. We likely will only be testing Cole-Krikpe. The CLI will be refactored to only run Cole-Kripke.

import argparse
import pandas as pd
import sys

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Run a specified algorithm on a given data file.')
    parser.add_argument('-a', '--algorithm', type=str, required=True,
                        help='Algorithm to run (e.g., O, S, C, SM, CM)')
    parser.add_argument('-d', '--datafile', type=str, required=True,
                        help='Path to the data file (e.g., data_table.csv)')
    
    args = parser.parse_args()
    
    # Load the data file
    try:
        df = pd.read_csv(args.datafile)
    except FileNotFoundError:
        print(f"Error: The data file '{args.datafile}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The data file '{args.datafile}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{args.datafile}': {e}")
        sys.exit(1)
    
    algorithm_name = args.algorithm
    # Singular Algorithms
    if algorithm_name == 'O':
        # Import and run the Choi algorithm
        try:
            from apply_choi import apply_choi
        except ImportError:
            print("Error: Could not import 'apply_choi' from 'choi.py'. Ensure the file exists and is in the Python path.")
            sys.exit(1)
        result = apply_choi(df)
    elif algorithm_name == 'S':
        try:
            from apply_sadeh import apply_sadeh_single
        except ImportError:
            print("Error: Could not import 'apply_sadeh_single' from 'apply_sadeh.py'. Ensure the file exists and is in the Python path.")
            sys.exit(1)
        result = apply_sadeh_single(df)
    elif algorithm_name == 'C':
        try:
            from apply_cole_kripke import apply_cole_kripke_single
        except ImportError:
            print("Error: Could not import 'apply_cole_kripke_single' from 'apply_cole_kirpke.py'. Ensure the file exists and is in the Python path.")
            sys.exit(1)
        result = apply_cole_kripke_single(df)
    
    #Multi Limb Algorithms
    elif algorithm_name == 'SM':
        try:
            from apply_sadeh import apply_sadeh_mult
        except ImportError:
            print("Error: Could not import 'apply_sadeh_mult' from 'apply_sadeh.py'. Ensure the file exists and is in the Python path.")
            sys.exit(1)
        result = apply_sadeh_mult(df)
    elif algorithm_name == 'CM':
        try:
            from apply_cole_kripke import apply_cole_kripke_mult
        except ImportError:
            print("Error: Could not import 'apply_cole_kripke_mult' from 'apply_cole_kirpke.py'. Ensure the file exists and is in the Python path.")
            sys.exit(1)
        result = apply_cole_kripke_mult(df)
    else:
        print(f"Error: Unknown algorithm '{args.algorithm}'. Please choose from 'O (Choi)', 'C (Cole - singular)', 'S (Sadeh - singular)', 'CM (Cole - mult)', 'SM (Sadeh - mult)'.")
        sys.exit(1)
    
    # Output the result
    print("Algorithm Output:")
    print(result)

if __name__ == "__main__":
    main()
