##### This code was used when only using single sensors. The abstraction between this and and combine_csv.py did not make logical sense as more sensors was added.
##### The code was refactored to combine the two scripts into one.

import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Combine 2 to 4 CSV files on 'dataTimestamp' column."
    )
    parser.add_argument(
        "input_files", 
        nargs="+",
        help="Paths to input CSV files (2 to 4 files)."
    )
    parser.add_argument(
        "-o", "--output_file", 
        default="combined.csv",
        help="Output CSV filename (default: combined.csv)."
    )
    args = parser.parse_args()

    # Check we have between 2 and 4 files
    if len(args.input_files) < 2 or len(args.input_files) > 4:
        print("Error: You must provide between 2 and 4 CSV files.")
        sys.exit(1)

    # Read each file into a pandas DataFrame
    dataframes = []
    for i, file_path in enumerate(args.input_files, start=1):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            sys.exit(1)

        # Check that the required columns exist
        required_cols = {"dataTimestamp", "axis1", "axis2", "axis3"}
        if not required_cols.issubset(df.columns):
            print(f"Error: {file_path} must have columns {required_cols}.")
            sys.exit(1)

        # Rename columns: axis1 -> axis1_i, axis2 -> axis2_i, axis3 -> axis3_i
        df = df.rename(
            columns={
                "axis1": f"axis1_{i}",
                "axis2": f"axis2_{i}",
                "axis3": f"axis3_{i}"
            }
        )
        dataframes.append(df)

    # Merge all DataFrames on 'dataTimestamp'
    # We'll do an inner join so that only timestamps present in all files are kept.
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on="dataTimestamp", how="inner")

    # Order columns: dataTimestamp first, then axis1_x, axis2_x, axis3_x
    # We'll detect how many files we have (N), and reorder the columns accordingly.
    num_files = len(args.input_files)
    # Build the final ordered column list
    # E.g. for 2 files: [dataTimestamp, axis1_1, axis1_2, axis2_1, axis2_2, axis3_1, axis3_2]
    # E.g. for 4 files: [dataTimestamp, axis1_1, axis1_2, axis1_3, axis1_4, ...]
    col_order = ["dataTimestamp"]
    for axis in ["axis1", "axis2", "axis3"]:
        for i in range(1, num_files + 1):
            col_order.append(f"{axis}_{i}")

    # Filter only the columns that exist (pandas merge might bring some out of order columns)
    col_order = [c for c in col_order if c in merged_df.columns]

    # Reorder the merged DataFrame
    merged_df = merged_df[col_order]

    # Write the combined CSV
    merged_df.to_csv(args.output_file, index=False)
    print(f"Combined CSV written to {args.output_file}")

if __name__ == "__main__":
    main()
