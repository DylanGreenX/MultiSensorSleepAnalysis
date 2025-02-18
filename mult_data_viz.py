#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot sleep index over time for up to 4 sensors from a CSV file.")
    parser.add_argument("file", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    try:
        # Load data
        data = pd.read_csv(args.file)

        # Convert timestamp to datetime
        data["dataTimestamp"] = pd.to_datetime(data["dataTimestamp"], errors="coerce")

        # Create figure
        plt.figure(figsize=(12, 6))

        # Define sleep state colors
        sleep_colors = {"W": "red", "S": "blue"}  # Wake = red, Sleep = blue

        # Define sensors
        sensors = ["Limb 1", "Limb 2", "Limb 3", "Limb 4"]
        sensor_labels = {1: "Limb 1", 2: "Limb 2", 3: "Limb 3", 4: "Limb 4"}

        for i, sensor in enumerate(sensors, start=1):
            sleep_col = f"{sensor} sleep"
            index_col = f"{sensor} sleep_index"

            if sleep_col in data.columns and index_col in data.columns:
                # Plot each sleep state with different colors
                for state, color in sleep_colors.items():
                    state_data = data[data[sleep_col] == state]
                    plt.scatter(state_data["dataTimestamp"], state_data[index_col],
                                color=color, label=f"{sensor_labels[i]} - {state}", marker='o', alpha=0.7)

                # Plot the overall trend line
                plt.plot(data["dataTimestamp"], data[index_col], linestyle='-', alpha=0.5, label=f"{sensor_labels[i]} Trend")

        # Formatting
        plt.xlabel("Time (HH:MM)")
        plt.ylabel("Sleep Index")
        plt.title("Multi-Sensor Sleep/Wake States Over Time")
        plt.grid(True)

        # Fix X-axis visibility
        plt.xticks(rotation=45, fontsize=8)
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))  # Show every 15 minutes
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Format as HH:MM

        # Show legend
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
