import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot sleep index over time from a CSV file.")
    parser.add_argument("file", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.file)

        # Convert timestamp to datetime
        data["dataTimestamp"] = pd.to_datetime(data["dataTimestamp"], errors="coerce")

        # Plot the sleep index over time
        plt.figure(figsize=(12, 6))

        # Define color mapping for sleep states
        sleep_colors = {"W": "red", "S": "blue"} # This is a dictionary for now. Thinking ahead for multi sensor.

        # Plot points for each sleep state
        for state, color in sleep_colors.items():
            state_data = data[data["sleep"] == state]
            plt.scatter(state_data["dataTimestamp"], state_data["sleep_index"], 
                        color=color, label=state, marker='o', s=5, alpha=0.7)

        # Plot the overall line for sleep index
        plt.plot(data["dataTimestamp"], data["sleep_index"], linestyle='-', color='gray', alpha=0.5)

        # Formatting
        plt.xlabel("Time (HH:MM)")
        plt.ylabel("Sleep Index")
        plt.title("Sleep/Wake States Over Time")
        plt.grid(True)

        # Fix X-axis visibility issues
        plt.xticks(rotation=45, fontsize=8)
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))  # Show every 15 minutes
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Format as HH:MM

        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
