import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Sadeh sleep index & sleep/wake over first 4 hours with duration metrics."
    )
    parser.add_argument(
        "file", help="CSV with dataTimestamp [s], count, sleep_index, sleep"
    )
    parser.add_argument(
        "--baseline",
        default="2025-02-03 21:00:00",
        help="Baseline timestamp to convert seconds to datetime"
    )
    args = parser.parse_args()

    # Load CSV
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        sys.exit(f"Error loading '{args.file}': {e}")

    # Validate columns
    required = {"dataTimestamp", "sleep_index", "sleep"}
    if not required.issubset(df.columns):
        sys.exit(f"CSV must contain columns: {', '.join(required)}")

    # Convert seconds-to-datetime
    baseline = pd.to_datetime(args.baseline)
    df["time"] = baseline + pd.to_timedelta(df["dataTimestamp"], unit="s")

    # Map sleep state to numeric
    df["sleep_num"] = df["sleep"].map({"S": 0, "W": 1})
    if df["sleep_num"].isnull().any():
        sys.exit("Error: 'sleep' column must only contain 'S' or 'W'.")

    # Compute epoch length in minutes
    if len(df) > 1:
        epoch_min = (df["time"].iloc[1] - df["time"].iloc[0]).total_seconds() / 60.0
    else:
        epoch_min = 1.0

    # Compute total awake/sleep minutes
    total_awake = df["sleep_num"].sum() * epoch_min
    total_sleep = (len(df) - df["sleep_num"].sum()) * epoch_min

    # Define 4-hour window
    start = df["time"].min()
    end = start + pd.Timedelta(hours=4)

    fig, ax = plt.subplots(figsize=(14, 4))

    # 1) Continuous sleep index
    ax.plot(
        df["time"],
        df["sleep_index"],
        label="Sleep Index",
        lw=1.5,
        color="blue",
        alpha=0.7
    )

    # 2) Threshold line
    thresh = 1.0
    ax.axhline(thresh, color="gray", ls="--", lw=1, label="Threshold = 1.0")

    # 3) Determine vertical scale
    top = df["sleep_index"].max() * 1.02

    # 4) Fill wake/sleep areas
    ax.fill_between(
        df["time"], 0, df["sleep_num"] * top,
        step="post", color="red", alpha=0.3, label="Wake"
    )
    ax.fill_between(
        df["time"], 0, (1 - df["sleep_num"]) * top,
        step="post", color="blue", alpha=0.1, label="Sleep"
    )

    # 5) Annotate each wake segment
    awake = df["sleep_num"].astype(bool).values
    segments = []
    in_seg = False
    for i, val in enumerate(awake):
        if val and not in_seg:
            seg_start = i
            in_seg = True
        elif not val and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(awake) - 1))

    span = top - thresh
    for idx, (s, e) in enumerate(segments):
        t0 = df.at[s, "time"]
        t1 = df.at[e, "time"]
        dur_min = (e - s + 1) * epoch_min
        mid = t0 + (t1 - t0) / 2
        offset = span * (0.05 + 0.12 * (idx % 2))
        y = thresh + offset
        ax.text(
            mid, y,
            f"{int(dur_min)} min",
            color="red", ha="center", va="bottom"
        )

    # 6) Bold-border legend
    leg = ax.legend(loc="upper right")
    leg.get_frame().set_linewidth(2)

    # 7) Metrics box under legend
    ax.text(
        0.98, 0.60,
        f"Total Awake: {int(total_awake)} min\nTotal Sleep: {int(total_sleep)} min",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round")
    )

    # 8) Format axes
    ax.set_xlim(start, end)
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Sleep Index")
    ax.set_title("Sadeh Sleep/Wake Classification")
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
