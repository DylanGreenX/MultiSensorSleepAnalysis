import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse, sys

def plot_weighted_consensus(file_path):
    """Plot weighted consensus visualization."""

    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        sys.exit(f"Error loading '{file_path}': {e}")

    # Verify columns
    if 'consensus_index' not in df or 'consensus_sleep' not in df:
        sys.exit("CSV must contain 'consensus_index' and 'consensus_sleep' columns")

    # Parse timestamps
    df['dataTimestamp'] = pd.to_datetime(df['dataTimestamp'], errors='coerce')

    # Map Sleep/Wake to binary
    df['consensus_num'] = df['consensus_sleep'].map({'S': 0, 'W': 1})
    if df['consensus_num'].isnull().any():
        sys.exit("Error: 'consensus_sleep' must only contain 'S' or 'W'")

    # Compute epoch length in minutes
    if len(df) > 1:
        epoch_min = (df['dataTimestamp'].iloc[1] - df['dataTimestamp'].iloc[0]).total_seconds() / 60.0
    else:
        epoch_min = 1.0

    # Compute total awake/sleep minutes
    total_awake = df['consensus_num'].sum() * epoch_min
    total_sleep = (len(df) - df['consensus_num'].sum()) * epoch_min

    # Determine 4-hour window
    start = df['dataTimestamp'].min()
    end   = start + pd.Timedelta(hours=4)

    # Begin plot
    fig, ax = plt.subplots(figsize=(14, 4))

    # Continuous consensus index
    ax.plot(
        df['dataTimestamp'],
        df['consensus_index'],
        label='Consensus Index',
        linewidth=1.5,
        color='blue',
        alpha=0.7
    )
    # Threshold line
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Threshold = 1.0')

    # Overlay wake (red) and sleep (blue) fill
    top = df['consensus_index'].max() * 1.02
    ax.fill_between(
        df['dataTimestamp'],
        0,
        df['consensus_num'] * top,
        step='post',
        color='red',
        alpha=0.3,
        label='Wake'
    )
    ax.fill_between(
        df['dataTimestamp'],
        0,
        (1 - df['consensus_num']) * top,
        step='post',
        color='blue',
        alpha=0.1,
        label='Sleep'
    )

    thresh = 1.0

    # Annotate each contiguous wake segment with its duration
    awake = df['consensus_num'].values.astype(bool)
    segments = []
    in_seg = False

    for i, val in enumerate(awake):
        if val and not in_seg:
            seg_start = i
            in_seg = True
        elif not val and in_seg:
            seg_end = i - 1
            segments.append((seg_start, seg_end))
            in_seg = False

    # flush final segment if it runs to the end
    if in_seg:
        segments.append((seg_start, len(awake) - 1))

    max_val = df['consensus_index'].max()
    span = max_val - thresh

    for idx, (s, e) in enumerate(segments):
        t0 = df.at[s, 'dataTimestamp']
        t1 = df.at[e, 'dataTimestamp']
        dur_min = (e - s + 1) * epoch_min
        mid = t0 + (t1 - t0) / 2

        # stagger: even segments slightly above threshold, odd a bit higher
        base_offset = span * 0.05
        stagger_offset = span * 0.12 * (idx % 2)  # alternate up/down
        y = thresh + base_offset + stagger_offset

        ax.text(
            mid, y,
            f"{int(dur_min)} m.",
            color='red', ha='center', va='bottom'
        )

    # Legend
    leg = ax.legend(loc='upper right')
    leg.get_frame().set_linewidth(2)

    # Metrics box moved down
    ax.text(
        0.98, 0.60,
        f"Total Awake: {int(total_awake)} min\nTotal Sleep: {int(total_sleep)} min",
        transform=ax.transAxes,
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )

    # Format axes
    ax.set_xlim(start, end)
    ax.set_xlabel('Time (HH:MM)')
    ax.set_ylabel('Consensus Index')
    ax.set_title('Weighted Consensus Index Sleep/Wake Classification')
    ax.grid(which='major', linestyle='--', alpha=0.5)
    ax.grid(which='minor', linestyle=':', alpha=0.3)

    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_majority_consensus(file_path):
    """Plot majority-vote consensus visualization."""
    # Load
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        sys.exit(f"Error loading '{file_path}': {e}")

    # Required
    if 'consensus_majority' not in df.columns:
        sys.exit("Error: 'consensus_majority' column not found.")

    # Timestamps
    df['dataTimestamp'] = pd.to_datetime(df['dataTimestamp'], errors='coerce')

    # Binary sleep/wake
    df['consensus_num'] = df['consensus_majority'].map({'S': 0, 'W': 1})
    if df['consensus_num'].isnull().any():
        sys.exit("Error: 'consensus_majority' must only contain 'S' or 'W'.")

    # Epoch length (min)
    if len(df) > 1:
        epoch_min = (df['dataTimestamp'].iloc[1] - df['dataTimestamp'].iloc[0]).total_seconds() / 60.0
    else:
        epoch_min = 1.0

    # Totals
    total_awake = df['consensus_num'].sum() * epoch_min
    total_sleep = (len(df) - df['consensus_num'].sum()) * epoch_min

    # 4h window
    start = df['dataTimestamp'].min()
    end = start + pd.Timedelta(hours=4)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot each limb's sleep_index trend
    for i in range(1, 5):
        col = f'Limb {i} sleep_index'
        if col in df.columns:
            ax.plot(
                df['dataTimestamp'], df[col],
                label=f'Limb {i} Trend', linewidth=1, alpha=0.6
            )

    # Threshold line
    thresh = 1.0
    ax.axhline(thresh, color='gray', linestyle='--', linewidth=1, label='Threshold = 1.0')

    # Vertical scale
    sleep_indices = [df[f'Limb {i} sleep_index'] for i in range(1,5) if f'Limb {i} sleep_index' in df]
    if sleep_indices:
        top = pd.concat(sleep_indices).max() * 1.02
    else:
        top = 1.02

    # Fill wake/sleep
    ax.fill_between(
        df['dataTimestamp'], 0, df['consensus_num'] * top,
        step='post', color='red', alpha=0.3, label='Wake'
    )
    ax.fill_between(
        df['dataTimestamp'], 0, (1 - df['consensus_num']) * top,
        step='post', color='blue', alpha=0.1, label='Sleep'
    )

    # Annotate wake segments
    awake = df['consensus_num'].values.astype(bool)
    segments, in_seg = [], False
    for idx, val in enumerate(awake):
        if val and not in_seg:
            seg_start, in_seg = idx, True
        elif not val and in_seg:
            segments.append((seg_start, idx-1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(awake)-1))

    span = top - thresh
    for j, (s, e) in enumerate(segments):
        t0 = df.at[s, 'dataTimestamp']
        t1 = df.at[e, 'dataTimestamp']
        dur = (e - s + 1) * epoch_min
        mid = t0 + (t1 - t0) / 2
        offset = span * (0.05 + 0.12 * (j % 2))
        y = thresh + offset
        ax.text(
            mid, y, f"{int(dur)} min",
            color='red', ha='center', va='bottom'
        )

    # Legend with bold border
    leg = ax.legend(loc='upper right')
    leg.get_frame().set_linewidth(2)

    # Metrics box under legend
    ax.text(
        0.98, 0.40,
        f"Total Awake: {int(total_awake)} min\nTotal Sleep: {int(total_sleep)} min",
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
    )

    # Format axes
    ax.set_xlim(start, end)
    ax.set_xlabel('Time (HH:MM)')
    ax.set_ylabel('Sleep Index')
    ax.set_title('Majority-Vote Consensus Sleep/Wake Classification')
    ax.grid(which='major', linestyle='--', alpha=0.5)
    ax.grid(which='minor', linestyle=':', alpha=0.3)

    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    """Main CLI interface for consensus visualization."""
    parser = argparse.ArgumentParser(description="Visualize consensus sleep classification results")
    parser.add_argument("file", help="CSV file with consensus results")
    parser.add_argument("-t", "--type", choices=["weighted", "majority"], default="weighted",
                       help="Type of consensus visualization (default: weighted)")
    args = parser.parse_args()

    if args.type == "weighted":
        plot_weighted_consensus(args.file)
    elif args.type == "majority":
        plot_majority_consensus(args.file)


if __name__ == '__main__':
    main()