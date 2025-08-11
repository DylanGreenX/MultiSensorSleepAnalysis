import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse, sys


def main():
    parser = argparse.ArgumentParser(
        description="Plot majority-vote consensus & limb trends over first 4 hours with duration metrics."
    )
    parser.add_argument(
        "file", help="CSV with dataTimestamp, consensus_majority, and 'Limb i sleep_index' columns"
    )
    args = parser.parse_args()

    # Load
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        sys.exit(f"Error loading '{args.file}': {e}")

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
    # if any limb trend exists, top is the max of all sleep_index columns
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

if __name__ == '__main__':
    main()