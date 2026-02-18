# MultiSensorSleepAnalysis

End-to-end pipeline for converting raw accelerometer data from multiple Axivity AX6 sensors into actigraphy counts, applying sleep/wake classification algorithms, and validating results against polysomnography (PSG).

> **Attribution**
> Cole-Kripke implementation adapted from [dipetkov/actigraph.sleepr](https://github.com/dipetkov/actigraph.sleepr/blob/master/R/apply_cole_kripke.R).
> Actigraphy count generation follows [Brond et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28604558/).

## Installation

```bash
git clone <repository-url>
cd MultiSensorSleepAnalysis
pip install -r requirements.txt
pip install -e .  # optional, for development
```

Requires Python 3.7+ and the packages in `requirements.txt` (pandas, numpy, scipy, matplotlib, tqdm).

## Project Structure

```
MultiSensorSleepAnalysis/
├── multisensor_sleep/
│   ├── algorithms/
│   │   ├── cole_kripke.py          # Cole-Kripke (1992): single, vote, weighted, majority
│   │   └── sadeh.py                # Sadeh (1994): single, vote, weighted, majority
│   ├── preprocessing/
│   │   └── preprocess.py           # Raw accelerometer -> 60s epoch actigraphy counts
│   ├── validation/
│   │   ├── psg_parser.py           # PSG hypnogram loader
│   │   └── validation_utils.py     # Epoch-by-epoch comparison and metrics
│   └── visualization/
│       ├── single_sensor.py        # Single-sensor sleep index timeline
│       ├── multi_sensor.py         # Multi-sensor per-limb timelines
│       ├── consensus.py            # Weighted consensus visualization
│       └── comparison.py           # Sadeh index + sleep/wake comparison
├── scripts/
│   ├── cli.py                      # Main CLI for running algorithms
│   ├── preprocess_data.py          # Preprocessing CLI
│   ├── validate_psg.py             # PSG validation CLI
│   └── visualize_results.py        # Visualization CLI
├── tests/
│   ├── test_cole_kripke.py        # Cole-Kripke algorithm tests (33 tests)
│   └── test_sadeh.py              # Sadeh algorithm tests (38 tests)
├── data/                           # Input data (gitignored; raw sensor CSVs, PSG files)
└── results/                        # Algorithm outputs (gitignored; regenerated per run)
```

## Quick Start

```bash
# 1. Preprocess raw accelerometer data (up to 4 sensors)
python scripts/preprocess_data.py data/raw/sensor1.csv data/raw/sensor2.csv

# 2. Run Cole-Kripke weighted consensus on 4 limbs
python scripts/cli.py -a CW -l 4 -d data/processed/combined_counts.csv -b "2025-11-08 21:00:00"

# 3. Validate against PSG
python scripts/validate_psg.py \
  -a results/algorithm_outputs/cole_weighted_mult_results.csv \
  -p data/psg_data.csv \
  --lights_out "2025-11-08 22:00:00" \
  --lights_on "2025-11-09 07:00:00"
```

## Algorithms

### Cole-Kripke (1992)

Weighted linear combination over a 7-epoch window. Sleep if index < 1.0.

| CLI Flag | Variant | Description |
|----------|---------|-------------|
| `C` | Single-sensor | Standard single-wrist Cole-Kripke |
| `CV` | Weighted vote | Per-limb binary S/W labels combined via anatomically-weighted vote |
| `CW` | Weighted index | Per-limb continuous indices combined via weighted average (threshold 0.7) |
| `CM` | Majority | Per-limb with unweighted majority vote (>=3/4 limbs = Sleep) |

### Sadeh (1994)

Computes four statistical features (MEAN, NAT, SD, LG) over an 11-epoch window. Sleep if PS >= 0.

| CLI Flag | Variant | Description |
|----------|---------|-------------|
| `S` | Single-sensor | Averaged multi-limb counts fed to single Sadeh |
| `SV` | Weighted vote | Per-limb binary S/W labels combined via anatomically-weighted vote |
| `SW` | Weighted index | Per-limb continuous indices combined via weighted average (threshold 0) |
| `SM` | Majority | Per-limb with unweighted majority vote (>=3/4 limbs = Sleep) |

### Multi-sensor fusion strategies

All multi-sensor variants use anatomical weighting: wrists (limbs 2 & 4) = 1.0, ankles (limbs 1 & 3) = 0.5.

| Suffix | Strategy | Weighting | Key difference |
|--------|----------|-----------|----------------|
| `V` | Weighted vote | On binary labels | Information lost at binarization — index magnitude discarded |
| `W` | Weighted index | On continuous indices | Preserves magnitude — a barely-sleeping limb pulls the average |
| `M` | Majority | Equal (unweighted) | Simple count — Sleep if >=3/4 limbs agree |

## Usage

### Preprocessing

Converts raw 100 Hz triaxial accelerometer CSVs into 60-second epoch actigraphy counts via the Brond et al. pipeline (resample, filter, dead-band, quantize, epoch summation).

```bash
python scripts/preprocess_data.py sensor1.csv sensor2.csv sensor3.csv sensor4.csv
```

Produces per-sensor count files and a `combined_counts.csv` with all limbs merged.

### Sleep Classification

```bash
python scripts/cli.py -a <algorithm> -l <limbs> -d <datafile> [-b <baseline>]
```

| Argument | Description |
|----------|-------------|
| `-a` | Algorithm: `C`, `CV`, `CW`, `CM`, `S`, `SV`, `SW`, `SM` |
| `-l` | Number of limbs (1-4) |
| `-d` | Path to preprocessed counts CSV |
| `-b` | Recording start timestamp (required for Cole-Kripke). Format: `"YYYY-MM-DD HH:MM:SS"` |

The `-b` (baseline) flag tells Cole-Kripke how to convert elapsed seconds in the data back to real timestamps in the output. It should match the recording start time used during preprocessing. Sadeh algorithms do not require it.

**Examples:**

```bash
# Single-sensor Cole-Kripke
python scripts/cli.py -a C -l 1 -d data/processed/combined_counts.csv -b "2025-11-08 21:00:00"

# Multi-sensor Sadeh weighted vote
python scripts/cli.py -a SV -l 4 -d data/processed/combined_counts.csv

# Multi-sensor Cole-Kripke majority vote
python scripts/cli.py -a CM -l 4 -d data/processed/combined_counts.csv -b "2025-11-08 21:00:00"
```

Results are saved to `results/algorithm_outputs/`.

### PSG Validation

```bash
python scripts/validate_psg.py \
  -a <actigraphy_results.csv> \
  -p <psg_hypnogram.csv> \
  --lights_out "YYYY-MM-DD HH:MM:SS" \
  --lights_on "YYYY-MM-DD HH:MM:SS"
```

PSG data should contain 30-second epochs with `timestamp` and `sleep_stage` columns (W, N1, N2, N3, REM). Stages are binarized: N1/N2/N3/REM = Sleep, W = Wake.

### Testing

```bash
pytest tests/                # Run all tests
pytest tests/ -v             # Verbose output
pytest tests/test_sadeh.py   # Run Sadeh tests only
```

Tests validate algorithm correctness (coefficients, formulas, thresholds), fusion strategies (vote, weighted, majority), anatomical weighting, and guard against known regressions (e.g., the Sadeh threshold bug).

### Visualization

```bash
python scripts/visualize_results.py <results_file>
```

## Output Files

| Directory | Contents |
|-----------|----------|
| `results/algorithm_outputs/` | Per-algorithm CSVs with timestamps, sleep indices, and S/W labels |
| `results/validation/` | Metrics, epoch-by-epoch comparisons, confusion matrices |
| `results/visualizations/` | Timeline plots, multi-sensor comparisons |

## References

- Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., & Gillin, J. C. (1992). Automatic sleep/wake identification from wrist activity. *Sleep*, 15(5), 461-469.
- Sadeh, A., Sharkey, K. M., & Carskadon, M. A. (1994). Activity-based sleep-wake identification: An empirical test of methodological issues. *Sleep*, 17(3), 201-207.
- Brond, J. C., Andersen, L. B., & Arvidsson, D. (2017). Generating ActiGraph counts from raw acceleration recorded by an alternative monitor. *Medicine & Science in Sports & Exercise*, 49(11), 2351-2360.
- Dipetkov's actigraph.sleepr: https://github.com/dipetkov/actigraph.sleepr
