# MultiSensorSleepAnalysis

This repository provides an end-to-end pipeline for converting raw accelerometer data (from one or more sensors - specifically the Axivity AX6) into actigraphy counts, applying sleep/wake classification algorithms, and validating results against polysomnography (PSG) ground truth data.

## Features
- **Data Preprocessing**: Convert raw accelerometer data to 60-second epoch actigraphy counts
- **Sleep Classification**: Multiple algorithms including Cole-Kripke variants and traditional methods
- **PSG Validation**: Compare actigraphy results against polysomnography ground truth
- **Comprehensive Visualization**: Timeline plots, validation metrics, and comparison charts
- **Modular Architecture**: Clean, extensible codebase with organized modules

> **Note**  
> Parts of this pipeline adapt code from the [dipetkov/actigraph.sleepr](https://github.com/dipetkov/actigraph.sleepr/blob/master/R/apply_cole_kripke.R) repository, translating the original R implementation into Python.  
>  
> The procedure for generating actigraph counts from raw accelerometer data is based on [this paper](https://journals.lww.com/acsm-msse/fulltext/2017/11000/generating_actigraph_counts_from_raw_acceleration.25.aspx).

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
   - [Data Preprocessing](#data-preprocessing)
   - [Sleep Classification](#sleep-classification)
   - [PSG Validation](#psg-validation)
   - [Visualization](#visualization)
5. [Output Files](#output-files)
6. [Legacy Code](#legacy-code)
7. [References](#references)  

---

## Installation

### Requirements
- Python 3.7+
- Required packages listed in `requirements.txt`

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd MultiSensorSleepAnalysis

# Install dependencies
pip install -r requirements.txt

# Install the package (optional, for development)
pip install -e .
```

## Project Structure

```
MultiSensorSleepAnalysis/
├── multisensor_sleep/           # Main package
│   ├── algorithms/              # Sleep classification algorithms
│   ├── preprocessing/           # Data preprocessing
│   ├── validation/              # PSG validation module
│   ├── visualization/           # Plotting and visualization
│   └── utils/                   # Shared utilities
├── scripts/                     # Command-line interfaces
├── data/                        # Input data (raw, processed, samples)
├── results/                     # All output files
│   ├── algorithm_outputs/       # Classification results
│   ├── validation/              # PSG validation results
│   ├── visualizations/          # Generated plots
│   └── reports/                 # Summary reports
├── tests/                       # Unit tests
├── legacy/                      # Archived code
└── docs/                        # Documentation
```

## Quick Start

```bash
# 1. Preprocess raw accelerometer data
python scripts/preprocess_data.py data/raw/sensor1.csv data/raw/sensor2.csv -o data/processed/

# 2. Run sleep classification
python scripts/cli.py -a C -l 1 -d data/processed/sensor_1_counts.csv

# 3. Visualize results
python scripts/visualize_results.py results/algorithm_outputs/cole_single_results.csv

# 4. Validate against PSG (if available)
python scripts/validate_psg.py -a results/algorithm_outputs/cole_single_results.csv -p data/psg_data.csv --lights_out "2025-01-01 22:00:00" --lights_on "2025-01-02 07:00:00"
```

---

## Detailed Usage

### Data Preprocessing

Convert raw accelerometer CSV files into 60-second epoch actigraphy counts.

```bash
python scripts/preprocess_data.py [files...] [options]
```

**Arguments:**
- `files`: Raw CSV files (up to 4, headerless format)
- `-r, --raw_rate`: Sampling rate in Hz (default: 100)
- `-o, --output_dir`: Output directory (default: data/processed/)

**Input Format:**
```
46:30.3, -0.059326, -0.519531, -0.745361
46:30.3, 0.631104, -0.555908, -0.665283
```

**Output:**
- `sensor_N_counts.csv` for each input file
- `combined_counts.csv` if multiple files provided
- Files saved to `data/processed/MMDD_HHMMSS/`

### Sleep Classification

Apply sleep/wake classification algorithms to preprocessed actigraphy data.

```bash
python scripts/cli.py -a [algorithm] -l [limbs] -d [datafile]
```

**Available Algorithms:**
- `C`: Cole-Kripke single sensor
- `CM`: Cole-Kripke multi-limb
- `CMM`: Cole-Kripke multi-limb with majority voting
- `CW`: Cole-Kripke multi-limb with weighted consensus
- `S`: Sadeh algorithm
- `TRO`: Troiano algorithm
- `CHO`: Choi algorithm

**Arguments:**
- `-a, --algorithm`: Algorithm type (required)
- `-l, --limbs`: Number of limbs/sensors (1-4, required)
- `-d, --datafile`: Path to preprocessed counts CSV (required)

**Output:**
- Results saved to `results/algorithm_outputs/`
- Filename format: `[algorithm]_results.csv`
- Contains timestamps, sleep indices, and S/W classifications

### PSG Validation

Validate actigraphy results against polysomnography (PSG) ground truth data.

```bash
python scripts/validate_psg.py -a [actigraphy_file] -p [psg_file] --lights_out [time] --lights_on [time] [options]
```

**Arguments:**
- `-a, --actigraphy`: Actigraphy results CSV file
- `-p, --psg`: PSG data CSV file (30-second epochs)
- `--lights_out`: Lights out timestamp (ISO format)
- `--lights_on`: Lights on timestamp (ISO format)
- `-o, --output`: Output directory (default: results/validation/)

**PSG Data Format:**
```csv
timestamp,sleep_stage
2025-01-01 22:30:00,Wake
2025-01-01 22:30:30,N2
2025-01-01 22:31:00,N3
```

**Output:**
- Validation metrics (sensitivity, specificity, F1 score)
- Epoch-by-epoch comparison
- Confusion matrix
- Timeline comparison plots

### Visualization

Generate plots and visualizations for sleep analysis results.

```bash
python scripts/visualize_results.py [results_file] [options]
```

**Features:**
- Timeline plots with sleep/wake states
- Multi-sensor comparisons
- Algorithm performance metrics
- Validation result visualizations

**Output:**
- Plots saved to `results/visualizations/`
- Interactive and static plot options
- Customizable color schemes and layouts

## Output Files

### Algorithm Results
**Location:** `results/algorithm_outputs/`
- `cole_single_results.csv`: Single-sensor Cole-Kripke results
- `cole_mult_results.csv`: Multi-sensor Cole-Kripke results
- `[algorithm]_results.csv`: Results from other algorithms

### Validation Results
**Location:** `results/validation/`
- `validation_metrics.csv`: Performance metrics summary
- `epoch_comparison.csv`: Epoch-by-epoch comparison
- `confusion_matrix.png`: Visual confusion matrix
- `timeline_comparison.png`: Side-by-side timeline plots

### Visualizations
**Location:** `results/visualizations/`
- Timeline plots showing sleep/wake patterns
- Multi-sensor comparison charts
- Algorithm performance visualizations

## Legacy Code

The `legacy/` folder contains older implementations preserved for reference. These include earlier algorithm versions and development approaches. Current users should use the main pipeline described above.

## References

- Dipetkov’s actigraph.sleepr Cole-Kripke code: https://github.com/dipetkov/actigraph.sleepr
- “Generating ActiGraph Counts from Raw Acceleration Recorded by an Alternative Monitor”: https://pubmed.ncbi.nlm.nih.gov/28604558/
