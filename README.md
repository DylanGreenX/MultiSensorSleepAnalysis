# MultiSensorSleepAnalysis

This repository provides an end-to-end pipeline for converting raw accelerometer data (from one or more sensors - specifically the Axivity AX6) into actigraphy counts, applying sleep/wake classification algorithms (C currently Cole-Kripke and Multi-Sensor variations of it), and visualizing the resulting time series data.

> **Note**  
> Parts of this pipeline adapt code from the [dipetkov/actigraph.sleepr](https://github.com/dipetkov/actigraph.sleepr/blob/master/R/apply_cole_kripke.R) repository, translating the original R implementation into Python.  
>  
> The procedure for generating actigraph counts from raw accelerometer data is based on [this paper](https://journals.lww.com/acsm-msse/fulltext/2017/11000/generating_actigraph_counts_from_raw_acceleration.25.aspx).

## Table of Contents
1. [Overview of the Workflow](#overview-of-the-workflow)  
2. [Requirements](#requirements)  
3. [Usage](#usage)  
   - [Data Generation & Preprocessing](#1-data-generation--preprocessing)  
   - [Applying Sleep/Wake Algorithms](#2-applying-sleepwake-algorithms)  
   - [Data Visualization](#3-data-visualization)
5. [Legacy Code](#legacy-code)  
6. [References](#references)  

---

## Overview of the Workflow

1. **Raw Data Collection**  
    You begin with one or more raw CSV files containing accelerometer data from different sensors (or limbs). Each file is typically *headerless*, with columns for timestamp and the x/y/z axes.

    When initially downloaded, data is formatted as follows:

    ```
    46:30.3, -0.059326, -0.519531, -0.745361
    46:30.3, 0.631104, -0.555908, -0.665283
    46:30.3, 0.431885, -0.566406, -0.79248
    46:30.3, 0.431152, -0.51709, -0.580078
    ```

    The following preprocessing step will handle labeling this accordingly for the output files.

2. **Preprocessing**  
   The `preprocess.py` script converts each raw CSV into a set of 60-second epoch “actigraphy counts.” It does so by reading, resampling, bandpassing, and aggregating each accelerometer file. This preprocessing script can handle up to four sensors at one time.

3. **Combining (Optional)**  
   If multiple sensors are provided (e.g., from multiple limbs), `preprocess.py` can merge the resulting epoch data into a single CSV (labeled with suffixes like `_1, _2, etc.`). An individual formatted file will be produced for each sensor, as well as a combined actigraphy count file. They are then stored in a file with the following naming convention `MMDD_HHMMSS`

4. **Applying Sleep/Wake Algorithms**  
   Next, you can run the Cole-Kripke algorithm (in either single-sensor or multi-limb mode) using `CLI.py`. This code is structured so it can be extensible. Other algorithms were initially implemented as well (seen in Legacy Code), however, are not currently being worked on. The CLI calls the functions from `apply_cole_kripke.py`:
   - **Single-limb** classification (flag `-a C`)  
   - **Multi-limb** classification (flag `-a CM`)  

5. **Visualization**  
   Finally, you can visualize the resulting data with:
   - `single_dat_viz.py` (for the single-limb CSV)  
   - `mult_data_viz.py` (for multi-limb CSV)  

---

## Requirements

- Python 3.7+  
- **Packages**: `pandas`, `numpy`, `matplotlib`, `scipy`, `argparse`, `tqdm`  
  - Install via `pip install pandas numpy matplotlib scipy tqdm`

---

## Usage

### 1. Data Generation & Preprocessing

**Script:** [`preprocess.py`](#preprocesspy)

This script converts one or more raw CSV files (headerless) into 60-second epoch actigraphy counts. By default, it is tuned for Axivity data at ~100Hz, but can be adjusted with the `-r` parameter.

Example usage:
```bash
python preprocess.py file1.csv file2.csv -r 100 -o out_folder
```
**Positional arguments:**
- `file1.csv file2.csv ...` up to 4 files

**Options:**
- `-r, --raw_rate`: Sampling rate of the raw files (e.g., 100 Hz)
- `-o, --output_dir`: Output folder name (created inside `../test_data/` by default)

**Output:**
- A CSV file for each input (sensor_1_counts.csv, sensor_2_counts.csv, …)
- Optionally a combined_counts.csv if more than one input file is provided.

### 2. Applying Sleep/Wake Algorithms

**Script:** [`CLI.py`](#clipy)

Once you have a CSV with epoch-level counts (from either a single sensor or multiple sensors merged), you can run Cole-Kripke classification to get sleep vs. wake labels.

Example usage:
```bash
python CLI.py -a C -d path_to_preprocessed_counts.csv
```

**Options:**
- `-a, --algorithm`:
    - `C` = Cole-Kripke for a single sensor
    - `CM` = Cole-Kripke for multiple limbs 
    
    *Multi Limb algorithms are intended for four sensors. Currently running CM with a combined file with less than four sensors will NOT fail gracefully. This will be revised at a later date. Allthough we can create combined files with less sensors, this was more of a future proofing methodology incase we deem it necessary to run algorithms on subsection (say two groups of two sensors)*

- `-d, --datafile`: Path to the actigraphy counts CSV

**Output:**
- For single-sensor mode (C), a file named `cole_single_results.csv`
- For multi-limb mode (CM), a file named `cole_mult_results.csv`
- Each line includes the timestamps and a “sleep index” per sensor/limb, plus a sleep column labeling each minute as S (sleep) or W (wake).

### 3. Data Visualization 

*Note this section is still heavily a work in progress. This really for my own ability to understand the data I am looking at. Visualization will be focused on heavily once I am happy with the core underlying algorithms.*

**Scripts:**
- `single_dat_viz.py`
- `mult_data_viz.py`

#### Single-Sensor Plot

For results generated by the single-sensor Cole-Kripke approach (`cole_single_results.csv`), you can visualize the sleep index over time by running:
```bash
python single_dat_viz.py cole_single_results.csv
```
A matplotlib window will pop up showing:
- Scatter points colored by sleep state (S in blue, W in red)
- A trend line for the sleep index

#### Multi-Limb Plot

For results generated by the multi-limb Cole-Kripke approach (`cole_mult_results.csv`), use:
```bash
python mult_data_viz.py cole_mult_results.csv
```
You’ll see different trends for each limb plotted over time, and the same color coding for sleep/wake classification. This allows comparing the sleep indices across multiple limbs.

## Legacy Code

There is a folder named `Legacy Code` containing older scripts (e.g., `legacy_CLI.py`, `apply_choi.py`, etc.). These files are preserved for reference on earlier development approaches but are not the recommended for use currently. If we determine later on that there is value in persuing these algorithms, they will be available. These files me also server to help create figures / explain our development cycle to get to our end goal.

## References

- Dipetkov’s actigraph.sleepr Cole-Kripke code: https://github.com/dipetkov/actigraph.sleepr
- “Generating ActiGraph Counts from Raw Acceleration Recorded by an Alternative Monitor”: https://pubmed.ncbi.nlm.nih.gov/28604558/
