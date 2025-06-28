# EDL Magnetotelluric (MT) Data Processing Suite

A Python-based workflow for reading, processing, and analyzing magnetotelluric (MT) time series data from ASCII files, with support for batch processing, frequency analysis, corrections, and integration with LEMI MT executable tools.

---

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Single Site Processing](#single-site-processing)
  - [Batch Processing](#batch-processing)
  - [Frequency Analysis](#frequency-analysis)
  - [LEMIMT Integration](#lemimt-integration)
- [Command Line Options](#command-line-options)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

---

## Features

- Read and process MT ASCII data from multiple sites and days
- Apply drift, rotation, and tilt corrections
- Smoothing and outlier removal
- Merge remote reference data
- Frequency analysis: Welch, Multi-taper, Spectrogram
- Coherence analysis (MT and remote reference)
- Batch processing across multiple sites
- Integration with LEMI MT executable (`lemimt.exe`)
- Flexible timezone handling
- Comprehensive logging and output management

---

## File Structure

The project expects the following directory and file organization:

```
2025_EDL/
├── EDL_Process.py
├── EDL_Batch.py
├── EDL_Reader.py
├── outputs/
│   └── [SiteName]/
│       ├── [SiteName]_output_processed.txt
│       ├── [SiteName]_10Hz_output_processed.txt
│       ├── [SiteName]_coherence.png
│       └── ...
├── HDD5449/
│   ├── 294/
│   │   ├── EDL_041020000000.gps
│   │   ├── EDL_041020000000.gst
│   │   ├── EDL_041020000000.pll
│   │   └── ...
│   ├── 295/
│   │   ├── EDL_041021000000.ambientTemperature
│   │   ├── EDL_041021000000.BX
│   │   └── ...
│   ├── config/
│   │   └── recorder.ini
│   └── ...
├── sensors/
│   ├── l120new.rsp
│   ├── e000.rsp
│   └── ...
├── Working_Backup/
│   ├── EDL_Process.py
│   ├── EDL_Batch.py
│   └── EDL_Reader.py
├── lemimt.exe
└── ...
```

- **HDDxxxx/**: Site folders, each with subfolders for each day (e.g., 294, 295), containing ASCII data files.
- **outputs/**: All processed data and plots are saved here, organized by site.
- **sensors/**: Instrument response files.
- **Working_Backup/**: Backup copies of main scripts.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/EDL_MT_Processing.git
   cd EDL_MT_Processing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* If using `lemimt.exe`, ensure it is present in the project root or specify its path.

---

## Usage

### Single Site Processing

```bash
python EDL_Process.py --input_dir HDD5449 --plot_data --save_plots --save_processed_data
```

### Batch Processing

```bash
python EDL_Batch.py --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_data --save_plots --save_processed_data
```

### Frequency Analysis

```bash
python EDL_Process.py --input_dir HDD5449 --perform_freq_analysis WMS --save_plots
```
- `W` = Welch, `M` = Multi-taper, `S` = Spectrogram (combine as needed)

### LEMIMT Integration

```bash
python EDL_Process.py --input_dir HDD5449 --save_processed_data --run_lemimt
```
- On non-Windows systems, the command to run on Windows will be logged.

---

## Command Line Options

| Option                        | Description                                                                                  |
|-------------------------------|----------------------------------------------------------------------------------------------|
| `--input_dir`                 | Directory containing ASCII data files (required)                                             |
| `--plot_data`                 | Display physical channel plots                                                               |
| `--save_plots`                | Save plots to files instead of displaying                                                   |
| `--save_processed_data`       | Save processed data to file                                                                 |
| `--perform_freq_analysis`     | Perform frequency analysis: W (Welch), M (Multi-taper), S (Spectrogram)                    |
| `--remote_reference`          | Site name to use as remote reference                                                        |
| `--apply_drift_correction`    | Apply drift correction                                                                      |
| `--apply_rotation`            | Apply rotation correction                                                                   |
| `--tilt_correction`           | Apply tilt correction                                                                       |
| `--run_lemimt`                | Run lemimt.exe on the processed output file (Windows only)                                  |
| `--lemimt_path`               | Full path to lemimt.exe (default: lemimt.exe in current directory)                          |
| ...                           | *(See script for full list of options)*                                                     |

---

## Outputs

- **Processed Data:**  
  `outputs/[SiteName]/[SiteName]_output_processed.txt`  
  (and decimated versions, e.g., `_10Hz_output_processed.txt`)

- **Plots:**  
  Power spectra, coherence, and time series plots in `outputs/[SiteName]/`

- **Logs:**  
  Processing logs in `process_ascii.log`, `EDL_Process.log`, and batch logs.

---

## Troubleshooting

- **lemimt.exe not found:**  
  Ensure you are running on Windows and the path to `lemimt.exe` is correct.

- **Processed file not found for lemimt:**  
  The script now searches for all matching processed files and uses the correct one, including decimated outputs.

- **Timezone issues:**  
  Use the `--timezone` argument to specify the correct timezone(s) for your data. 
