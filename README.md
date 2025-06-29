# EDL Magnetotelluric (MT) Data Processing Suite

A Python-based workflow for reading, processing, and analyzing magnetotelluric (MT) time series data from ASCII files, with support for batch processing, frequency analysis, corrections, and integration with LEMI MT executable tools.

---


## Features

- Read and process MT ASCII data from multiple sites and days
- Apply drift, rotation, and tilt corrections
- Smoothing and outlier removal
- Merge remote reference data
- Frequency analysis: Welch, Multi-taper, Spectrogram
- Coherence analysis (MT and remote reference) and heatmaps
- Batch processing across multiple sites
- Integration with LEMI MT executable (`lemimt.exe`)
- Flexible timezone handling
- Parallel processing of multiple sites with clear logging and summary tables.
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
├── [SiteName]/
│   ├── [DayFolder]/
│   │   ├── EDL_041020000000.gps
│   │   ├── EDL_041020000000.gst
│   │   ├── EDL_041020000000.pll
│   │   ├── EDL_041020000000.BX
│   │   ├── EDL_041020000000.BY
│   │   ├── EDL_041020000000.BZ
│   │   ├── EDL_041020000000.EX
│   │   ├── EDL_041020000000.EY
│   │   └── ...
│   ├── [DayFolder]/
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
├── lemimt.exe
└── ...
```

- **HDDxxxx/**: Site folders, each with subfolders for each day (e.g., 294, 295), containing ASCII data files.
- **outputs/**: All processed data and plots are saved here, organized by site.
- **sensors/**: Instrument response files.

---
## Installation

- Python 3.7+
- Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `configparser`, `glob`, `pytz` (for timezone support)
- **Windows users:** Place `lemimt.exe` in the `2025_EDL/` folder (the project root) for automated transfer function estimation.

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy configparser pytz
```

**Clone the repository:**
```bash
git clone https://github.com/yourusername/EDL_MT_Processing.git
cd EDL_MT_Processing
```

## Configuration File Format

The `Processing.txt` file can be used to specify sites and their parameters for batch processing. It supports both 3-column and 4-column formats:

### 4-Column Format (with remote reference)
```csv
Site, xarm, yarm, RemoteReference
SiteA-50m, 45, 50, SiteD-50m
SiteB-10m, 10, 9.8, SiteD-50m
SiteC-10m, 10, 9.8, SiteD-50m
SiteD-50m, 50, 49.5, SiteA-50m
SiteA-50m, 45, 50, SiteB-10m
SiteB-10m, 10, 9.8, 
SiteC-10m, 10, 9.8, 
SiteD-50m, 50, 49.5
```
**Column Definitions:**
- **Site**: Site directory name
- **xarm**: X dipole length in meters
- **yarm**: Y dipole length in meters
- **RemoteReference**: (Optional) Remote reference site name

## Batch Processing with --config_file

Both `EDL_Batch.py` and `Test_Batch.py` now support a `--config_file` argument:

```bash
python EDL_Batch.py --config_file Processing.txt --parent_dir . --plot_data --save_plots
```

- If only one site is listed in the config file, it will process that site (with a message suggesting single-site mode).
- If multiple sites are listed, all will be processed in batch mode.
- The `--sites` argument is still supported for manual site listing.

See the updated script help (`-h`) for more details.

## Usage

### Single-Site Processing

```bash
python EDL_Process.py --input_dir HDD5449 --plot_data --save_plots --save_processed_data
```

#### Common Options
- `--input_dir [DIR]`         Directory with ASCII data files
- `--plot_data`               Show/save physical channel plots
- `--save_plots`              Save plots to outputs/[SiteName]/
- `--save_processed_data`     Save processed output text file
- `--apply_drift_correction`  Apply drift correction
- `--apply_rotation`          Apply rotation correction
- `--tilt_correction`         Apply tilt correction (add 'RR' for remote reference tilt)
- `--apply_smoothing`         Apply smoothing (median/adaptive)
- `--perform_freq_analysis`   Frequency analysis (W=Welch, M=Multi-taper, S=Spectrogram, e.g. 'WMS')
- `--apply_filtering`         Apply frequency filtering (see below)
- `--remote_reference [SITE]` Use another site as remote reference
- `--decimate [N ...]`        Decimate by factors N (e.g. 2 5 10)
- `--run_lemimt`              Run lemimt.exe on processed output (Windows only)

#### Example: Full Pipeline
```bash
python EDL_Process.py --input_dir HDD5449 --plot_data --save_plots --save_processed_data --apply_drift_correction --apply_rotation --tilt_correction --apply_smoothing --smoothing_method adaptive --perform_freq_analysis WMS --plot_coherence --remote_reference HDD5974 --decimate 2 5 --run_lemimt
```

### Batch Processing

#### Method 1: Manual Site Listing
Process multiple sites in parallel by listing them manually:
```bash
python EDL_Batch.py --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_data --save_plots --save_processed_data --apply_drift_correction --apply_rotation --tilt_correction --max_workers 4
```

#### Method 2: Configuration File (Recommended)
Use `Processing.txt` to specify sites and their parameters:
```bash
python EDL_Batch.py --config_file Processing.txt --plot_data --save_plots --save_processed_data --apply_drift_correction --apply_rotation --tilt_correction --max_workers 4
```

**Benefits of using `--config_file`:**
- Site-specific dipole lengths (xarm, yarm) are automatically applied
- Site-specific remote reference assignments are used
- No need to manually list all sites
- Easy to modify processing parameters for different sites
- Supports different remote reference configurations per site

#### Batch Options
- `--sites [SITES ...]`       List of site directories to process (mutually exclusive with --config_file)
- `--config_file [FILE]`      Path to Processing.txt or similar config file (mutually exclusive with --sites)
- `--parent_dir [DIR]`        Parent directory containing site folders
- `--max_workers [N]`         Number of parallel processes (default: 4)
- All single-site options can be used in batch mode

### Filtering Examples

- Comb filter (default, 50 Hz):
  ```bash
  python EDL_Process.py --input_dir HDD5449 --apply_filtering --filter_type comb --save_processed_data
  ```
- Bandpass filter:
  ```bash
  python EDL_Process.py --input_dir HDD5449 --apply_filtering --filter_type bandpass --filter_low_freq 0.01 --filter_high_freq 1.0 --save_processed_data
  ```
- Adaptive filtering (with remote reference):
  ```bash
  python EDL_Process.py --input_dir HDD5449 --remote_reference HDD5974 --apply_filtering --filter_type adaptive --filter_reference_channel rBx --save_processed_data
  ```

### Heatmap and Frequency Analysis

- Coherence heatmaps:
  ```bash
  python EDL_Process.py --input_dir HDD5449 --plot_heatmaps --save_plots
  ```
- Frequency analysis (Welch, Multi-taper, Spectrogram):
    ```bash
    python EDL_Process.py --input_dir HDD5449 --perform_freq_analysis WMS --save_plots
    ```

### Output Naming Conventions

- Processed data: `[SiteName]_[SampleRate]Hz_Process.txt` (e.g., `HDD5449_10Hz_Process.txt`)
- Config file:    `[SiteName]_[SampleRate]Hz_Process.cfg`
- Plots:          `outputs/[SiteName]/[SiteName]_[SampleRate]Hz_[plot_type].png`
- Logs:           `process_ascii.log` (main), `[SiteName].log` (per site)

### lemimt.exe Integration
- **Windows users:** Place `lemimt.exe` in the `2025_EDL/` folder (the project root). The `--run_lemimt` option will run lemimt.exe automatically on the processed file.
- On macOS/Linux: A batch script is generated and the command is logged for manual execution on Windows.

## Example Workflows

#### Single Site, All Corrections, Save Everything
```bash
python EDL_Process.py --input_dir HDD5449 --plot_data --save_plots --save_processed_data --apply_drift_correction --apply_rotation --tilt_correction --remote_reference HDD5974 --perform_freq_analysis WMS --plot_heatmaps --run_lemimt
```

#### Batch Processing with Configuration File (Recommended)
```bash
python EDL_Batch.py --config_file Processing.txt --plot_data --save_plots --save_processed_data --apply_drift_correction --apply_rotation --tilt_correction --max_workers 4
```

#### Batch Processing with Manual Site Listing
```bash
python EDL_Batch.py --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_data --save_plots --save_processed_data --apply_drift_correction --apply_rotation --tilt_correction --max_workers 4
```

#### Filtering and Decimation
```bash
python EDL_Process.py --input_dir HDD5449 --apply_filtering --filter_type comb --decimate 2 5 --save_processed_data
```

#### Heatmap and Frequency Analysis
```bash
python EDL_Process.py --input_dir HDD5449 --plot_heatmaps --perform_freq_analysis WMS --save_plots
```

## Output Files
- Processed data: `[SiteName]_[SampleRate]Hz_Process.txt`
- Config:         `[SiteName]_[SampleRate]Hz_Process.cfg`
- Plots:          `outputs/[SiteName]/`
- Logs:           `process_ascii.log`, `[SiteName].log`

## Troubleshooting
- If `lemimt.exe` cannot be run on your platform, transfer the processed file and batch script to a Windows machine.
- For timezone issues, use `--timezone "Australia/Adelaide"` or your local zone.
- For remote reference, ensure both sites are present and have valid data.
- For more help, run `python EDL_Process.py --help` or `python EDL_Batch.py --help`.

---
