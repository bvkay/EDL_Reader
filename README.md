# EDL Magnetotelluric (MT) Data Processing Suite

A Python-based workflow for reading, processing, and analyzing MT time series data from Earth Data Logger (EDL) ASCII files, with support for batch processing, frequency analysis, corrections, and integration with LEMI MT executable tools.

---


## Key Features

- **ASCII Data Processing**: Reads EDL ASCII files (.BX, .BY, .BZ, .EX, .EY) and converts to physical units.
- **Metadata Handling**: Extracts metadata from `recorder.ini`, `*.gps` and `Processing.txt`.
- **Corrections**: Optional drift, rotation, and tilt correction (including remote reference tilt).
- **Smoothing**: Median/MAD and adaptive median filtering for outlier removal.
- **Filtering**: Comb, bandpass, highpass, lowpass, and adaptive filtering (for powerline and other noise).
- **Remote Reference**: Merges remote reference site data for robust processing.
- **Frequency Analysis**: Welch, multi-taper, and spectrogram analysis.
- **Heatmaps**: Coherence heatmaps, histograms, and barcode plots for quality control.
- **Batch Processing**: Parallel processing of multiple sites with clear logging and summary tables.
- **Configuration-Driven Processing**: Use `Processing.txt` to specify sites and their parameters.
- **lemimt.exe Integration**: Optionally runs lemimt.exe on processed output (Windows only).
- **Comprehensive Logging**: Main and per-site logs, with clear headers and summary tables.
- **Parallel Processing**: Multi-threaded batch processing with worker prefixes (P01_, P02_, etc.).

---

## File Structure

The project expects the following directory and file organization:

```
2025_EDL/
├── EDL_Process.py
├── EDL_Batch.py
├── EDL_Reader.py
├── Processing.txt
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

- **SiteName/**: Site folders, each with subfolders for each day (e.g., 294, 295), containing EDL ASCII data files.
- **outputs/**: All plots are saved here, organized by site.
- **sensors/**: Instrument response files (not relevant for LP fluxgate sensor).

---
## Installation

- Python 3.7+
- Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `configparser`, `glob`, `pytz` (for timezone support)
- **Windows users:** Place `lemimt.exe` in the `2025_EDL/` folder (the project root) for automated transfer function estimation.

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy configparser pytz
```

## Configuration File Format

The `Processing.txt` file can be used to specify sites and their parameters for batch processing. It supports both 3-column and 4-column formats:

### 4-Column Format (with remote reference)
```csv
Site, xarm, yarm, RemoteReference
SiteA-50m, 100.0, 100.0, SiteD-10m
SiteB-50m, 100.0, 100.0, [SiteC-10m; SiteD-10m]
SiteC-10m, 50.0, 50.0,
SiteD-10m, 50.0, 50.0, SiteA-50m

```
**Column Definitions:**
- **Site**: Site directory name
- **xarm**: X dipole length in meters
- **yarm**: Y dipole length in meters
- **RemoteReference**: (Optional) Remote reference site name, can be multiple with square brakets

## Batch Processing with --config_file

Both `EDL_Batch.py` and `Test_Batch.py` now support a `--config_file` argument:

```bash
python EDL_Batch.py --config_file Processing.txt --parent_dir . --plot_data --save_plots
```

- If only one site is listed in the config file, it will process that site (with a message suggesting single-site mode).
- If multiple sites are listed, all will be processed in batch mode.
- The `--sites` argument is still supported for manual site listing.

See the updated script help (`-h`) for more details.

### Single-Site Processing

```bash
python EDL_Process.py --site_name SiteA-50m --plot_timeseries --tilt_correction
```

#### Common Options
- `--site_name [SITE]`       Site directory name (required)
- `--plot_timeseries`        Show/save physical channel plots
- `--apply_drift_correction` Apply drift correction
- `--apply_rotation`         Apply rotation correction
- `--tilt_correction`        Apply tilt correction to Bx, By, rBx, rBy channels
- `--apply_smoothing`        Apply smoothing (median/adaptive)
- `--perform_freq_analysis`  Frequency analysis (W=Welch, M=Multi-taper, S=Spectrogram, e.g. 'WMS')
- `--apply_filtering`        Apply frequency filtering (see below)
- `--remote_reference [SITE]` Use another site as remote reference
- `--decimate [N ...]`       Decimate by factors N (e.g. 2 5 10)
- `--run_lemimt`             Run lemimt.exe on processed output (Windows only)

#### Example: Full Pipeline
```bash
python EDL_Process.py --site_name SiteA-50m --plot_timeseries --apply_drift_correction --apply_rotation --tilt_correction --apply_smoothing --smoothing_method adaptive --perform_freq_analysis WMS --remote_reference SiteD-10m --decimate 2 5 --run_lemimt
```

### Batch Processing

Use `Processing.txt` to specify sites and their parameters:
```bash
python EDL_Batch.py --input_config --plot_timeseries --tilt_correction --max_workers 4
```

#### Batch Options
- `--input_config`           Use Processing.txt for site configuration (required)
- `--max_workers [N]`        Number of parallel workers (default: 4)
- All single-site options can be used in batch mode

#### Example: Full Batch Pipeline
```bash
python EDL_Batch.py --input_config --plot_timeseries --apply_drift_correction --apply_rotation --tilt_correction --apply_smoothing --smoothing_method adaptive --perform_freq_analysis WMS --plot_heatmaps --run_lemimt --max_workers 4
```

### Filtering Examples

- Comb filter (default, 50 Hz):
  ```bash
  python EDL_Process.py --site_name SiteA-50m --apply_filtering --filter_type comb --plot_timeseries
  ```
- Bandpass filter:
  ```bash
  python EDL_Process.py --site_name SiteA-50m --apply_filtering --filter_type bandpass --filter_low_freq 0.01 --filter_high_freq 1.0 --plot_timeseries
  ```
- Adaptive filtering (with remote reference):
  ```bash
  python EDL_Process.py --site_name SiteA-50m --remote_reference SiteD-10m --apply_filtering --filter_type adaptive --filter_reference_channel rBx --plot_timeseries
  ```

### Heatmap and Frequency Analysis

- Coherence heatmaps:
  ```bash
  python EDL_Process.py --site_name SiteA-50m --plot_heatmaps
  ```
- Frequency analysis (Welch, Multi-taper, Spectrogram):
  ```bash
  python EDL_Process.py --site_name SiteA-50m --perform_freq_analysis WMS --plot_timeseries
  ```

### Advanced Processing Examples

#### Basic Processing
```bash
python EDL_Process.py --site_name SiteA-50m --plot_timeseries --tilt_correction
```

#### With Remote Reference
```bash
python EDL_Process.py --site_name SiteB-50m --remote_reference SiteD-10m --plot_timeseries
```

#### With Smoothing and Analysis
```bash
python EDL_Process.py --site_name SiteC-10m --apply_smoothing --plot_timeseries --run_lemimt
```

#### Batch Processing Examples
```bash
# Basic batch processing
python EDL_Batch.py --input_config --max_workers 4 --plot_timeseries --tilt_correction

# Full processing pipeline
python EDL_Batch.py --input_config --max_workers 2 --run_lemimt

# Advanced processing with filtering
python EDL_Process.py --site_name SiteA-50m --plot_timeseries --apply_filtering --filter_type comb --filter_notch_freq 50.0

# Heatmap generation
python EDL_Process.py --site_name SiteB-50m --plot_heatmaps --heatmap_thresholds "0.9,0.7,0.5"

# Decimation processing
python EDL_Process.py --site_name SiteC-10m --decimate 2 5 10 --plot_timeseries
```

### Output Naming Conventions

- Processed data: `P##_[SiteName]_[SampleRate]Hz_Process.txt` (e.g., `P01_SiteA-50m_10Hz_Process.txt`)
- Config file:    `P##_[SiteName]_[SampleRate]Hz_Process.cfg`
- Plots:          `outputs/[SiteName]/[SiteName]_[SampleRate]Hz_[plot_type].png`
- Logs:           `process_ascii.log` (main), `[SiteName].log` (per site)

**Note:** The `P##_` prefix is automatically assigned based on the worker number during parallel processing to prevent file overwrites.

### Remote Reference Processing
- Automatically reads station identifiers from `[recorder]` section of recorder.ini
- Supports multiple remote references per site: `[SiteA; SiteB]`
- Creates separate output files for each remote reference
- Enhanced error handling and logging for remote reference loading

### Parallel Processing
- Uses ThreadPoolExecutor for efficient parallel processing
- Worker numbers (P01_, P02_, etc.) prevent file overwrites
- Environment variables pass worker information to subprocesses
- Configurable number of workers with `--max_workers`

## Troubleshooting

### Common Issues

1. **Remote Reference Not Found**: Ensure the remote site directory exists and contains valid data files
2. **Station Identifier Issues**: Check that recorder.ini contains `station_long_identifier` in the `[recorder]` section
3. **File Overwrites**: Worker prefixes (P01_, P02_, etc.) are automatically applied during parallel processing
4. **lemimt.exe Not Found**: Place lemimt.exe in the project root directory for Windows processing

### Log Files
- Main log: `process_ascii.log`
- Site-specific logs: `[SiteName].log`
- Batch processing summary is printed to console
