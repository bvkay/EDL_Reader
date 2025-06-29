import os
import io
import argparse
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import welch, coherence, spectrogram, butter, filtfilt, iirnotch, lfilter  # Added filtering functions
from EDL_Reader import ASCIIReader  # Updated ASCII reader module
import configparser
import re
import traceback
import glob
from math import radians, cos, sin, asin, sqrt
import matplotlib.dates as mdates
from scipy.signal import windows
import time
import threading

# Global logging configuration
LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
CURRENT_LOG_LEVEL = "INFO"  # Can be overridden
BATCH_MODE = False  # Set to True during batch processing
SITE_NAME = "Unknown"  # Current site being processed

# Main log file for batch processing
MAIN_LOG_FILE = "process_ascii.log"

# Global metadata collection for summary table
PROCESSING_SUMMARY = {}

def print_processing_header(processing_type="MT PROCESSING"):
    """Print a clear header for the processing run."""
    header = f"""
{'='*60}
{'='*20} {processing_type} {'='*20}
{'='*60}
"""
    print(header)
    write_log(header.strip())

def create_processing_summary_table(site_name, metadata_dict):
    """Create a comprehensive summary table for the processing results.
    
    Args:
        site_name (str): Name of the site processed
        metadata_dict (dict): Dictionary containing all processing metadata
    """
    summary = f"""
{'='*80}
PROCESSING SUMMARY FOR {site_name}
{'='*80}

SITE CONFIGURATION:
{'-'*40}
Site Name:           {site_name}
X Dipole Length:     {metadata_dict.get('xarm', 'N/A')} m
Y Dipole Length:     {metadata_dict.get('yarm', 'N/A')} m
Remote Reference:    {metadata_dict.get('remote_reference', 'None')}
Sample Rate:         {metadata_dict.get('sample_rate', 'N/A')} Hz
Active Channels:     {', '.join(metadata_dict.get('active_channels', []))}

LOCATION & TIMING:
{'-'*40}
Latitude:            {metadata_dict.get('latitude', 'N/A')}°
Longitude:           {metadata_dict.get('longitude', 'N/A')}°
Elevation:           {metadata_dict.get('elevation', 'N/A')} m
Start Time:          {metadata_dict.get('start_time', 'N/A')}
End Time:            {metadata_dict.get('end_time', 'N/A')}
Timezone:            {metadata_dict.get('timezone', 'N/A')}
Duration:            {metadata_dict.get('duration', 'N/A')}

DATA STATISTICS:
{'-'*40}
Total Files:         {metadata_dict.get('total_files', 'N/A')}
Data Points:         {metadata_dict.get('data_points', 'N/A'):,}
Data Shape:          {metadata_dict.get('data_shape', 'N/A')}

CHANNEL MEANS (nT for magnetic, mV/km for electric):
{'-'*40}"""
    
    # Add channel means
    channel_means = metadata_dict.get('channel_means', {})
    for channel, mean_val in channel_means.items():
        summary += f"\n{channel:15} {mean_val:10.2f}"
    
    # Add field strengths
    field_strengths = metadata_dict.get('field_strengths', {})
    if field_strengths:
        summary += f"\n\nFIELD STRENGTHS:"
        summary += f"\n{'-'*40}"
        for field_type, value in field_strengths.items():
            summary += f"\n{field_type:20} {value:10.2f} nT"
    
    # Add corrections applied
    corrections = metadata_dict.get('corrections_applied', [])
    if corrections:
        summary += f"\n\nCORRECTIONS APPLIED:"
        summary += f"\n{'-'*40}"
        for correction in corrections:
            summary += f"\n• {correction}"
    
    # Add tilt correction angles if applied
    tilt_angles = metadata_dict.get('tilt_angles', {})
    if tilt_angles:
        summary += f"\n\nTILT CORRECTION ANGLES:"
        summary += f"\n{'-'*40}"
        for angle_type, angle_val in tilt_angles.items():
            summary += f"\n{angle_type:25} {angle_val:8.2f}°"
    
    summary += f"\n\n{'='*80}\n"
    
    return summary

def set_log_level(level):
    """Set the global log level."""
    global CURRENT_LOG_LEVEL
    if level.upper() in LOG_LEVELS:
        CURRENT_LOG_LEVEL = level.upper()
    else:
        print(f"Invalid log level: {level}. Using INFO.")
        CURRENT_LOG_LEVEL = "INFO"

def set_batch_mode(enabled=True):
    """Enable or disable batch mode logging."""
    global BATCH_MODE
    BATCH_MODE = enabled

def set_site_name(site_name):
    """Set the current site name for logging."""
    global SITE_NAME
    SITE_NAME = site_name

def get_site_log_file(site_name):
    """Get the site-specific log file path."""
    return f"{site_name}.log"

def write_log(message, level="INFO", site_name=None):
    """Enhanced logging function with site-specific files and level filtering.
    
    Args:
        message (str): The log message.
        level (str, optional): Log level. Defaults to "INFO".
        site_name (str, optional): Site name for site-specific logging.
    """
    # Check log level filtering
    if LOG_LEVELS.get(level.upper(), 0) < LOG_LEVELS.get(CURRENT_LOG_LEVEL, 0):
        return
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    site_prefix = f"[{site_name or SITE_NAME}] " if site_name or SITE_NAME != "Unknown" else ""
    log_message = f"{timestamp} - {level.upper()} - {site_prefix}{message}\n"
    
    # Write to main log file
    with open(MAIN_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_message)
    
    # Write to site-specific log file if in batch mode and site name is provided
    if BATCH_MODE and (site_name or SITE_NAME != "Unknown"):
        site_log_file = get_site_log_file(site_name or SITE_NAME)
        with open(site_log_file, "a", encoding="utf-8") as site_log:
            site_log.write(log_message)
    
    # Print to console for ERROR and WARNING levels, or if not in batch mode
    if level.upper() in ["ERROR", "WARNING"] or not BATCH_MODE:
        print(log_message.strip())

def write_batch_log(message, level="INFO"):
    """Write batch-specific log messages."""
    write_log(message, level, "BATCH")

def write_site_log(message, level="INFO"):
    """Write site-specific log messages."""
    write_log(message, level, SITE_NAME)

def create_batch_summary(sites, results, start_time):
    """Create a batch processing summary report.
    
    Args:
        sites (list): List of sites processed
        results (dict): Results for each site
        start_time (datetime): Batch start time
    """
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    summary = f"""
{'='*80}
BATCH PROCESSING SUMMARY
{'='*80}
Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration}
Total Sites: {len(sites)}

SITE RESULTS:
{'-'*40}"""
    
    successful = 0
    failed = 0
    
    for site in sites:
        if site in results:
            status = results[site].get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                successful += 1
                summary += f"\n✓ {site}: SUCCESS"
                if 'data_shape' in results[site]:
                    summary += f" (Data: {results[site]['data_shape']})"
            else:
                failed += 1
                summary += f"\n✗ {site}: FAILED"
                if 'error' in results[site]:
                    summary += f" - {results[site]['error']}"
        else:
            failed += 1
            summary += f"\n✗ {site}: NOT PROCESSED"
    
    summary += f"""

SUMMARY:
{'-'*40}
Successful: {successful}
Failed: {failed}
Success Rate: {(successful/len(sites)*100):.1f}%

Log Files:
- Main log: {MAIN_LOG_FILE}
- Site logs: {', '.join([f'{site}.log' for site in sites])}
{'='*80}
"""
    
    # Write summary to main log
    with open(MAIN_LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(summary)
    
    # Print summary to console
    print(summary)
    
    return summary

def dict_to_datetime(time_dict):
    """Converts a time dictionary to a datetime object.
    
    Args:
        time_dict (dict): Dictionary with keys "year", "month", "day", "hour", "minute", "second".
    
    Returns:
        datetime.datetime: Constructed datetime.
    """
    return datetime.datetime(
        year=time_dict["year"],
        month=time_dict["month"],
        day=time_dict["day"],
        hour=time_dict["hour"],
        minute=time_dict["minute"],
        second=time_dict["second"]
    )


def apply_drift_correction_to_df(df, metadata):
    """Applies linear drift correction to the DataFrame.
    
    Correction: corrected_time = time + (time / total_duration) * time_drift,
    and adds a datetime column.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'time' column (seconds from survey start).
        metadata (dict): Contains "start_time", "finish_time", and "time_drift".
    
    Returns:
        pd.DataFrame: DataFrame with "time_corrected" and "time_corrected_dt".
    """
    try:
        write_log("Starting drift correction calculation...")
        write_log(f"Input DataFrame shape: {df.shape}")
        write_log(f"Input DataFrame columns: {list(df.columns)}")
        
        # Check for NaNs in time column
        if df['time'].isnull().any():
            n_nans = df['time'].isnull().sum()
            write_log(f"WARNING: Found {n_nans} NaN values in time column", level="WARNING")
        
        start_dt = dict_to_datetime(metadata["start_time"])
        finish_dt = dict_to_datetime(metadata["finish_time"])
        total_duration = (finish_dt - start_dt).total_seconds()
        drift_value = metadata["time_drift"]
        
        write_log(f"Start time: {start_dt}")
        write_log(f"Finish time: {finish_dt}")
        write_log(f"Total duration: {total_duration} seconds")
        write_log(f"Drift value: {drift_value}")
        
        # Check for division by zero or very small total_duration
        if total_duration <= 0:
            write_log("WARNING: Total duration is zero or negative, skipping drift correction", level="WARNING")
            df["time_corrected"] = df["time"]
            df["time_corrected_dt"] = df["time"].apply(
                lambda s: start_dt + datetime.timedelta(seconds=s)
            )
            return df
        
        df["time_corrected"] = df["time"] + (df["time"] / total_duration) * drift_value
        df["time_corrected_dt"] = df["time_corrected"].apply(
            lambda s: start_dt + datetime.timedelta(seconds=s)
        )
        
        write_log("Drift correction calculation completed successfully")
        return df
        
    except Exception as e:
        write_log(f"Error in drift correction: {e}", level="ERROR")
        write_log(f"Drift correction traceback: {traceback.format_exc()}", level="ERROR")
        raise


def convert_counts_to_physical_units(df, metadata):
    """Converts raw counts into physical units.
    
    Magnetics:
      Bx = (chan0/2**23 - 1) * 70000  
      Bz = (chan1/2**23 - 1) * 70000  
      By = -(chan2/2**23 - 1) * 70000  
    Electrics:
      Ex = -(chan7/2**23 - 1) * (100000 / xarm)  
      Ey = -(chan6/2**23 - 1) * (100000 / yarm)
    
    Args:
        df (pd.DataFrame): DataFrame with channels "chan0", "chan1", etc.
        metadata (dict): Contains "xarm" and "yarm".
    
    Returns:
        pd.DataFrame: DataFrame with new columns "Bx", "Bz", "By", "Ex", and "Ey".
    """
    divisor = 10**7
    df["Bx"] = (df["chan0"].astype(float) / divisor) * 70000.0
    df["Bz"] = (df["chan1"].astype(float) / divisor) * 70000.0
    df["By"] = -(df["chan2"].astype(float) / divisor) * 70000.0
    xarm = metadata.get("xarm", 1.0)
    yarm = metadata.get("yarm", 1.0)
    df["Ex"] = -(df["chan7"].astype(float) / xarm)
    df["Ey"] = -(df["chan6"].astype(float) / yarm)
    return df


def rotate_data(df, metadata):
    """Rotates horizontal magnetic (and optionally electric) fields.
    
    Computes angle_avg = mean(arctan2(By, Bx)) and rotates the horizontal fields.
    If metadata["erotate"] == 1, the electric fields are also rotated.
    
    Args:
        df (pd.DataFrame): DataFrame with "Bx", "By", "Bz", "Ex", "Ey".
        metadata (dict): Contains "erotate".
    
    Returns:
        pd.DataFrame: DataFrame with rotated fields ("Hx", "Dx", "Z_rot", and optionally "Ex_rot", "Ey_rot").
    """
    angles = np.arctan2(df["By"].values, df["Bx"].values)
    angle_avg = np.mean(angles)
    write_log(f"Computed rotation angle: {np.degrees(angle_avg):.2f} degrees")

    df["Hx"] = df["Bx"] * np.cos(angle_avg) + df["By"] * np.sin(angle_avg)
    df["Dx"] = df["By"] * np.cos(angle_avg) - df["Bx"] * np.sin(angle_avg)
    df["Z_rot"] = df["Bz"]

    if metadata.get("erotate", 0) == 1:
        df["Ex_rot"] = df["Ex"] * np.cos(angle_avg) + df["Ey"] * np.sin(angle_avg)
        df["Ey_rot"] = df["Ey"] * np.cos(angle_avg) - df["Ex"] * np.sin(angle_avg)
    else:
        df["Ex_rot"] = df["Ex"]
        df["Ey_rot"] = df["Ey"]
    return df

def tilt_correction(df, include_remote_reference=False):
    """Applies tilt correction to make mean(By) = 0 for all Bx, By, rBx, rBy channels.
    
    Args:
        df (pd.DataFrame): DataFrame with Bx, By columns and optionally rBx, rBy columns
        include_remote_reference (bool): Deprecated parameter, kept for compatibility
        
    Returns:
        tuple: (corrected_df, tilt_angle_degrees)
    """
    df_corrected = df.copy()
    tilt_angle_degrees = 0.0
    
    # Calculate tilt angle from main channels
    if 'Bx' in df_corrected.columns and 'By' in df_corrected.columns:
        tilt_angle = np.arctan2(df_corrected['By'].mean(), df_corrected['Bx'].mean())
        tilt_angle_degrees = np.degrees(tilt_angle)
        
        # Apply rotation to main channels
        Bx_rot = df_corrected['Bx'] * np.cos(-tilt_angle) - df_corrected['By'] * np.sin(-tilt_angle)
        By_rot = df_corrected['Bx'] * np.sin(-tilt_angle) + df_corrected['By'] * np.cos(-tilt_angle)
        
        df_corrected['Bx'] = Bx_rot
        df_corrected['By'] = By_rot
        
        write_log(f"Applied tilt correction to main channels: {tilt_angle_degrees:.2f} degrees")
    
    # Always apply tilt correction to remote reference channels if available
    if 'rBx' in df_corrected.columns and 'rBy' in df_corrected.columns:
        # Calculate separate tilt angle for remote reference channels using their means
        remote_tilt_angle = np.arctan2(df_corrected['rBy'].mean(), df_corrected['rBx'].mean())
        remote_tilt_angle_degrees = np.degrees(remote_tilt_angle)
        
        # Apply rotation to remote reference channels using their own tilt angle
        rBx_rot = df_corrected['rBx'] * np.cos(-remote_tilt_angle) - df_corrected['rBy'] * np.sin(-remote_tilt_angle)
        rBy_rot = df_corrected['rBx'] * np.sin(-remote_tilt_angle) + df_corrected['rBy'] * np.cos(-remote_tilt_angle)
        
        df_corrected['rBx'] = rBx_rot
        df_corrected['rBy'] = rBy_rot
        
        write_log(f"Applied tilt correction to remote reference channels: {remote_tilt_angle_degrees:.2f} degrees")
    
    return df_corrected, tilt_angle_degrees

def log_summary_stats(df, label="Summary"):
    """Logs summary statistics for magnetic fields and derived fields.

    Computes mean and standard deviation for Bx, By, Bz, total magnetic field,
    and horizontal magnetic field.

    Args:
        df (pd.DataFrame): DataFrame with "Bx", "By", "Bz".
        label (str): Label for the statistics (e.g., "Before tilt correction").

    Returns:
        None
    """
    mean_Bx = df["Bx"].mean()
    std_Bx = df["Bx"].std()
    mean_By = df["By"].mean()
    std_By = df["By"].std()
    mean_Bz = df["Bz"].mean()
    std_Bz = df["Bz"].std()
    B_total = np.sqrt(df["Bx"]**2 + df["By"]**2 + df["Bz"]**2)
    mean_B_total = B_total.mean()
    std_B_total = B_total.std()
    B_horizontal = np.sqrt(df["Bx"]**2 + df["By"]**2)
    mean_B_horizontal = B_horizontal.mean()
    std_B_horizontal = B_horizontal.std()

    stats_text = (
        f"{label}:\n"
        f"  Bx: mean = {mean_Bx:.2f}, std = {std_Bx:.2f}\n"
        f"  By: mean = {mean_By:.2f}, std = {std_By:.2f}\n"
        f"  Bz: mean = {mean_Bz:.2f}, std = {std_Bz:.2f}\n"
        f"  Total B: mean = {mean_B_total:.2f}, std = {std_B_total:.2f}\n"
        f"  Horizontal B: mean = {mean_B_horizontal:.2f}, std = {std_B_horizontal:.2f}"
    )
    write_log(stats_text)

### Smoothing Functions: Only Median/MAD and Adaptive Median Filtering are retained ###

def smooth_outlier_points(df, channels=["Bx", "By", "Bz", "Ex", "Ey"], window=50, threshold=3.0):
    """Applies outlier detection and smoothing using rolling median and MAD.
    
    Outliers are detected and then interpolated linearly.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        channels (list, optional): Channels to process. Defaults to ["Bx", "By", "Bz", "Ex", "Ey"].
        window (int, optional): Rolling window size. Defaults to 50.
        threshold (float, optional): Multiplier for MAD. Defaults to 3.0.
    
    Returns:
        tuple: (df, outlier_info) where outlier_info maps channels to outlier intervals.
    """
    outlier_info = {}
    for ch in channels:
        mask = detect_outliers(df[ch], window=window, threshold=threshold)
        # Use datetime column if available, otherwise use index
        time_col = df["datetime"] if "datetime" in df.columns else df.index
        intervals = get_intervals_from_mask(mask, time_col)
        outlier_info[ch] = intervals
        df[ch] = smooth_outliers(df[ch], mask)
    return df, outlier_info


def detect_outliers(series, window=50, threshold=3.0):
    """Detects outliers using a rolling median and MAD approach.
    
    Args:
        series (pd.Series): Input time-series.
        window (int, optional): Rolling window size. Defaults to 50.
        threshold (float, optional): Multiplier for MAD. Defaults to 3.0.
    
    Returns:
        pd.Series: Boolean Series marking outliers.
    """
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    abs_diff = (series - rolling_median).abs()
    rolling_mad = abs_diff.rolling(window=window, center=True, min_periods=1).median()
    rolling_mad = rolling_mad.replace(0, 1e-6)
    return abs_diff > threshold * rolling_mad


def smooth_outliers(series, outlier_mask):
    """Interpolates linearly over regions marked as outliers.
    
    Args:
        series (pd.Series): Input time-series.
        outlier_mask (pd.Series): Boolean mask indicating outliers.
    
    Returns:
        pd.Series: Smoothed series.
    """
    series_clean = series.copy()
    series_clean[outlier_mask] = np.nan
    return series_clean.interpolate(method="linear")


def get_intervals_from_mask(mask, time_values):
    """Extracts (start_time, end_time) intervals where the mask is True.
    
    Args:
        mask (pd.Series): Boolean mask from outlier detection.
        time_values (pd.Series): Corresponding time values.
    
    Returns:
        list: List of tuples (start_time, end_time).
    """
    intervals = []
    in_interval = False
    start_time = None
    for i, flag in enumerate(mask):
        if flag and not in_interval:
            in_interval = True
            start_time = time_values.iloc[i]
        elif not flag and in_interval:
            intervals.append((start_time, time_values.iloc[i - 1]))
            in_interval = False
    if in_interval:
        intervals.append((start_time, time_values.iloc[-1]))
    return intervals


def smooth_median_mad(df, channels=["Bx", "By", "Bz", "Ex", "Ey"], window=50, threshold=3.0):
    """Wrapper for the median/MAD smoothing method.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        channels (list, optional): Channels to smooth. Defaults to ["Bx", "By", "Bz", "Ex", "Ey"].
        window (int, optional): Rolling window size. Defaults to 50.
        threshold (float, optional): Multiplier for MAD. Defaults to 3.0.
    
    Returns:
        tuple: (df, outlier_info)
    """
    return smooth_outlier_points(df, channels=channels, window=window, threshold=threshold)


def smooth_adaptive_median(signal, min_window=51, max_window=2500, threshold=10.0):
    """Applies an adaptive median filter to a 1D numpy array.
    
    For each sample, starts with a window of size min_window (odd) and expands until the sample
    is within threshold * MAD of the median or until max_window is reached.
    
    Args:
        signal (np.ndarray): Input 1D array.
        min_window (int, optional): Minimum window size (odd). Defaults to 51.
        max_window (int, optional): Maximum window size (odd). Defaults to 2500.
        threshold (float, optional): Threshold multiplier for MAD. Defaults to 10.0.
    
    Returns:
        np.ndarray: Smoothed array.
    """
    smoothed = np.copy(signal)
    n = len(signal)
    if min_window % 2 == 0:
        min_window += 1
    if max_window % 2 == 0:
        max_window += 1

    for i in range(n):
        window_size = min_window
        smoothed_value = signal[i]
        while window_size <= max_window:
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window = signal[start:end]
            med = np.median(window)
            mad = np.median(np.abs(window - med))
            if mad == 0:
                mad = 1e-6
            if np.abs(signal[i] - med) <= threshold * mad:
                smoothed_value = med
                break
            window_size += 2
        smoothed[i] = smoothed_value
    return smoothed


def smooth_adaptive(df, channels=["Bx", "By", "Bz", "Ex", "Ey"], min_window=3, max_window=2500, threshold=10.0):
    """Applies adaptive median filtering to specified channels of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'time' column and measurement channels.
        channels (list, optional): Channels to smooth. Defaults to ["Bx", "By", "Bz", "Ex", "Ey"].
        min_window (int, optional): Minimum window size (odd). Defaults to 3.
        max_window (int, optional): Maximum window size (odd). Defaults to 2500.
        threshold (float, optional): Threshold multiplier for MAD. Defaults to 10.0.
    
    Returns:
        tuple: (df_adapt, {})
    """
    df_adapt = df.copy()
    for ch in channels:
        data = df_adapt[ch].values
        df_adapt[ch] = smooth_adaptive_median(data, min_window=min_window, max_window=max_window, threshold=threshold)
    return df_adapt, {}


### Plotting Functions ###

def plot_power_spectra(df, channels, fs, nperseg=1024, save_plots=False, site_name="UnknownSite", output_dir="."):
    """Plots power spectra for specified channels.
    
    Args:
        df (pd.DataFrame): DataFrame with channel data.
        channels (list): List of channel names to plot.
        fs (float): Sampling frequency.
        nperseg (int): Number of points per segment for FFT.
        save_plots (bool): Whether to save plots to files.
        site_name (str): Site name for plot titles and filenames.
        output_dir (str): Directory to save plots in.
    """
    try:
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3*len(channels)))
        if len(channels) == 1:
            axes = [axes]
        
        for i, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].dropna()
                if len(data) > 0:
                    f, Pxx = welch(data, fs=fs, nperseg=min(nperseg, len(data)//2))
                    axes[i].semilogy(f, Pxx)
                    axes[i].set_xlabel('Frequency [Hz]')
                    axes[i].set_ylabel('Power Spectral Density')
                    axes[i].set_title(f'{channel} - {site_name}')
                    axes[i].grid(True)
                else:
                    axes[i].text(0.5, 0.5, f'No data for {channel}', ha='center', va='center', transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, f'Channel {channel} not found', ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(output_dir, f"{site_name}_power_spectra.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            write_log(f"Power spectra plot saved as {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        write_log(f"Error in plot_power_spectra: {e}", level="ERROR")


def plot_coherence_plots(df, fs, nperseg=1024, save_plots=False, site_name="UnknownSite", output_dir="."):
    """Plots coherence between MT-specific channel pairs.
    
    Standard MT pairs: Bx-Ey, By-Ex
    Remote reference pairs (if available): Bx-rBx, By-rBy, Ex-rEx, Ey-rEy
    
    Args:
        df (pd.DataFrame): DataFrame with channel data.
        fs (float): Sampling frequency.
        nperseg (int): Number of points per segment for FFT.
        save_plots (bool): Whether to save plots to files.
        site_name (str): Site name for plot titles and filenames.
        output_dir (str): Directory to save plots in.
    """
    try:
        # Define MT-specific coherence pairs
        pairs = []
        
        # Standard MT pairs
        if 'Bx' in df.columns and 'Ey' in df.columns:
            pairs.append(('Bx', 'Ey'))
        if 'By' in df.columns and 'Ex' in df.columns:
            pairs.append(('By', 'Ex'))
        
        # Remote reference pairs (if available)
        if 'rBx' in df.columns:
            if 'Bx' in df.columns:
                pairs.append(('Bx', 'rBx'))
            if 'Ex' in df.columns:
                pairs.append(('Ex', 'rEx'))
        if 'rBy' in df.columns:
            if 'By' in df.columns:
                pairs.append(('By', 'rBy'))
            if 'Ey' in df.columns:
                pairs.append(('Ey', 'rEy'))
        
        # Additional cross-coherence pairs for remote reference
        if 'rBx' in df.columns and 'Ey' in df.columns:
            pairs.append(('rBx', 'Ey'))
        if 'rBy' in df.columns and 'Ex' in df.columns:
            pairs.append(('rBy', 'Ex'))
        
        if not pairs:
            write_log(f"No valid coherence pairs found for {site_name}", level="WARNING")
            return
        
        # Create figure with subplots for each pair
        n_pairs = len(pairs)
        fig_height = 4 * n_pairs  # 4 inches per pair
        fig, axes = plt.subplots(n_pairs, 2, figsize=(12, fig_height))  # Changed from 3 to 2 columns
        
        # Handle single pair case
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for i, (ch1, ch2) in enumerate(pairs):
            if ch1 in df.columns and ch2 in df.columns:
                data1 = df[ch1].dropna()
                data2 = df[ch2].dropna()
                if len(data1) > 0 and len(data2) > 0:
                    # Ensure both channels have the same length
                    min_len = min(len(data1), len(data2))
                    data1 = data1[:min_len]
                    data2 = data2[:min_len]
                    
                    f, Cxy = coherence(data1, data2, fs=fs, nperseg=min(nperseg, min_len//2))
                    axes[i, 0].plot(f, Cxy)
                    axes[i, 0].set_xlabel('Frequency [Hz]')
                    axes[i, 0].set_ylabel('Coherence')
                    axes[i, 0].set_title(f'{ch1}-{ch2} Coherence - {site_name}')
                    axes[i, 0].grid(True)
                    axes[i, 0].set_ylim(0, 1)
                    
                    # Add subplot for filter response
                    axes[i, 1].plot(f, np.abs(Cxy))
                    axes[i, 1].set_xlabel('Frequency [Hz]')
                    axes[i, 1].set_ylabel('Magnitude')
                    axes[i, 1].set_title(f'{ch1}-{ch2} Filter Response - {site_name}')
                    axes[i, 1].grid(True)
                else:
                    axes[i, 0].text(0.5, 0.5, f'No data for {ch1}-{ch2}', ha='center', va='center', transform=axes[i, 0].transAxes)
                    axes[i, 1].text(0.5, 0.5, f'No data for {ch1}-{ch2}', ha='center', va='center', transform=axes[i, 1].transAxes)
            else:
                missing = []
                if ch1 not in df.columns:
                    missing.append(ch1)
                if ch2 not in df.columns:
                    missing.append(ch2)
                axes[i, 0].text(0.5, 0.5, f'Channels not found: {missing}', ha='center', va='center', transform=axes[i, 0].transAxes)
                axes[i, 1].text(0.5, 0.5, f'Channels not found: {missing}', ha='center', va='center', transform=axes[i, 1].transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(output_dir, f"{site_name}_coherence.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            write_log(f"Coherence plot saved as {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        write_log(f"Error in plot_coherence_plots: {e}", level="ERROR")


def plot_physical_channels(df, boundaries=None, plot_boundaries=True,
                           smoothed_intervals=None, plot_smoothed_windows=True,
                           tilt_corrected=False, save_plots=False, site_name="UnknownSite", output_dir=".",
                           drift_corrected=False, rotated=False, plot_tilt=False, plot_original_data=False,
                           gps_data=None, tilt_angle_degrees=None, remote_gps_data=None, distance_km=None,
                           remote_reference_site=None, xarm=None, yarm=None, skip_minutes=[0, 0], timezone="UTC"):
    """Plots physical channel data in subplots.
    
    Since tilt correction overwrites the original 'Bx' and 'By' columns,
    this function always uses "Bx", "By", etc.
    
    Args:
        df (pd.DataFrame): DataFrame with measurement channels.
        boundaries (list): List of time values for file boundaries.
        plot_boundaries (bool): Whether to draw vertical lines at boundaries.
        smoothed_intervals (dict): Dictionary of smoothed window intervals.
        plot_smoothed_windows (bool): Whether to shade smoothed windows.
        tilt_corrected (bool): For labeling purposes (unused in column mapping).
        save_plots (bool): If True, saves the figure; otherwise displays it.
        site_name (str): Site name used for naming the saved file.
        output_dir (str, optional): Directory to save plots. Defaults to current directory.
        drift_corrected (bool): Whether drift correction was applied.
        rotated (bool): Whether rotation was applied.
        plot_tilt (bool): Whether to plot tilt-corrected data in timeseries (default: False).
        plot_original_data (bool): Whether to plot original data alongside corrected data (default: False).
        gps_data (dict, optional): Dictionary containing GPS coordinates and altitude.
        tilt_angle_degrees (float, optional): Tilt angle in degrees for tilt-corrected data.
        remote_gps_data (dict, optional): GPS data from remote reference site.
        distance_km (float, optional): Distance between main and remote sites in km.
        remote_reference_site (str, optional): Name of the remote reference site.
        xarm (float, optional): X dipole length in meters.
        yarm (float, optional): Y dipole length in meters.
        skip_minutes (list): Minutes to skip from start and end of data (e.g., [30, 20] skips first 30 and last 20 minutes).
        timezone (str): Timezone for the data.
    
    Returns:
        None
    """
    # Determine which channels to plot based on what corrections were applied
    if rotated and 'Hx' in df.columns and 'Dx' in df.columns:
        # Use rotated channels if available
        channel_mapping = {"Hx": "Hx", "Dx": "Dx", "Z_rot": "Z_rot", "Ex_rot": "Ex_rot", "Ey_rot": "Ey_rot"}
        h_field = np.sqrt(df['Hx']**2 + df['Dx']**2) if all(col in df.columns for col in ['Hx', 'Dx']) else None
    else:
        # Use standard channels
        channel_mapping = {"Bx": "Bx", "By": "By", "Bz": "Bz", "Ex": "Ex", "Ey": "Ey"}
        h_field = np.sqrt(df['Bx']**2 + df['By']**2) if all(col in df.columns for col in ['Bx', 'By']) else None

    # Add remote reference channels if present
    remote_tilt_angle = None
    if 'rBx' in df.columns and 'rBy' in df.columns:
        channel_mapping.update({"rBx": "rBx", "rBy": "rBy"})
        write_log(f"Adding remote reference channels (rBx, rBy) to plot")
        
        # Check if remote reference channels were tilt-corrected
        if 'rBx_original' in df.columns and 'rBy_original' in df.columns:
            # Calculate remote reference tilt angle using rBy and rBx means
            rBy_mean = df['rBy_original'].mean()
            rBx_mean = df['rBx_original'].mean()
            remote_tilt_angle = np.arctan2(rBy_mean, rBx_mean) * 180 / np.pi
            write_log(f"Remote reference tilt correction angle (from rBy/rBx means): {remote_tilt_angle:.2f}°")

    # Calculate total field strength for magnetic channels
    total_field_strength = None
    if 'Bx' in df.columns and 'By' in df.columns and 'Bz' in df.columns:
        total_field_strength = np.sqrt(df['Bx']**2 + df['By']**2 + df['Bz']**2)
    elif 'Hx' in df.columns and 'Dx' in df.columns and 'Z_rot' in df.columns:
        total_field_strength = np.sqrt(df['Hx']**2 + df['Dx']**2 + df['Z_rot']**2)

    # Dynamically adjust figure size based on number of channels
    num_channels = len(channel_mapping)
    fig_height = max(12, num_channels * 2.5)  # At least 12 inches, 2.5 inches per channel
    fig, axes = plt.subplots(num_channels, 1, figsize=(14, fig_height), sharex=True)
    
    # Handle case where there's only one channel (axes becomes a single object, not an array)
    if num_channels == 1:
        axes = [axes]
    
    # Create main title
    main_title = f"Time Series for {site_name}"
    if remote_reference_site:
        main_title += f" with Remote Reference {remote_reference_site}"
    fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
    
    # Determine x-axis data
    if "datetime" in df.columns:
        x_data = df["datetime"]
        # Get timezone abbreviation for x-axis label
        tz_str = timezone[0] if isinstance(timezone, list) else timezone
        tz_abbrev = get_timezone_abbreviation(tz_str)
        x_label = f"Time ({tz_abbrev})"
    else:
        x_data = df["time"]
        x_label = "Time (seconds)"
    
    # Prepare text boxes content
    left_text = []
    right_text = []
    center_text = []
    
    # GPS coordinates on the left
    if gps_data:
        if gps_data.get("latitude") is not None:
            left_text.append(f"Latitude: {gps_data['latitude']:.6f}°")
        if gps_data.get("longitude") is not None:
            left_text.append(f"Longitude: {gps_data['longitude']:.6f}°")
        if gps_data.get("altitude") is not None:
            left_text.append(f"Altitude: {gps_data['altitude']} m")
    
    # Field strengths on the right
    if h_field is not None:
        h_mean = h_field.mean()
        h_std = h_field.std()
        right_text.append(f"Horizontal Magnetic Field: {h_mean:.2f} ± {h_std:.2f} nT")
    if total_field_strength is not None:
        total_mean = total_field_strength.mean()
        total_std = total_field_strength.std()
        right_text.append(f"Total Magnetic Field: {total_mean:.2f} ± {total_std:.2f} nT")
    
    # Distance to remote reference in the center
    if distance_km is not None:
        center_text.append(f"Distance to Remote Reference: {distance_km:.2f} km")
    
    # Tilt correction angles in the center
    if tilt_angle_degrees is not None:
        center_text.append(f"Main Site Tilt: {tilt_angle_degrees:.2f}°")
    if remote_tilt_angle is not None:
        center_text.append(f"Remote Reference Tilt: {remote_tilt_angle:.2f}°")
    
    # Add text boxes closer to the subplots to minimize white space
    text_y_position = 0.96  # Position below title but close to subplots
    if left_text:
        fig.text(0.02, text_y_position, '\n'.join(left_text), fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.15", facecolor="lightgray", alpha=0.8),
                verticalalignment='top')
    
    if right_text:
        fig.text(0.98, text_y_position, '\n'.join(right_text), fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.15", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', horizontalalignment='right')
    
    if center_text:
        fig.text(0.5, text_y_position, '\n'.join(center_text), fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.15", facecolor="lightgreen", alpha=0.8),
                verticalalignment='top', horizontalalignment='center')
    
    for ax, (label, col) in zip(axes, channel_mapping.items()):
        if col in df.columns:
            # Set subplot title with dipole length for Ex/Ey
            if col in ["Ex", "Ey", "Ex_rot", "Ey_rot"]:
                dipole_length = xarm if col in ["Ex", "Ex_rot"] else yarm
                if dipole_length:
                    ax.set_title(f"{label} ({dipole_length:.1f} m)", fontsize=12, fontweight='bold')
                else:
                    ax.set_title(label, fontsize=12, fontweight='bold')
            else:
                ax.set_title(label, fontsize=12, fontweight='bold')
            
            # For Bx and By, if plot_tilt is enabled, plot tilt-corrected data by default
            if plot_tilt and col in ["Bx", "By"] and f"{col}_original" in df.columns:
                # Always plot tilt-corrected data when plot_tilt is enabled
                ax.plot(x_data, df[col], label=f"{col} (tilt-corrected)", color='blue', linewidth=1.5)
                tilt_mean = df[col].mean()
                ax.axhline(tilt_mean, color='blue', linestyle='-', alpha=0.5, label=f"Mean ({col} tilt-corrected) = {tilt_mean:.2f}")
                
                # Only plot original data if plot_original_data is also enabled
                if plot_original_data:
                    ax.plot(x_data, df[f"{col}_original"], label=f"{col} (original)", color='gray', alpha=0.7)
                    orig_mean = df[f"{col}_original"].mean()
                    ax.axhline(orig_mean, color='gray', linestyle=':', label=f"Mean ({col} original) = {orig_mean:.2f}")
            
            # For remote reference channels, handle tilt correction similarly with different colors
            elif plot_tilt and col in ["rBx", "rBy"] and f"{col}_original" in df.columns:
                # Plot tilt-corrected remote reference data with different color (purple)
                ax.plot(x_data, df[col], label=f"{col} (tilt-corrected)", color='purple', linewidth=1.5)
                tilt_mean = df[col].mean()
                ax.axhline(tilt_mean, color='purple', linestyle='-', alpha=0.5, label=f"Mean ({col} tilt-corrected) = {tilt_mean:.2f}")
                
                # Only plot original data if plot_original_data is also enabled
                if plot_original_data:
                    ax.plot(x_data, df[f"{col}_original"], label=f"{col} (original)", color='orange', alpha=0.7)
                    orig_mean = df[f"{col}_original"].mean()
                    ax.axhline(orig_mean, color='orange', linestyle=':', label=f"Mean ({col} original) = {orig_mean:.2f}")
            
            # For remote reference channels without tilt correction, use different color
            elif col in ["rBx", "rBy"]:
                # Plot remote reference data with different color (purple)
                ax.plot(x_data, df[col], label=col, color='purple', linewidth=1.5)
                mean_val = df[col].mean()
                ax.axhline(mean_val, color='purple', linestyle='-', alpha=0.5, label=f"Mean = {mean_val:.2f}")
            
            else:
                # Standard plotting for all other channels (including Bz, Ex, Ey when plot_tilt is enabled)
                # If plot_original_data is enabled and we have original columns, plot them
                if plot_original_data and f"{col}_original" in df.columns:
                    ax.plot(x_data, df[f"{col}_original"], label=f"{col} (original)", color='gray', alpha=0.7)
                    orig_mean = df[f"{col}_original"].mean()
                    ax.axhline(orig_mean, color='gray', linestyle=':', label=f"Mean ({col} original) = {orig_mean:.2f}")
                
                # Plot the main data
                ax.plot(x_data, df[col], label=col, alpha=0.7)
                mean_val = df[col].mean()
                ax.axhline(mean_val, color='black', linestyle=':', label=f"Mean = {mean_val:.2f}")
            
            # Set y-axis label
            if col in ["Bx", "By", "Bz", "Hx", "Dx", "Z_rot", "rBx", "rBy"]:
                ax.set_ylabel("Magnetic Field (nT)", fontsize=10)
            elif col in ["Ex", "Ey", "Ex_rot", "Ey_rot"]:
                ax.set_ylabel("Electric Field (mV/km)", fontsize=10)
            else:
                ax.set_ylabel(col, fontsize=10)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=8)
            
            # Format x-axis for datetime
            if "datetime" in df.columns:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Set x-axis limits to show full data width with no blank spots
            if len(x_data) > 0:
                ax.set_xlim(x_data.min(), x_data.max())
                
                # Add visual indicators for trimmed data regions if skip_minutes was used
                if skip_minutes[0] > 0 or skip_minutes[1] > 0:
                    # Calculate the original data range that was trimmed
                    if "datetime" in df.columns:
                        # For datetime data, we need to estimate the trimmed regions
                        # Since we don't have the original data, we'll show the current range
                        # and add dashed lines at the boundaries
                        current_start = x_data.min()
                        current_end = x_data.max()
                        
                        # Add dashed lines to indicate trimmed regions
                        if skip_minutes[0] > 0:
                            # Add dashed line at the start to indicate data was trimmed
                            ax.axvline(current_start, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                                     label=f'Start (trimmed {skip_minutes[0]} min)')
                        
                        if skip_minutes[1] > 0:
                            # Add dashed line at the end to indicate data was trimmed
                            ax.axvline(current_end, color='red', linestyle='--', alpha=0.7, linewidth=2,
                                     label=f'End (trimmed {skip_minutes[1]} min)')
                    else:
                        # For time-based data, we can be more precise
                        if skip_minutes[0] > 0:
                            skip_seconds_start = skip_minutes[0] * 60
                            ax.axvline(skip_seconds_start, color='red', linestyle='--', alpha=0.7, linewidth=2,
                                     label=f'Start (trimmed {skip_minutes[0]} min)')
                        
                        if skip_minutes[1] > 0:
                            skip_seconds_end = x_data.max() - (skip_minutes[1] * 60)
                            ax.axvline(skip_seconds_end, color='red', linestyle='--', alpha=0.7, linewidth=2,
                                     label=f'End (trimmed {skip_minutes[1]} min)')
    
    # Set x-axis label on the last subplot only
    if len(axes) > 0:
        axes[-1].set_xlabel(x_label, fontsize=12)
    
    # Adjust layout to minimize white space
    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3, rect=(0, 0, 1, 0.95))
    
    # Add year annotation inline with the Time axis title
    if "datetime" in df.columns and len(df) > 0:
        last_ax = axes[-1]
        last_year = df["datetime"].iloc[-1].year
        # Position the year annotation at the same y-level as the x-axis label
        xlabel_pos = last_ax.get_xlabel()
        if xlabel_pos:
            # Get the position of the x-axis label and place year annotation nearby
            last_ax.annotate(f"({last_year})", xy=(0.98, -0.15), xycoords="axes fraction", 
                            fontsize=10, ha="right", va="top", color="gray",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    if save_plots:
        filename = os.path.join(output_dir, f"{site_name}_physical_channels.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        write_log(f"Physical channels plot saved as {filename}")
    else:
        plt.show()


def extract_datetime_from_filename(filename):
    """Extracts datetime from EDL filename format.
    
    Filename format: EDL_YYMMDDHHMMSS.CHANNEL
    Example: EDL_041020190000.BX -> 2004-10-20 19:00:00
    
    Args:
        filename (str): EDL filename
        
    Returns:
        datetime.datetime: Extracted datetime object
    """
    try:
        # Extract the timestamp part after underscore and before dot
        timestamp_part = filename.split('_')[1].split('.')[0]
        
        # Parse YYMMDDHHMMSS format
        year = 2000 + int(timestamp_part[0:2])  # Assume 20xx for YY
        month = int(timestamp_part[2:4])
        day = int(timestamp_part[4:6])
        hour = int(timestamp_part[6:8])
        minute = int(timestamp_part[8:10])
        second = int(timestamp_part[10:12])
        
        return datetime.datetime(year, month, day, hour, minute, second)
    except (IndexError, ValueError) as e:
        write_log(f"Error parsing datetime from filename {filename}: {e}", level="ERROR")
        return None


def extract_gps_coordinates(gps_file_path):
    """Extracts GPS coordinates from EDL GPS file.
    
    GPS Message Format:
    AL (Altitude) message: >RAL[GPS_time][Altitude][Vertical_Velocity][Source][Age];*[checksum]<
    - GPS_time: 5 digits (e.g., 08057)
    - Altitude: 6 digits with sign (e.g., +00083)
    - Vertical_Velocity: 5 digits with sign (e.g., +000)
    - Source: 1 digit (1 = 3D GPS)
    - Age: 1 digit (2 = Fresh)
    
    PV (Position Velocity) message: >RPV[GPS_time][Latitude][Longitude][Speed][Heading][Source][Age];*[checksum]<
    - GPS_time: 5 digits (e.g., 08057)
    - Latitude: 8 digits with sign (e.g., -3497172) - divide by 100000 for decimal degrees
    - Longitude: 9 digits with sign (e.g., +13948378) - divide by 100000 for decimal degrees
    - Speed: 3 digits (e.g., 000)
    - Heading: 3 digits (e.g., 000)
    - Source: 1 digit (1 = 3D GPS)
    - Age: 1 digit (2 = Fresh)
    
    Args:
        gps_file_path (str): Path to the GPS file
        
    Returns:
        dict: Dictionary containing latitude, longitude, and altitude
    """
    gps_data = {}
    try:
        with open(gps_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('>R'):
                continue
            # Remove leading '>' and trailing '<' if present
            if line.startswith('>'):
                line = line[1:]
            if line.endswith('<'):
                line = line[:-1]
            # Split at ';' to remove checksum
            line = line.split(';')[0]
            
            # AL message: >RAL[GPS_time][Altitude][Vertical_Velocity][Source][Age]
            if line.startswith('RAL'):
                try:
                    # Extract altitude: positions 8-13 (after RAL + 5-digit GPS time)
                    # Format: RAL00017+00085+00012
                    #        01234567890123456789
                    altitude_str = line[8:14]  # +00085 (after RAL + 5 digits)
                    gps_data['altitude'] = float(altitude_str)
                    write_log(f"Extracted altitude: {gps_data['altitude']} m")
                except Exception as e:
                    write_log(f"Error parsing altitude from line '{line}': {e}", level="WARNING")
            
            # PV message: >RPV[GPS_time][Latitude][Longitude][Speed][Heading][Source][Age]
            elif line.startswith('RPV'):
                try:
                    # Extract latitude: positions 8-15 (after RPV + 5-digit GPS time)
                    # Format: RPV00017-3497176+1394837700000012
                    #        012345678901234567890123456789
                    lat_str = line[8:16]   # -3497176 (after RPV + 5 digits)
                    # Extract longitude: positions 16-25 (9 digits)
                    lon_str = line[16:25]  # +13948377 (9 digits)
                    
                    gps_data['latitude'] = float(lat_str) / 100000.0
                    gps_data['longitude'] = float(lon_str) / 100000.0
                    write_log(f"Extracted coordinates: Lat={gps_data['latitude']:.6f}°, Lon={gps_data['longitude']:.6f}°")
                except Exception as e:
                    write_log(f"Error parsing coordinates from line '{line}': {e}", level="WARNING")
        
        return gps_data
    except FileNotFoundError:
        write_log(f"GPS file not found: {gps_file_path}", level="WARNING")
        return gps_data
    except Exception as e:
        write_log(f"Error reading GPS file {gps_file_path}: {e}", level="ERROR")
        return gps_data


def find_gps_file(input_dir, station_identifier=None):
    """Finds the first GPS file in the input directory or its subdirectories.
    
    Uses the station_long_identifier from recorder.ini to find GPS files with the correct naming pattern.
    GPS files follow the pattern: {station_identifier}YYMMDDHHMMSS.gps
    
    Args:
        input_dir (str): Input directory to search
        station_identifier (str): Station identifier from recorder.ini (e.g., 'EDL_', 'EDL03_', 'EDL04_')
        
    Returns:
        str: Path to the first GPS file found, or None if not found
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.gps'):
                # If we have the station identifier, use it to find the correct GPS file
                if station_identifier and file.startswith(station_identifier):
                    write_log(f"Found GPS file with station identifier '{station_identifier}': {file}")
                    return os.path.join(root, file)
                # Fallback: check if the file has a timestamp pattern (YYMMDDHHMMSS) after an underscore
                elif '_' in file:
                    parts = file.split('_')
                    if len(parts) >= 2:
                        timestamp_part = parts[-1].replace('.gps', '')
                        if len(timestamp_part) == 12 and timestamp_part.isdigit():
                            write_log(f"Found GPS file with timestamp pattern: {file}")
                            return os.path.join(root, file)
                elif 'EDL_' in file:
                    write_log(f"Found GPS file with EDL_ pattern: {file}")
                    return os.path.join(root, file)
    
    # If no GPS file found with specific patterns, log the expected pattern
    if station_identifier:
        write_log(f"No GPS file found with expected pattern: {station_identifier}*.gps in {input_dir}", level="WARNING")
    
    # Fallback: look for any .gps file
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.gps'):
                write_log(f"Found GPS file (fallback): {file}")
                return os.path.join(root, file)
    
    # If still not found, log a warning
    write_log(f"No GPS file found in {input_dir}", level="WARNING")
    return None


def calculate_distance_between_sites(gps1, gps2):
    """Calculate the distance between two GPS coordinates using the Haversine formula.
    
    Args:
        gps1 (dict): GPS data for first site with 'latitude' and 'longitude' keys
        gps2 (dict): GPS data for second site with 'latitude' and 'longitude' keys
        
    Returns:
        float: Distance in kilometers
    """
    try:
        lat1 = gps1.get('latitude')
        lon1 = gps1.get('longitude')
        lat2 = gps2.get('latitude')
        lon2 = gps2.get('longitude')
        
        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            return None
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
        
    except Exception as e:
        write_log(f"Error calculating distance between sites: {e}", level="WARNING")
        return None


def load_remote_reference_data(remote_site, remote_site_dir, start_datetime, end_datetime, sample_interval, main_site_gps=None, timezone="UTC"):
    """Loads remote reference data (Bx, By) from the specified remote site.
    
    Args:
        remote_site (str): Name of the remote reference site
        remote_site_dir (str): Directory containing the remote site data
        start_datetime (datetime): Start datetime for the main site
        end_datetime (datetime): End datetime for the main site
        sample_interval (float): Sample interval in seconds
        main_site_gps (dict, optional): GPS data from main site for distance calculation
        timezone (str): Timezone for the remote reference data
        
    Returns:
        tuple: (DataFrame with datetime, rBx, rBy columns, remote GPS data dict, distance float)
    """
    try:
        write_log(f"Loading remote reference data from {remote_site}")
        
        # Read the recorder.ini file from the remote site to get the correct station identifier
        remote_recorder_ini = os.path.join(remote_site_dir, "config", "recorder.ini")
        station_identifier = "EDL_"  # Default fallback
        
        if os.path.exists(remote_recorder_ini):
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(remote_recorder_ini)
                
                # Check both [recorder] and [Station] sections for station_long_identifier
                if 'recorder' in config and 'station_long_identifier' in config['recorder']:
                    station_identifier = config['recorder']['station_long_identifier']
                    write_log(f"Read station identifier '{station_identifier}' from {remote_recorder_ini} [recorder] section")
                elif 'Station' in config and 'station_long_identifier' in config['Station']:
                    station_identifier = config['Station']['station_long_identifier']
                    write_log(f"Read station identifier '{station_identifier}' from {remote_recorder_ini} [Station] section")
                else:
                    write_log(f"station_long_identifier not found in {remote_recorder_ini}, using default 'EDL_'", level="WARNING")
            except Exception as e:
                write_log(f"Error reading {remote_recorder_ini}: {e}, using default 'EDL_'", level="WARNING")
        else:
            write_log(f"recorder.ini not found at {remote_recorder_ini}, using default 'EDL_'", level="WARNING")
        
        write_log(f"Using station identifier '{station_identifier}' for remote site {remote_site}")
        
        # Load remote reference GPS data
        remote_gps_data = None
        distance_km = None
        
        try:
            remote_gps_file_path = find_gps_file(remote_site_dir, station_identifier)
            if remote_gps_file_path:
                write_log(f"Found remote GPS file: {remote_gps_file_path}")
                remote_gps_data = extract_gps_coordinates(remote_gps_file_path)
                if remote_gps_data and main_site_gps:
                    # Calculate distance between sites
                    distance_km = calculate_distance_between_sites(main_site_gps, remote_gps_data)
                    write_log(f"Distance between {remote_site} and main site: {distance_km:.2f} km")
                elif remote_gps_data:
                    write_log(f"Remote GPS data loaded for {remote_site}, but no main site GPS for distance calculation")
            else:
                write_log(f"No GPS file found for remote site {remote_site}")
        except Exception as e:
            write_log(f"Error loading remote GPS data: {e}", level="WARNING")
        
        # Use ASCIIReader to read only BX and BY channels from remote site
        dummy_outfile = io.StringIO()
        remote_metadata = {
            'sample_interval': sample_interval,
            'station_long_identifier': station_identifier
        }
        
        data_reader = ASCIIReader(remote_site_dir, remote_metadata, average=False, log_first_rows=False)
        # Load only BX and BY channels for remote reference
        remote_df = data_reader.read_specific_channels(['BX', 'BY'], dummy_outfile)
        
        if remote_df.empty:
            write_log(f"No remote reference data found for {remote_site}", level="WARNING")
            return None, None, None
        
        # Convert to physical units (we only need Bx, By)
        divisor = 10**7
        remote_df["rBx"] = (remote_df["BX"].astype(float) / divisor) * 70000.0
        remote_df["rBy"] = -(remote_df["BY"].astype(float) / divisor) * 70000.0
        
        # Build datetime column for remote data
        # Find the first BX file to extract start time using the same logic as ASCIIReader
        pattern = os.path.join(remote_site_dir, "**", f"{station_identifier}*.BX")
        bx_files = glob.glob(pattern, recursive=True)
        
        if not bx_files:
            write_log(f"No BX files found in remote site {remote_site} using pattern '{station_identifier}*.BX'", level="WARNING")
            return None, None, None
        
        # Sort files to get the earliest one
        bx_files.sort()
        first_bx_file = os.path.basename(bx_files[0])
        write_log(f"Remote reference using first file: {first_bx_file}")
        
        remote_start_datetime = extract_datetime_from_filename(first_bx_file)
        if remote_start_datetime:
            # Apply GPS week rollover correction if needed (same logic as main site)
            original_time = remote_start_datetime
            if remote_start_datetime.year < 2020:  # Likely GPS week rollover issue
                gps_week_rollover = datetime.timedelta(weeks=1024)  # 19.7 years
                remote_start_datetime = remote_start_datetime + gps_week_rollover
                write_log(f"Remote GPS week rollover correction applied:")
                write_log(f"  Original (wrong): {original_time}")
                write_log(f"  Corrected (actual): {remote_start_datetime}")
                write_log(f"  Correction: +{gps_week_rollover.days} days ({gps_week_rollover.days * 24} hours)")
            
            time_seconds = np.arange(len(remote_df)) * sample_interval
            remote_df["datetime"] = [remote_start_datetime + datetime.timedelta(seconds=s) for s in time_seconds]
            
            # Apply timezone conversion if requested
            if timezone != "UTC":
                try:
                    import pytz
                    utc_tz = pytz.UTC
                    target_tz = pytz.timezone(timezone)
                    remote_df["datetime"] = remote_df["datetime"].dt.tz_localize(utc_tz).dt.tz_convert(target_tz)
                    write_log(f"Applied timezone '{timezone}' to remote reference data")
                except ImportError:
                    write_log(f"pytz not available, using UTC timezone for remote reference", level="WARNING")
                except Exception as e:
                    write_log(f"Error converting remote reference timezone: {e}, using UTC", level="WARNING")
            
            write_log(f"Remote reference data loaded: {len(remote_df)} samples from {remote_start_datetime} to {remote_df['datetime'].iloc[-1]}")
            write_log(f"Main site time range: {start_datetime} to {end_datetime}")
            write_log(f"Remote site time range: {remote_df['datetime'].min()} to {remote_df['datetime'].max()}")
            
            # For remote reference, we don't need to filter to exact time range
            # The merge function will handle alignment with tolerance
            # Just keep the columns we need
            result_df = remote_df[['datetime', 'rBx', 'rBy']].copy()
            write_log(f"Remote reference data prepared: {len(result_df)} samples")
            
            return result_df, remote_gps_data, distance_km
        else:
            write_log(f"Could not extract start time from remote site {remote_site} file {first_bx_file}", level="WARNING")
            return None, None, None
            
    except Exception as e:
        write_log(f"Error loading remote reference data from {remote_site}: {e}", level="ERROR")
        return None, None, None


def merge_with_remote_reference(main_df, remote_df, tolerance_seconds=1.0):
    """Merges main site data with remote reference data based on datetime matching.
    
    Args:
        main_df (pd.DataFrame): Main site data with datetime column
        remote_df (pd.DataFrame): Remote reference data with datetime column
        tolerance_seconds (float): Time tolerance for matching in seconds
        
    Returns:
        pd.DataFrame: Merged dataframe with 7 columns (Bx, By, Bz, Ex, Ey, rBx, rBy)
    """
    try:
        write_log("Merging main site data with remote reference data")
        
        if remote_df is None or remote_df.empty:
            write_log("No remote reference data available, returning main site data only", level="WARNING")
            return main_df
        
        # Ensure both dataframes have datetime as index for merging
        main_copy = main_df.copy()
        remote_copy = remote_df.copy()
        
        # Set datetime as index for both dataframes
        main_copy.set_index('datetime', inplace=True)
        remote_copy.set_index('datetime', inplace=True)
        
        # Use merge_asof if available, otherwise use merge with tolerance
        try:
            # Try merge_asof (pandas >= 0.19.0)
            merged_df = main_copy.merge_asof(
                remote_copy, 
                left_index=True, 
                right_index=True,
                tolerance=pd.Timedelta(seconds=tolerance_seconds),
                direction='nearest'
            )
        except AttributeError:
            # Fallback for older pandas versions - use merge with tolerance
            write_log("merge_asof not available, using standard merge with tolerance", level="WARNING")
            
            # Reset index to get datetime as column for standard merge
            main_copy.reset_index(inplace=True)
            remote_copy.reset_index(inplace=True)
            
            # Merge on datetime with tolerance
            merged_df = pd.merge_asof(
                main_copy, 
                remote_copy, 
                on='datetime',
                tolerance=None,
                direction='nearest'
            )
            
            # Set datetime back as index
            merged_df.set_index('datetime', inplace=True)
        
        # Reset index to get datetime back as a column
        merged_df.reset_index(inplace=True)
        
        # Check for missing remote reference data
        missing_remote = merged_df['rBx'].isnull().sum()
        total_samples = len(merged_df)
        
        if missing_remote > 0:
            write_log(f"WARNING: {missing_remote}/{total_samples} samples missing remote reference data", level="WARNING")
        
        write_log(f"Successfully merged data: {len(merged_df)} samples with remote reference")
        write_log(f"Final columns: {list(merged_df.columns)}")
        
        return merged_df
        
    except Exception as e:
        write_log(f"Error merging with remote reference: {e}", level="ERROR")
        return main_df


class ProcessASCII:
    """Processes ASCII EDL magnetotelluric data files for a given site.
    
    Reads metadata, loads ASCII files (.BX, .BY, .BZ, .EX, .EY) from day folders,
    concatenates raw data, applies corrections (drift, rotation, tilt), optional smoothing,
    generates plots (or saves them), and saves raw and processed outputs.
    Output files and plots are named using the site name (derived from the input directory).
    """
    
    def __init__(self, input_dir, param_file, average=False, perform_freq_analysis=False,
                 plot_timeseries=False, apply_smoothing=False, smoothing_window=2500, threshold_factor=10.0,
                 plot_boundaries=True, plot_smoothed_windows=True, plot_coherence=False,
                 log_first_rows=False, smoothing_method="median", sens_start=0, sens_end=5000,
                 skip_minutes=[0, 0], apply_drift_correction=False, apply_rotation=False,
                 plot_drift=False, plot_rotation=False, plot_tilt=False, timezone="UTC", plot_original_data=False,
                 save_raw_data=False, save_processed_data=False, remote_reference=None,
                 apply_filtering=False, filter_type="comb", filter_channels=None, filter_params=None,
                 plot_heatmaps=False, heatmap_nperseg=1024, heatmap_noverlap=None, heatmap_thresholds=None):
        """
        Initialize the ProcessASCII class.
        
        Args:
            input_dir (str): Directory containing the ASCII data files.
            param_file (str): Path to the parameter file (config/recorder.ini).
            average (bool): Whether to average the data (unused).
            perform_freq_analysis (bool): Whether to perform frequency analysis.
            plot_timeseries (bool): Whether to plot the data.
            apply_smoothing (bool): Whether to apply smoothing.
            smoothing_window (int): Window size for smoothing.
            threshold_factor (float): Threshold factor for outlier detection.
            plot_boundaries (bool): Whether to plot file boundaries.
            plot_smoothed_windows (bool): Whether to plot smoothed windows.
            plot_coherence (bool): Whether to plot coherence.
            log_first_rows (bool): Whether to log the first few rows.
            smoothing_method (str): Smoothing method ("median" or "adaptive").
            sens_start (int): Start sample for sensitivity test (unused).
            sens_end (int): End sample for sensitivity test (unused).
            skip_minutes (list): Minutes to skip from start and end of data (e.g., [30, 20] skips first 30 and last 20 minutes).
            apply_drift_correction (bool): Whether to apply drift correction.
            apply_rotation (bool): Whether to apply rotation.
            plot_drift (bool): Whether to plot drift correction.
            plot_rotation (bool): Whether to plot rotation.
            plot_tilt (bool): Whether to plot tilt correction.
            timezone (str): Timezone for the data.
            plot_original_data (bool): Whether to plot original data.
            save_raw_data (bool): Whether to save raw data.
            save_processed_data (bool): Whether to save processed data.
            remote_reference (str, optional): Remote reference site name.
            apply_filtering (bool): Whether to apply frequency filtering.
            filter_type (str): Type of filter ("comb", "bandpass", "highpass", "lowpass", "adaptive", "custom").
            filter_channels (list, optional): Channels to filter. If None, filters all main channels.
            filter_params (dict, optional): Additional filter parameters.
            plot_heatmaps (bool): Whether to generate heatmap plots for quality control.
            heatmap_nperseg (int): Number of points per segment for heatmap FFT.
            heatmap_noverlap (int, optional): Number of points to overlap between heatmap segments.
            heatmap_thresholds (dict, optional): Custom coherence thresholds for heatmap quality scoring.
        """
        self.input_dir = input_dir
        self.param_file = param_file
        self.average = average
        self.perform_freq_analysis = perform_freq_analysis
        self.plot_timeseries = plot_timeseries
        self.apply_smoothing = apply_smoothing
        self.smoothing_window = smoothing_window
        self.threshold_factor = threshold_factor
        self.plot_boundaries = plot_boundaries
        self.plot_smoothed_windows = plot_smoothed_windows
        self.plot_coherence = plot_coherence
        self.log_first_rows = log_first_rows
        self.smoothing_method = smoothing_method
        self.sens_start = sens_start
        self.sens_end = sens_end
        self.skip_minutes = skip_minutes
        self.apply_drift_correction = apply_drift_correction
        self.apply_rotation = apply_rotation
        self.plot_drift = plot_drift
        self.plot_rotation = plot_rotation
        self.plot_tilt = plot_tilt
        self.timezone = timezone
        self.plot_original_data = plot_original_data
        self.save_raw_data = save_raw_data
        self.save_processed_data = save_processed_data
        self.remote_reference = remote_reference
        self.apply_filtering = apply_filtering
        self.filter_type = filter_type
        self.filter_channels = filter_channels or ["Bx", "By", "Bz", "Ex", "Ey"]
        self.filter_params = filter_params or {}
        self.plot_heatmaps = plot_heatmaps
        self.heatmap_nperseg = heatmap_nperseg
        self.heatmap_noverlap = heatmap_noverlap
        self.heatmap_thresholds = heatmap_thresholds
        
        # Extract site name from input directory
        self.site_name = os.path.basename(os.path.abspath(input_dir))
        set_site_name(self.site_name)
        
        # Create output directory
        self.output_dir = os.path.join("outputs", self.site_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results tracking
        self.results = {
            'status': 'INITIALIZED',
            'data_shape': None,
            'error': None,
            'start_time': datetime.datetime.now(),
            'end_time': None
        }
        
        # Initialize metadata collection for summary table
        self.summary_metadata = {
            'site_name': self.site_name,
            'xarm': None,
            'yarm': None,
            'remote_reference': remote_reference,
            'sample_rate': None,
            'active_channels': [],
            'latitude': None,
            'longitude': None,
            'elevation': None,
            'start_time': None,
            'end_time': None,
            'timezone': timezone,
            'duration': None,
            'total_files': 0,
            'data_points': 0,
            'data_shape': None,
            'channel_means': {},
            'field_strengths': {},
            'corrections_applied': [],
            'tilt_angles': {}
        }
        
        # Read Processing.txt configuration
        self.processing_config = read_processing_config()
        
        # Read site metadata
        self.metadata = self.read_metadata()
        
        # Override dipole lengths with Processing.txt values if available
        if self.site_name in self.processing_config:
            site_config = self.processing_config[self.site_name]
            self.metadata['xarm'] = site_config['xarm']
            self.metadata['yarm'] = site_config['yarm']
            # Update summary metadata
            self.summary_metadata['xarm'] = site_config['xarm']
            self.summary_metadata['yarm'] = site_config['yarm']
        else:
            # Use default metadata
            self.summary_metadata['xarm'] = self.metadata.get('xarm', 100.0)
            self.summary_metadata['yarm'] = self.metadata.get('yarm', 100.0)
        
        self.tilt_correction = False  # Set via command-line.
        self.save_plots = False   # Set via command-line.
        self.decimation_factor = 1  # Default to no decimation
        self.run_lemimt = False  # Set via command-line
        self.lemimt_path = "lemimt.exe"  # Set via command-line

    def read_metadata(self):
        """Reads metadata from the recorder.ini file in the input directory.
        
        Returns:
            dict: Metadata dictionary.
        """
        metadata = {}
        try:
            config_file_path = os.path.join(self.input_dir, self.param_file)
            config = configparser.ConfigParser()
            config.read(config_file_path)
            
            # Extract basic station information
            recorder_section = config['recorder']
            metadata['station_long_identifier'] = recorder_section.get('station_long_identifier', 'EDL_')
            metadata['station_short_identifier'] = recorder_section.get('station_short_identifier', 'edl_')
            
            # Extract channel information
            channels = {}
            active_channels = []
            for i in range(22):  # Check for channels 0-21
                long_id_key = f'channel_{i}_long_id'
                short_id_key = f'channel_{i}_short_id'
                samplerate_key = f'channel_{i}_samplerate'
                format_key = f'channel_{i}_format'
                
                if long_id_key in recorder_section:
                    channels[i] = {
                        'long_id': recorder_section[long_id_key],
                        'short_id': recorder_section.get(short_id_key, ''),
                        'samplerate': int(recorder_section.get(samplerate_key, 0)),
                        'format': recorder_section.get(format_key, 'ascii')
                    }
                    if channels[i]['samplerate'] > 0:
                        active_channels.append(recorder_section[long_id_key])
            
            metadata['channels'] = channels
            
            # Get sample rate for the main channels (BX, BY, BZ, EX, EY)
            main_channels = ['BX', 'BY', 'BZ', 'EX', 'EY']
            sample_rates = []
            for channel_info in channels.values():
                if channel_info['long_id'] in main_channels and channel_info['samplerate'] > 0:
                    sample_rates.append(channel_info['samplerate'])
            
            if sample_rates:
                metadata['sample_interval'] = 1.0 / sample_rates[0]  # Use first non-zero sample rate
                self.summary_metadata['sample_rate'] = sample_rates[0]
            else:
                metadata['sample_interval'] = 0.1  # Default to 10 Hz if not found
                self.summary_metadata['sample_rate'] = 10.0
            
            # Update summary metadata
            self.summary_metadata['active_channels'] = active_channels
            
            # Set default values for processing parameters
            metadata['xarm'] = 100.0  # Default dipole length in meters
            metadata['yarm'] = 100.0  # Default dipole length in meters
            metadata['erotate'] = 1   # Default to enable rotation
            metadata['time_drift'] = 0  # Default no time drift
            
            # Set default start and finish times (will be updated based on actual data)
            current_time = datetime.datetime.now()
            metadata['start_time'] = {
                'day': current_time.day,
                'month': current_time.month, 
                'year': current_time.year,
                'hour': current_time.hour,
                'minute': current_time.minute,
                'second': current_time.second
            }
            metadata['finish_time'] = metadata['start_time'].copy()
            
        except Exception as e:
            write_log(f"[{self.site_name}] Error reading metadata file {self.param_file} from {self.input_dir}: {e}", level="ERROR")
            # Set minimal default metadata
            metadata = {
                'sample_interval': 0.1,
                'xarm': 100.0,
                'yarm': 100.0,
                'erotate': 1,
                'time_drift': 0,
                'channels': {}
            }
            self.summary_metadata['sample_rate'] = 10.0
        return metadata

    def process_all_files(self):
        """Process all ASCII files in the input directory."""
        try:
            # Print processing header
            print_processing_header("MT PROCESSING")
            
            self.results['status'] = 'PROCESSING'
            write_site_log(f"Processing site: {self.site_name}")
            
            # Extract GPS coordinates
            gps_file_path = find_gps_file(self.input_dir, self.site_name)
            self.gps_data = None  # Store as instance attribute
            if gps_file_path:
                self.gps_data = extract_gps_coordinates(gps_file_path)
                if self.gps_data:
                    # Update summary metadata
                    self.summary_metadata['latitude'] = self.gps_data.get('latitude')
                    self.summary_metadata['longitude'] = self.gps_data.get('longitude')
                    self.summary_metadata['elevation'] = self.gps_data.get('altitude')
            
            # Load ASCII data
            reader = ASCIIReader(self.input_dir, self.metadata, self.average, self.log_first_rows)
            combined_raw_df = reader.read_all_data(io.StringIO())
            
            if combined_raw_df is None or combined_raw_df.empty:
                write_site_log("No ASCII data files found to process.", level="ERROR")
                self.results['status'] = 'FAILED'
                self.results['error'] = 'No data files found'
                return
            
            # Update summary metadata
            self.summary_metadata['data_shape'] = combined_raw_df.shape
            self.summary_metadata['data_points'] = len(combined_raw_df)
            
            # Count total files
            total_files = 0
            for channel in ['BX', 'BY', 'BZ', 'EX', 'EY']:
                pattern = os.path.join(self.input_dir, "**", f"*.{channel}")
                files = glob.glob(pattern, recursive=True)
                total_files += len(files)
            self.summary_metadata['total_files'] = total_files
            
            write_site_log(f"Loaded {total_files} files, {len(combined_raw_df):,} data points")
            
            # Save raw data if requested
            if self.save_raw_data:
                raw_output_path = os.path.join(self.output_dir, f"{self.site_name}_output_raw.txt")
                combined_raw_df.to_csv(raw_output_path, index=False, sep='\t')
                write_site_log(f"Raw data saved to {raw_output_path}")
            else:
                write_site_log(f"Skipping saving raw data (not requested)")
            
            # Build time column
            sample_interval = self.metadata.get('sample_interval', 1.0)
            
            # Determine timezone handling
            if isinstance(self.timezone, list):
                main_timezone = self.timezone[0]
            else:
                main_timezone = self.timezone
            
            # Find BX files to extract start time
            station_identifier = self.metadata.get('station_long_identifier', 'EDL_')
            bx_files = glob.glob(os.path.join(self.input_dir, "**", f"{station_identifier}*.BX"), recursive=True)
            if not bx_files:
                bx_files = glob.glob(os.path.join(self.input_dir, "**", "*.BX"), recursive=True)
            
            if bx_files:
                first_file = sorted(bx_files)[0]
                
                # Extract start time from filename
                start_datetime = extract_datetime_from_filename(os.path.basename(first_file))
                
                if start_datetime:
                    # Apply GPS week rollover correction if needed
                    original_time = start_datetime
                    if start_datetime.year < 2020:  # Likely GPS week rollover issue
                        gps_week_rollover = datetime.timedelta(weeks=1024)  # 19.7 years
                        start_datetime = start_datetime + gps_week_rollover
                        write_site_log(f"Applied GPS week rollover correction: {original_time} → {start_datetime}")
                    
                    # Convert timezone if requested
                    if main_timezone != "UTC":
                        try:
                            import pytz
                            utc_tz = pytz.UTC
                            target_tz = pytz.timezone(main_timezone)
                            start_datetime = utc_tz.localize(start_datetime).astimezone(target_tz)
                            write_site_log(f"Converted start time to {main_timezone}: {start_datetime}")
                        except ImportError:
                            write_site_log(f"pytz not available, using UTC timezone", level="WARNING")
                        except Exception as e:
                            write_site_log(f"Error converting timezone: {e}, using UTC", level="WARNING")
                    
                    # Update summary metadata
                    # Determine timezone handling
                    if isinstance(self.timezone, list):
                        main_timezone = self.timezone[0]
                    else:
                        main_timezone = self.timezone
                    
                    # Build datetime column
                    combined_raw_df['datetime'] = [
                        start_datetime + datetime.timedelta(seconds=i * sample_interval)
                        for i in range(len(combined_raw_df))
                    ]
                    
                    # Update summary metadata
                    self.summary_metadata['start_time'] = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    self.summary_metadata['end_time'] = combined_raw_df['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
                    self.summary_metadata['duration'] = str(combined_raw_df['datetime'].iloc[-1] - start_datetime)
                    
                    write_site_log(f"Time range: {self.summary_metadata['start_time']} to {self.summary_metadata['end_time']}")
                else:
                    write_site_log(f"Failed to extract datetime from filename, using simple time column", level="WARNING")
                    combined_raw_df['datetime'] = [
                        datetime.datetime.now() + datetime.timedelta(seconds=i * sample_interval)
                        for i in range(len(combined_raw_df))
                    ]
            else:
                write_site_log(f"No BX files found, using simple time column", level="WARNING")
                combined_raw_df['datetime'] = [
                    datetime.datetime.now() + datetime.timedelta(seconds=i * sample_interval)
                    for i in range(len(combined_raw_df))
                ]
            
            # Skip minutes if requested (after datetime column is built)
            if self.skip_minutes[0] > 0:
                skip_timedelta = datetime.timedelta(minutes=self.skip_minutes[0])
                original_start = combined_raw_df['datetime'].min()
                new_start = original_start + skip_timedelta
                combined_raw_df = combined_raw_df[combined_raw_df['datetime'] >= new_start]
                write_site_log(f"Skipped first {self.skip_minutes[0]} minutes")
            
            if self.skip_minutes[1] > 0:
                skip_timedelta = datetime.timedelta(minutes=self.skip_minutes[1])
                original_end = combined_raw_df['datetime'].max()
                new_end = original_end - skip_timedelta
                combined_raw_df = combined_raw_df[combined_raw_df['datetime'] <= new_end]
                write_site_log(f"Skipped last {self.skip_minutes[1]} minutes")
            
            # Convert to physical units
            write_site_log(f"Converting to physical units...")
            processed_df = self.convert_ascii_to_physical_units(combined_raw_df)
            
            # Load remote reference data if requested
            if self.remote_reference:
                write_site_log(f"Loading remote reference: {self.remote_reference}")
                remote_ref_site = self.remote_reference
                remote_ref_dir = os.path.join(os.path.dirname(self.input_dir), remote_ref_site)
                
                # Determine timezone for remote reference
                if isinstance(self.timezone, list) and len(self.timezone) > 1:
                    remote_timezone = self.timezone[1]
                else:
                    remote_timezone = main_timezone
                write_site_log(f"Using timezone '{remote_timezone}' for remote reference site")
                
                if os.path.exists(remote_ref_dir):
                    remote_df, remote_gps_data, distance_km = load_remote_reference_data(
                        remote_ref_site, remote_ref_dir, 
                        processed_df['datetime'].min(), processed_df['datetime'].max(),
                        sample_interval, self.gps_data, remote_timezone
                    )
                    
                    if remote_df is not None and not remote_df.empty:
                        # Ensure both datasets have the same timezone before merging
                        if isinstance(self.timezone, list) and len(self.timezone) > 1 and self.timezone[0] != self.timezone[1]:
                            # Convert remote reference to main site timezone
                            try:
                                import pytz
                                remote_tz = pytz.timezone(remote_timezone)
                                main_tz = pytz.timezone(main_timezone)
                                remote_df['datetime'] = remote_df['datetime'].dt.tz_localize(remote_tz).dt.tz_convert(main_tz)
                                write_site_log(f"Converted remote reference timezone from {remote_timezone} to {main_timezone}")
                            except Exception as e:
                                write_site_log(f"Error converting remote reference timezone: {e}", level="WARNING")
                        
                        processed_df = merge_with_remote_reference(processed_df, remote_df)
                        if 'rBx' in processed_df.columns:
                            write_site_log(f"Remote reference merged successfully")
                        else:
                            write_site_log(f"Remote reference merge failed", level="WARNING")
                    else:
                        write_site_log(f"No remote reference data available", level="WARNING")
                else:
                    write_site_log(f"Remote reference directory not found: {remote_ref_dir}", level="WARNING")
            
            # Apply decimation if requested
            if hasattr(self, 'decimation_factor') and self.decimation_factor > 1:
                write_site_log(f"Applying {self.decimation_factor}x decimation...")
                processed_df, sample_interval = decimate_dataframe(processed_df, self.decimation_factor, sample_interval)
                self.summary_metadata['sample_rate'] = 1.0 / sample_interval
            
            # Preserve original columns before tilt correction if plotting tilt
            if self.plot_tilt:
                if 'Bx' in processed_df.columns:
                    processed_df['Bx_original'] = processed_df['Bx'].copy()
                if 'By' in processed_df.columns:
                    processed_df['By_original'] = processed_df['By'].copy()
                if 'rBx' in processed_df.columns:
                    processed_df['rBx_original'] = processed_df['rBx'].copy()
                if 'rBy' in processed_df.columns:
                    processed_df['rBy_original'] = processed_df['rBy'].copy()

            # Apply tilt correction
            tilt_angle_degrees = None
            remote_tilt_angle = None
            if self.tilt_correction:
                write_site_log(f"Applying tilt correction...")
                include_remote = self.tilt_correction == "RR" and 'rBx' in processed_df.columns and 'rBy' in processed_df.columns
                processed_df, tilt_angle_degrees = tilt_correction(processed_df, include_remote_reference=include_remote)
                
                # Update summary metadata
                self.summary_metadata['corrections_applied'].append("Tilt correction")
                self.summary_metadata['tilt_angles']['Main site'] = tilt_angle_degrees
                
                # Extract remote tilt angle if available
                if include_remote and 'rBx' in processed_df.columns and 'rBy' in processed_df.columns:
                    remote_tilt_angle = np.degrees(np.arctan2(processed_df['rBy'].mean(), processed_df['rBx'].mean()))
                    self.summary_metadata['tilt_angles']['Remote reference'] = remote_tilt_angle
            
            # Apply smoothing if requested
            if self.apply_smoothing:
                write_site_log(f"Applying {self.smoothing_method} smoothing...")
                if self.smoothing_method == "median":
                    processed_df, _ = smooth_median_mad(processed_df, window=self.smoothing_window, threshold=self.threshold_factor)
                elif self.smoothing_method == "adaptive":
                    processed_df, _ = smooth_adaptive(processed_df, min_window=3, max_window=self.smoothing_window, threshold=self.threshold_factor)
                self.summary_metadata['corrections_applied'].append(f"{self.smoothing_method.capitalize()} smoothing")
            
            # Apply frequency filtering if requested
            if self.apply_filtering:
                write_site_log(f"Applying {self.filter_type} filtering...")
                
                # Create filter configuration
                filter_config = {
                    'type': self.filter_type,
                    'channels': self.filter_channels,
                    'method': 'filtfilt'  # Use zero-phase filtering by default
                }
                
                # Add filter-specific parameters
                if self.filter_type == 'comb':
                    filter_config.update({
                        'notch_freq': self.filter_params.get('notch_freq', 50.0),
                        'quality_factor': self.filter_params.get('quality_factor', 30.0),
                        'harmonics': self.filter_params.get('harmonics', None)
                    })
                elif self.filter_type == 'bandpass':
                    filter_config.update({
                        'low_freq': self.filter_params.get('low_freq', 0.1),
                        'high_freq': self.filter_params.get('high_freq', 10.0),
                        'order': self.filter_params.get('order', 4)
                    })
                elif self.filter_type == 'highpass':
                    filter_config.update({
                        'cutoff_freq': self.filter_params.get('cutoff_freq', 0.1),
                        'order': self.filter_params.get('order', 4)
                    })
                elif self.filter_type == 'lowpass':
                    filter_config.update({
                        'cutoff_freq': self.filter_params.get('cutoff_freq', 10.0),
                        'order': self.filter_params.get('order', 4)
                    })
                elif self.filter_type == 'adaptive':
                    filter_config.update({
                        'reference_channel': self.filter_params.get('reference_channel', 'rBx'),
                        'filter_length': self.filter_params.get('filter_length', 64),
                        'mu': self.filter_params.get('mu', 0.01)
                    })
                elif self.filter_type == 'custom':
                    filter_config.update({
                        'b': self.filter_params.get('b', [1.0]),
                        'a': self.filter_params.get('a', [1.0])
                    })
                
                # Apply the filter
                fs = 1.0 / sample_interval
                processed_df = filter_data(processed_df, filter_config, fs)
                
                # Update summary metadata
                filter_description = f"{self.filter_type.capitalize()} filtering"
                if self.filter_type == 'comb':
                    notch_freq = filter_config.get('notch_freq', 50.0)
                    filter_description += f" (notch at {notch_freq} Hz)"
                elif self.filter_type == 'bandpass':
                    low_freq = filter_config.get('low_freq', 0.1)
                    high_freq = filter_config.get('high_freq', 10.0)
                    filter_description += f" ({low_freq}-{high_freq} Hz)"
                elif self.filter_type == 'highpass':
                    cutoff_freq = filter_config.get('cutoff_freq', 0.1)
                    filter_description += f" (cutoff: {cutoff_freq} Hz)"
                elif self.filter_type == 'lowpass':
                    cutoff_freq = filter_config.get('cutoff_freq', 10.0)
                    filter_description += f" (cutoff: {cutoff_freq} Hz)"
                
                self.summary_metadata['corrections_applied'].append(filter_description)
                
                # Plot filter response if requested
                if self.save_plots and self.filter_type != 'adaptive':
                    try:
                        # Get filter coefficients for plotting
                        if self.filter_type == 'comb':
                            b, a = design_comb_filter(fs, 
                                                    filter_config.get('notch_freq', 50.0),
                                                    filter_config.get('quality_factor', 30.0),
                                                    filter_config.get('harmonics', None))
                        elif self.filter_type == 'bandpass':
                            b, a = design_bandpass_filter(fs,
                                                        filter_config.get('low_freq', 0.1),
                                                        filter_config.get('high_freq', 10.0),
                                                        filter_config.get('order', 4))
                        elif self.filter_type == 'highpass':
                            b, a = design_highpass_filter(fs,
                                                        filter_config.get('cutoff_freq', 0.1),
                                                        filter_config.get('order', 4))
                        elif self.filter_type == 'lowpass':
                            b, a = design_lowpass_filter(fs,
                                                       filter_config.get('cutoff_freq', 10.0),
                                                       filter_config.get('order', 4))
                        else:
                            b, a = np.array([1.0]), np.array([1.0])
                        
                        plot_filter_response(b, a, fs, f"{self.filter_type.capitalize()} Filter", 
                                           save_plot=True, output_dir=self.output_dir)
                    except Exception as e:
                        write_site_log(f"Error plotting filter response: {e}", level="WARNING")
            
            # Calculate channel means and field strengths
            for col in processed_df.columns:
                if col in ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'rBx', 'rBy'] and col in processed_df.columns:
                    self.summary_metadata['channel_means'][col] = processed_df[col].mean()
            
            # Calculate field strengths
            if 'Bx' in processed_df.columns and 'By' in processed_df.columns and 'Bz' in processed_df.columns:
                total_field = np.sqrt(processed_df['Bx']**2 + processed_df['By']**2 + processed_df['Bz']**2)
                horizontal_field = np.sqrt(processed_df['Bx']**2 + processed_df['By']**2)
                self.summary_metadata['field_strengths']['Total magnetic field'] = total_field.mean()
                self.summary_metadata['field_strengths']['Horizontal magnetic field'] = horizontal_field.mean()
            
            # Save processed data if requested
            if self.save_processed_data:
                # Prepare output data - only include channel columns (no datetime)
                output_columns = []
                
                # Add channel columns based on what's available
                if self.apply_rotation and 'Hx' in processed_df.columns:
                    output_columns.extend(['Hx', 'Dx', 'Z_rot'])
                    if 'Ex_rot' in processed_df.columns:
                        output_columns.extend(['Ex_rot', 'Ey_rot'])
                else:
                    output_columns.extend(['Bx', 'By', 'Bz'])
                    if 'Ex' in processed_df.columns:
                        output_columns.extend(['Ex', 'Ey'])
                
                # Add remote reference channels if available
                if 'rBx' in processed_df.columns:
                    output_columns.extend(['rBx', 'rBy'])
                
                output_df = pd.DataFrame(processed_df[output_columns].copy())
                
                # Add sample rate suffix to filename for all decimation factors
                suffix = get_sample_rate_suffix(sample_interval)
                # New naming: CPU{pid}_{timestamp}_SITENAME_10Hz_Process.txt
                cpu_prefix = get_cpu_prefix()
                filename = f"{cpu_prefix}_{self.site_name}{suffix}_Process.txt"
                processed_output_path = os.path.join(".", filename)
                float_cols = output_df.select_dtypes(include=['float', 'float64', 'float32']).columns
                output_df[float_cols] = output_df[float_cols].round(3)
                output_df.to_csv(processed_output_path, index=False, header=False, sep='\t', float_format='%.3f')
                write_site_log(f"Processed data saved: {filename}")

                # Generate config file after processed data is saved
                if self.gps_data:
                    has_remote = self.remote_reference is not None
                    site_name_for_config = self.site_name
                    if hasattr(self, 'decimation_factor') and self.decimation_factor > 1:
                        suffix = get_sample_rate_suffix(sample_interval)
                        site_name_for_config = f"{self.site_name}{suffix}"
                    config_path = generate_processing_config(
                        site_name_for_config,
                        self.gps_data,
                        sample_interval,
                        has_remote_reference=has_remote,
                        remote_reference_site=self.remote_reference,
                        output_file_path=processed_output_path
                    )
            
            # Update results
            self.results['data_shape'] = processed_df.shape
            self.results['status'] = 'SUCCESS'
            self.results['end_time'] = datetime.datetime.now()
            
            # Generate plots if requested
            if self.plot_timeseries:
                write_site_log(f"Generating plots...")
                
                # Get sample rate suffix for plotting
                suffix = ""
                if hasattr(self, 'decimation_factor') and self.decimation_factor > 1:
                    suffix = get_sample_rate_suffix(sample_interval)
                
                site_name_for_plot = f"{self.site_name}{suffix}"
                plot_physical_channels(
                    processed_df, 
                    plot_boundaries=self.plot_boundaries,
                    plot_smoothed_windows=self.plot_smoothed_windows,
                    tilt_corrected=self.tilt_correction,
                    save_plots=self.save_plots,
                    site_name=site_name_for_plot,
                    output_dir=self.output_dir,
                    drift_corrected=self.apply_drift_correction,
                    rotated=self.apply_rotation,
                    plot_tilt=self.plot_tilt,
                    plot_original_data=self.plot_original_data,
                    gps_data=self.gps_data,
                    tilt_angle_degrees=tilt_angle_degrees,
                    remote_reference_site=self.remote_reference,
                    xarm=self.metadata.get("xarm"),
                    yarm=self.metadata.get("yarm"),
                    skip_minutes=self.skip_minutes,
                    timezone=main_timezone
                )
            
            # Perform frequency analysis if requested
            if self.perform_freq_analysis:
                write_site_log(f"Performing frequency analysis...")
                
                # Add sample rate suffix to site name if decimation is applied
                site_name_for_freq = self.site_name
                if hasattr(self, 'decimation_factor') and self.decimation_factor > 1:
                    suffix = get_sample_rate_suffix(sample_interval)
                    site_name_for_freq = f"{self.site_name}{suffix}"
                
                self.frequency_analysis(processed_df, site_name_for_freq)
            
            # Plot coherence if requested
            if self.plot_coherence:
                write_site_log(f"Generating coherence plots...")
                
                # Add sample rate suffix to site name if decimation is applied
                site_name_for_coherence = self.site_name
                if hasattr(self, 'decimation_factor') and self.decimation_factor > 1:
                    suffix = get_sample_rate_suffix(sample_interval)
                    site_name_for_coherence = f"{self.site_name}{suffix}"
                
                plot_coherence_plots(
                    processed_df, 
                    fs=1.0/sample_interval,
                    save_plots=self.save_plots,
                    site_name=site_name_for_coherence,
                    output_dir=self.output_dir
                )
            
            # Generate heatmap plots if requested
            if self.plot_heatmaps:
                write_site_log(f"Generating heatmap plots for quality control...")
                
                # Add sample rate suffix to site name if decimation is applied
                site_name_for_heatmaps = self.site_name
                if hasattr(self, 'decimation_factor') and self.decimation_factor > 1:
                    suffix = get_sample_rate_suffix(sample_interval)
                    site_name_for_heatmaps = f"{self.site_name}{suffix}"
                
                # Parse custom thresholds if provided
                heatmap_thresholds = self.heatmap_thresholds
                if isinstance(heatmap_thresholds, str):
                    try:
                        # Parse thresholds from string format like "0.8,0.6,0.6"
                        parts = heatmap_thresholds.split(',')
                        if len(parts) == 3:
                            heatmap_thresholds = {
                                'good': float(parts[0]),
                                'fair': float(parts[1]),
                                'poor': float(parts[2])
                            }
                    except Exception as e:
                        write_site_log(f"Error parsing heatmap thresholds: {e}, using defaults", level="WARNING")
                        heatmap_thresholds = None
                
                plot_heatmaps(
                    processed_df,
                    fs=1.0/sample_interval,
                    save_plots=self.save_plots,
                    site_name=site_name_for_heatmaps,
                    output_dir=self.output_dir,
                    nperseg=self.heatmap_nperseg,
                    noverlap=self.heatmap_noverlap,
                    coherence_thresholds=heatmap_thresholds
                )
            
            # Display summary table
            summary_table = create_processing_summary_table(self.site_name, self.summary_metadata)
            print(summary_table)
            write_log(summary_table)
            
            write_site_log(f"Processing completed successfully for {self.site_name}")
            
        except Exception as e:
            error_msg = f"Error processing {self.site_name}: {e}"
            write_site_log(error_msg, level="ERROR")
            write_site_log(f"Traceback: {traceback.format_exc()}", level="ERROR")
            self.results['status'] = 'FAILED'
            self.results['error'] = str(e)
            self.results['end_time'] = datetime.datetime.now()
            raise

    def convert_ascii_to_physical_units(self, df):
        """Converts ASCII data from raw counts into physical units.
        
        Magnetics:
          Bx = (BX/10^7) * 70000  
          Bz = (BZ/10^7) * 70000  
          By = -(BY/10^7) * 70000  
        Electrics:
          Ex = -(EX / xarm)  
          Ey = -(EY / yarm)
        
        Args:
            df (pd.DataFrame): DataFrame with ASCII columns "BX", "BY", "BZ", "EX", "EY".
            metadata (dict): Contains "xarm" and "yarm".
        
        Returns:
            pd.DataFrame: DataFrame with converted physical units in "Bx", "By", "Bz", "Ex", "Ey".
        """
        divisor = 10**7
        xarm = self.metadata.get("xarm", 100.0)
        yarm = self.metadata.get("yarm", 100.0)
        
        # Convert magnetic fields
        df["Bx"] = (df["BX"].astype(float) / divisor) * 70000.0
        df["Bz"] = (df["BZ"].astype(float) / divisor) * 70000.0
        df["By"] = -(df["BY"].astype(float) / divisor) * 70000.0
        
        # Convert electric fields
        df["Ex"] = -(df["EX"].astype(float) / xarm)
        df["Ey"] = -(df["EY"].astype(float) / yarm)
        
        # Remove original columns
        df = df.drop(columns=["BX", "BY", "BZ", "EX", "EY"])
        
        return df

    def frequency_analysis(self, df, identifier):
        """Performs frequency analysis on the processed data.
        
        Args:
            df (pd.DataFrame): Processed DataFrame.
            identifier (str): Identifier for the analysis.
        """
        try:
            sample_interval = self.metadata["sample_interval"]
            fs = 1.0 / sample_interval
            
            # Determine which channels to analyze
            channels = []
            if 'Bx' in df.columns:
                channels.append('Bx')
            if 'By' in df.columns:
                channels.append('By')
            if 'Bz' in df.columns:
                channels.append('Bz')
            if 'Ex' in df.columns:
                channels.append('Ex')
            if 'Ey' in df.columns:
                channels.append('Ey')
            
            # Add remote reference channels if available
            if 'rBx' in df.columns:
                channels.append('rBx')
            if 'rBy' in df.columns:
                channels.append('rBy')
            
            if not channels:
                write_log(f"No channels available for frequency analysis for {identifier}", level="WARNING")
                return
            
            # Parse frequency analysis options
            freq_options = self.perform_freq_analysis.upper() if isinstance(self.perform_freq_analysis, str) else "W"
            
            # Perform requested frequency analysis methods
            if 'W' in freq_options:
                write_log(f"Performing Welch spectral analysis for {identifier}")
                plot_welch_spectra(df, channels, fs, 
                                  save_plots=self.save_plots, 
                                  site_name=identifier, 
                                  output_dir=self.output_dir)
            
            if 'M' in freq_options:
                write_log(f"Performing Multi-taper spectral analysis for {identifier}")
                plot_multitaper_spectra(df, channels, fs, 
                                       save_plots=self.save_plots, 
                                       site_name=identifier, 
                                       output_dir=self.output_dir)
            
            if 'S' in freq_options:
                write_log(f"Performing Spectrogram analysis for {identifier}")
                plot_spectrograms(df, channels, fs, 
                                 save_plots=self.save_plots, 
                                 site_name=identifier, 
                                 output_dir=self.output_dir)
            
            write_log(f"Frequency analysis completed for {identifier} (methods: {freq_options})")
            
        except Exception as e:
            write_log(f"Error in frequency analysis for {identifier}: {e}", level="ERROR")

    def run_lemimt_processing(self, processed_file_path):
        """Run lemimt.exe on the processed output file with improved error handling and permissions.
        
        Args:
            processed_file_path (str): Path to the processed output file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import platform
            import subprocess
            import stat
            import shutil
            import traceback
            
            write_site_log("=" * 60)
            write_site_log("STARTING LEMIMT PROCESSING")
            write_site_log("=" * 60)
            
            # Enhanced platform detection and handling
            current_platform = platform.system()
            write_site_log(f"Platform detected: {current_platform}")
            
            # Check if we're on Windows
            if current_platform != "Windows":
                write_site_log(f"lemimt.exe can only run on Windows. Current platform: {current_platform}", level="WARNING")
                write_site_log(f"To run lemimt processing, transfer the file {os.path.abspath(processed_file_path)} to a Windows machine", level="INFO")
                filename_only = os.path.basename(processed_file_path)
                write_site_log(f"Command to run on Windows: {self.lemimt_path} -r -f {filename_only}", level="INFO")
                
                write_site_log("=" * 60)
                write_site_log("LEMIMT PROCESSING SKIPPED (Non-Windows platform)")
                write_site_log("=" * 60)
                return False
            
            # Enhanced executable validation
            write_site_log("Validating lemimt executable...")
            lemimt_path = self._validate_lemimt_executable()
            if not lemimt_path:
                write_site_log("=" * 60)
                write_site_log("LEMIMT PROCESSING FAILED (Invalid executable)")
                write_site_log("=" * 60)
                return False
            
            # Enhanced file validation
            write_site_log("Validating input files...")
            if not self._validate_input_files(processed_file_path):
                write_site_log("=" * 60)
                write_site_log("LEMIMT PROCESSING FAILED (Invalid input files)")
                write_site_log("=" * 60)
                return False
            
            # Get working directory and ensure we're in the right place
            working_dir = os.path.dirname(os.path.abspath(processed_file_path))
            if not working_dir:
                working_dir = os.getcwd()
            
            # Use only the filename for the lemimt command
            filename_only = os.path.basename(processed_file_path)
            
            # Construct the command with enhanced options
            cmd = [lemimt_path, "-r", "-f", filename_only]
            write_site_log(f"Working directory: {working_dir}")
            write_site_log(f"Input file: {filename_only}")
            write_site_log(f"Full command: {' '.join(cmd)}")
            write_site_log("Preparing to execute lemimt...")
            
            # Enhanced subprocess execution with better error handling
            result = self._execute_lemimt_command(cmd, working_dir)
            
            if result['success']:
                write_site_log("=" * 60)
                write_site_log("LEMIMT PROCESSING COMPLETED SUCCESSFULLY")
                write_site_log("=" * 60)
                write_site_log(f"Return code: {result.get('returncode', 'N/A')}")
                if result.get('stdout'):
                    write_site_log(f"lemimt output: {result['stdout']}")
                
                # TODO: Uncomment the following lines to delete processed data and config files after lemimt processing
                # write_site_log("Cleaning up processed files...")
                # self._cleanup_processed_files(processed_file_path)
                
                return True
            else:
                write_site_log("=" * 60)
                write_site_log("LEMIMT PROCESSING FAILED")
                write_site_log("=" * 60)
                write_site_log(f"Error: {result.get('error', 'Unknown error')}")
                write_site_log(f"Return code: {result.get('returncode', 'N/A')}")
                if result.get('stderr'):
                    write_site_log(f"lemimt stderr: {result['stderr']}")
                return False
                
        except Exception as e:
            write_site_log("=" * 60)
            write_site_log("LEMIMT PROCESSING FAILED (Unexpected error)")
            write_site_log("=" * 60)
            write_site_log(f"Error: {e}")
            write_site_log(f"Traceback: {traceback.format_exc()}", level="ERROR")
            return False

    def _validate_lemimt_executable(self):
        """Validate the lemimt executable at the specified path."""
        try:
            import stat
            
            # Check if the specified path exists
            if not os.path.exists(self.lemimt_path):
                write_site_log(f"lemimt.exe not found at: {self.lemimt_path}", level="ERROR")
                return None
            
            lemimt_path = os.path.abspath(self.lemimt_path)
            write_site_log(f"Found lemimt.exe at: {lemimt_path}")
            
            # Check if it's actually an executable
            if not os.access(lemimt_path, os.X_OK):
                write_site_log(f"lemimt.exe is not executable: {lemimt_path}", level="ERROR")
                # Try to make it executable (Windows doesn't need this, but good practice)
                try:
                    os.chmod(lemimt_path, os.stat(lemimt_path).st_mode | stat.S_IEXEC)
                    write_site_log(f"Made lemimt.exe executable")
                except Exception as e:
                    write_site_log(f"Could not make lemimt.exe executable: {e}", level="WARNING")
            
            # Validate it's actually the right file
            if not lemimt_path.lower().endswith('.exe'):
                write_site_log(f"Warning: lemimt path doesn't end with .exe: {lemimt_path}", level="WARNING")
            
            return lemimt_path
            
        except Exception as e:
            write_site_log(f"Error validating lemimt executable: {e}", level="ERROR")
            return None

    def _validate_input_files(self, processed_file_path):
        """Validate that all required input files exist."""
        try:
            # Check processed data file
            if not os.path.exists(processed_file_path):
                write_site_log(f"Processed data file not found: {processed_file_path}", level="ERROR")
                return False
            
            # Check if file is readable
            if not os.access(processed_file_path, os.R_OK):
                write_site_log(f"Processed data file is not readable: {processed_file_path}", level="ERROR")
                return False
            
            # Check file size (should not be empty)
            file_size = os.path.getsize(processed_file_path)
            if file_size == 0:
                write_site_log(f"Processed data file is empty: {processed_file_path}", level="ERROR")
                return False
            
            write_site_log(f"Processed data file validated: {processed_file_path} ({file_size} bytes)")
            
            # Check for corresponding config file
            config_file_path = os.path.splitext(processed_file_path)[0] + ".cfg"
            if not os.path.exists(config_file_path):
                write_site_log(f"Config file not found: {config_file_path}", level="WARNING")
                # This is not fatal, lemimt might work without it
            else:
                write_site_log(f"Config file found: {config_file_path}")
            
            return True
            
        except Exception as e:
            write_site_log(f"Error validating input files: {e}", level="ERROR")
            return False

    def _execute_lemimt_command(self, cmd, working_dir):
        """Execute the lemimt command with enhanced error handling."""
        try:
            import platform
            import subprocess
            
            # Set up environment variables for better compatibility
            env = os.environ.copy()
            
            # Add current directory to PATH if not already there
            current_dir = os.getcwd()
            if current_dir not in env.get('PATH', ''):
                env['PATH'] = current_dir + os.pathsep + env.get('PATH', '')
            
            write_site_log(f"Executing command in directory: {working_dir}")
            write_site_log(f"Command: {' '.join(cmd)}")
            
            # Run the command with enhanced options
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout (increased from 5)
                shell=False,  # Explicitly set to False for security
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout.strip() if result.stdout else None,
                'stderr': result.stderr.strip() if result.stderr else None,
                'error': None
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"lemimt processing timed out after 10 minutes",
                'returncode': -1
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': f"lemimt executable not found: {cmd[0]}",
                'returncode': -1
            }
        except PermissionError:
            return {
                'success': False,
                'error': f"Permission denied running lemimt. Try running as administrator.",
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'returncode': -1
            }

    def _cleanup_processed_files(self, processed_file_path):
        """Clean up processed data and config files after successful lemimt processing."""
        try:
            import os
            
            processed_file_name = os.path.basename(processed_file_path)
            config_file_name = processed_file_name.replace('.txt', '.cfg')
            
            # Delete processed data file
            if os.path.exists(processed_file_name):
                os.remove(processed_file_name)
                write_site_log(f"Deleted processed data file: {processed_file_name}")
            
            # Delete config file
            if os.path.exists(config_file_name):
                os.remove(config_file_name)
                write_site_log(f"Deleted config file: {config_file_name}")
                
        except Exception as e:
            write_site_log(f"Error during cleanup: {e}", level="WARNING")


def read_processing_config(processing_file="Processing.txt"):
    """Reads the Processing.txt file to get site-specific parameters and remote reference information.
    
    Supports both 3-column format (Site, xarm, yarm) and 4-column format (Site, xarm, yarm, RemoteReference).
    
    Args:
        processing_file (str): Path to the Processing.txt file
        
    Returns:
        dict: Dictionary mapping site names to their parameters and remote reference
    """
    config = {}
    try:
        with open(processing_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header line if it exists (only if it looks like a header with column names)
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Only skip if it's a header line with column names, not if it contains "site" in the site name
            if line.lower().startswith('site,') or line.lower().startswith('site name,') or line.lower().startswith('site_name,'):
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) == 4:
                site, xarm, yarm, remote_ref = parts
                config[site] = {
                    'xarm': float(xarm),
                    'yarm': float(yarm),
                    'remote_reference': remote_ref
                }
            elif len(parts) == 3:
                site, xarm, yarm = parts
                config[site] = {
                    'xarm': float(xarm),
                    'yarm': float(yarm),
                    'remote_reference': None
                }
        return config
    except FileNotFoundError:
        write_log(f"Processing.txt file not found: {processing_file}", level="WARNING")
        return {}
    except Exception as e:
        write_log(f"Error reading Processing.txt: {e}", level="ERROR")
        return {}


def decimal_degrees_to_degrees_minutes(decimal_degrees, is_latitude=True):
    """Convert decimal degrees to degrees-minutes format with hemisphere indicator.
    
    Args:
        decimal_degrees (float): Decimal degrees (can be positive or negative)
        is_latitude (bool): True if converting latitude, False if longitude
        
    Returns:
        str: Formatted string in "DD MM.MMMMMM,H" format where:
             DD = degrees (integer)
             MM.MMMMMM = minutes (decimal)
             H = hemisphere (N/S for latitude, E/W for longitude)
    
    Examples:
        decimal_degrees_to_degrees_minutes(32.16071, True) -> "32 9.64273,N"
        decimal_degrees_to_degrees_minutes(-5.62940, False) -> "5 37.76425,W"
    """
    # Determine hemisphere based on sign and coordinate type
    if is_latitude:
        # Latitude: positive = North (N), negative = South (S)
        hemisphere = 'N' if decimal_degrees >= 0 else 'S'
    else:
        # Longitude: positive = East (E), negative = West (W)
        hemisphere = 'E' if decimal_degrees >= 0 else 'W'
    
    # Work with absolute value
    abs_degrees = abs(decimal_degrees)
    
    # Extract degrees (integer part)
    degrees = int(abs_degrees)
    
    # Calculate minutes (decimal part)
    minutes = (abs_degrees - degrees) * 60.0
    
    # Format: "DD MM.MMMMMM,H"
    return f"{degrees} {minutes:.5f},{hemisphere}"


def get_cpu_prefix():
    """Get a worker number prefix for output files to prevent overwriting.
    
    Returns:
        str: A worker prefix based on the current worker number (P01_, P02_, etc.).
    """
    import os
    import threading
    
    # First check for environment variable (set by batch processing)
    worker_num = os.environ.get('WORKER_NUM')
    if worker_num:
        try:
            return f"P{int(worker_num):02d}_"
        except ValueError:
            pass
    
    # Get the current thread name to determine worker number
    thread_name = threading.current_thread().name
    
    # Extract worker number from thread name (e.g., "ThreadPoolExecutor-0_0" -> "01")
    if "ThreadPoolExecutor" in thread_name:
        # ThreadPoolExecutor uses format like "ThreadPoolExecutor-0_0", "ThreadPoolExecutor-0_1", etc.
        try:
            # Extract the worker number from the thread name
            parts = thread_name.split('_')
            if len(parts) >= 2:
                worker_num = int(parts[-1]) + 1  # Convert to 1-based indexing
                return f"P{worker_num:02d}_"
        except (ValueError, IndexError):
            pass
    
    # Fallback: use process ID if we can't determine worker number
    pid = os.getpid()
    return f"P{pid:02d}_"


def get_next_available_identifier(base_site_name, output_dir="."):
    """Determine the next available identifier letter (A-Z) for a given base site name.
    
    This function checks for existing EDI files with the pattern {base_site_name}-[A-Z].edi
    and returns the next available letter.
    
    Args:
        base_site_name (str): Base site name (e.g., "MT_HDD5449_RR-HDD5974_10Hz")
        output_dir (str): Directory to check for existing files
        
    Returns:
        str: Next available identifier letter (A-Z)
    """
    import glob
    import os
    
    # Check for existing EDI files with this base name
    pattern = os.path.join(output_dir, f"{base_site_name}-*.edi")
    existing_files = glob.glob(pattern)
    
    # Extract used letters from existing files
    used_letters = set()
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Extract the letter after the last dash
        if filename.endswith('.edi'):
            parts = filename[:-4].split('-')  # Remove .edi and split by dash
            if len(parts) > 1:
                last_part = parts[-1]
                if len(last_part) == 1 and last_part.isalpha():
                    used_letters.add(last_part.upper())
    
    # Find the first available letter
    available_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - used_letters
    if available_letters:
        return min(available_letters)  # Return the first available letter alphabetically
    else:
        # If all letters are used, start over with A (this shouldn't happen in practice)
        write_log(f"WARNING: All identifier letters A-Z are used for {base_site_name}, reusing A", level="WARNING")
        return 'A'


def get_next_run_letter(base_site_name, output_dir="."):
    """Determine the next available run letter (A-Z) for a given base site name.
    Scans for existing files and returns the next available first letter.
    """
    import glob
    import os
    pattern = os.path.join(output_dir, f"{base_site_name}-???.edi")
    existing_files = glob.glob(pattern)
    used_first_letters = set()
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        if filename.endswith('.edi') and '-' in filename:
            dash_split = filename.rsplit('-', 1)
            if len(dash_split) == 2:
                identifier = dash_split[1][:3]  # Get first 3 characters
                if len(identifier) == 3 and identifier[0].isalpha() and identifier[1:].isdigit():
                    used_first_letters.add(identifier[0].upper())
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if letter not in used_first_letters:
            return letter
    # If all used, start over (shouldn't happen)
    return 'A'

def get_file_number(file_index):
    """Get the file number (00-99) for the given file index (0-based)."""
    if 0 <= file_index <= 99:
        return f"{file_index:02d}"  # Format as two-digit number
    else:
        return "00"  # fallback


def generate_processing_config(site_name, gps_data, sample_interval, has_remote_reference=False, 
                              remote_reference_site=None, output_file_path=None, run_index=None, file_index=0):
    """Generate a processing configuration file with a letter-number identifier (e.g., A00, A01, B00)."""
    # Get CPU prefix to prevent file overwriting
    cpu_prefix = get_cpu_prefix()
    
    if output_file_path:
        config_path = os.path.splitext(output_file_path)[0] + ".cfg"
    else:
        suffix = get_sample_rate_suffix(sample_interval)
        config_filename = f"{site_name}{suffix}_Process.cfg"
        config_path = os.path.join(".", config_filename)
    if os.path.dirname(config_path) not in ("", "."):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config_site_name = f"MT_{site_name}"
    if has_remote_reference and remote_reference_site:
        config_site_name += f"_RR-{remote_reference_site}"
    sample_rate = 1.0 / sample_interval
    if sample_rate >= 1:
        freq_str = f"_{int(sample_rate)}Hz"
    else:
        freq_str = f"_{sample_rate:.1f}Hz"
    config_site_name += freq_str
    # Letter-number identifier (e.g., A00, A01, B00)
    base_site_name = config_site_name
    output_dir = os.path.dirname(config_path) if os.path.dirname(config_path) else "."
    if run_index is None:
        run_letter = get_next_run_letter(base_site_name, output_dir)
    else:
        run_letter = run_index
    file_number = get_file_number(file_index)
    config_site_name += f"-{run_letter}{file_number}"
    latitude = gps_data.get('latitude', 0.0)
    longitude = gps_data.get('longitude', 0.0)
    elevation = gps_data.get('altitude', 0.0)
    latitude_dm = decimal_degrees_to_degrees_minutes(latitude, is_latitude=True)
    longitude_dm = decimal_degrees_to_degrees_minutes(longitude, is_latitude=False)
    declination = '0.0'  # TODO: Calculate actual declination
    nchan = 7 if has_remote_reference else 5
    C1 = '1'
    C2 = '1'
    C3 = '1'
    C4 = '1'
    C5 = '1'
    R1 = '1' if has_remote_reference else None
    R2 = '1' if has_remote_reference else None
    write_log(f"Writing coordinates to config file: Lat={latitude:.6f}° ({latitude_dm}), Lon={longitude:.6f}° ({longitude_dm}), Elev={elevation:.1f}m")
    write_log(f"Using config site name: {config_site_name}")
    with open(config_path, 'w') as f:
        f.write(f"SITE {config_site_name}\n")
        f.write(f"LATITUDE {latitude_dm}\n")
        f.write(f"LONGITUDE {longitude_dm}\n")
        f.write(f"ELEVATION {elevation}\n")
        f.write(f"DECLINATION {declination}\n")
        f.write(f"SAMPLING {sample_interval}\n\n")
        f.write(f"NCHAN {nchan}\n")
        if has_remote_reference:
            f.write(f"  1   {C1} 1  l120new.rsp\n")
            f.write(f"  2   {C2} 1  l120new.rsp\n")
            f.write(f"  3   {C3} 1  l120new.rsp\n")
            f.write(f"  4   {C4} 1  e000.rsp\n")
            f.write(f"  5   {C5} 1  e000.rsp\n")
            f.write(f"  6   {R1} 1  l120new.rsp\n")
            f.write(f"  7   {R2} 1  l120new.rsp\n")
            f.write(f"NREFCH        2\n")
        else:
            f.write(f"  1   {C1} 1  l120new.rsp\n")
            f.write(f"  2   {C2} 1  l120new.rsp\n")
            f.write(f"  3   {C3} 1  l120new.rsp\n")
            f.write(f"  4   {C4} 1  e000.rsp\n")
            f.write(f"  5   {C5} 1  e000.rsp\n")
    write_log(f"Generated processing config file: {config_path}")
    write_log(f"Coordinates: Lat={latitude_dm}, Lon={longitude_dm}")
    return config_path


def decimate_dataframe(df, decimation_factor, sample_interval):
    """Decimate a DataFrame by taking every nth row.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        decimation_factor (int): Decimation factor (e.g., 2 for every 2nd row)
        sample_interval (float): Original sample interval in seconds
        
    Returns:
        tuple: (decimated_df, new_sample_interval)
    """
    if decimation_factor <= 1:
        return df, sample_interval
    
    # Take every nth row
    decimated_df = df.iloc[::decimation_factor].copy()
    
    # Update sample interval
    new_sample_interval = sample_interval * decimation_factor
    
    write_log(f"Decimated data by factor {decimation_factor}: {len(df)} -> {len(decimated_df)} samples, "
              f"sample interval: {sample_interval:.3f}s -> {new_sample_interval:.3f}s")
    
    return decimated_df, new_sample_interval


def get_sample_rate_suffix(sample_interval):
    """Get a suffix for filenames based on sample rate.
    
    Args:
        sample_interval (float): Sample interval in seconds
        
    Returns:
        str: Suffix like '_10Hz', '_5Hz', etc.
    """
    sample_rate = 1.0 / sample_interval
    if sample_rate >= 1:
        return f"_{int(sample_rate)}Hz"
    else:
        return f"_{sample_rate:.1f}Hz"


def calculate_magnetic_declination(latitude, longitude, date=None):
    """Calculate magnetic declination for a given location and date.
    
    This is a placeholder function. In practice, you would use a proper
    magnetic declination calculator like the World Magnetic Model (WMM).
    
    Args:
        latitude (float): Latitude in decimal degrees
        longitude (float): Longitude in decimal degrees
        date (datetime, optional): Date for calculation. Defaults to current date.
        
    Returns:
        float: Magnetic declination in degrees
    """
    # TODO: Implement proper magnetic declination calculation
    # For now, return a placeholder value
    # You could use libraries like:
    # - geomag (Python wrapper for WMM)
    # - pyproj with magnetic models
    # - Online APIs like NOAA's magnetic declination calculator
    
    return 0.0  # Placeholder


def get_timezone_abbreviation(timezone_name):
    """Get the abbreviation for a timezone name.
    
    Args:
        timezone_name (str): Timezone name like 'Australia/Adelaide'
        
    Returns:
        str: Timezone abbreviation like 'ACDT'
        
    Alternative display ideas for future consideration:
    - Show timezone in plot title: "Time Series for HDD5449 (ACDT)"
    - Add timezone info box in plot corner
    - Use different colors for different timezones
    - Show timezone conversion info in legend
    """
    timezone_abbrevs = {
        'Australia/Adelaide': 'ACDT',
        'Australia/Sydney': 'AEDT', 
        'Australia/Perth': 'AWST',
        'Australia/Darwin': 'ACST',
        'Australia/Brisbane': 'AEST',
        'America/New_York': 'EST',
        'America/Chicago': 'CST',
        'America/Denver': 'MST',
        'America/Los_Angeles': 'PST',
        'Europe/London': 'GMT',
        'Europe/Paris': 'CET',
        'Asia/Tokyo': 'JST',
        'UTC': 'UTC'
    }
    return timezone_abbrevs.get(timezone_name, timezone_name.split('/')[-1].upper())


def plot_welch_spectra(df, channels, fs, nperseg=1024, save_plots=False, site_name="UnknownSite", output_dir="."):
    """Plots Welch power spectra for specified channels.
    
    Args:
        df (pd.DataFrame): DataFrame with channel data.
        channels (list): List of channel names to plot.
        fs (float): Sampling frequency.
        nperseg (int): Number of points per segment for FFT.
        save_plots (bool): Whether to save plots to files.
        site_name (str): Site name for plot titles and filenames.
        output_dir (str): Directory to save plots in.
    """
    try:
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3*len(channels)))
        if len(channels) == 1:
            axes = [axes]
        
        for i, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].dropna()
                if len(data) > 0:
                    f, Pxx = welch(data, fs=fs, nperseg=min(nperseg, len(data)//2))
                    axes[i].semilogy(f, Pxx)
                    axes[i].set_xlabel('Frequency [Hz]')
                    axes[i].set_ylabel('Power Spectral Density')
                    axes[i].set_title(f'{channel} Welch Power Spectrum - {site_name}')
                    axes[i].grid(True)
                else:
                    axes[i].text(0.5, 0.5, f'No data for {channel}', ha='center', va='center', transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, f'Channel {channel} not found', ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(output_dir, f"{site_name}_welch_spectra.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            write_log(f"Welch spectra plot saved as {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        write_log(f"Error in plot_welch_spectra: {e}", level="ERROR")


def plot_multitaper_spectra(df, channels, fs, nperseg=1024, save_plots=False, site_name="UnknownSite", output_dir="."):
    """Plots Multi-taper power spectra for specified channels.
    
    Args:
        df (pd.DataFrame): DataFrame with channel data.
        channels (list): List of channel names to plot.
        fs (float): Sampling frequency.
        nperseg (int): Number of points per segment for FFT.
        save_plots (bool): Whether to save plots to files.
        site_name (str): Site name for plot titles and filenames.
        output_dir (str): Directory to save plots in.
    """
    try:
        from scipy.signal import windows
        
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3*len(channels)))
        if len(channels) == 1:
            axes = [axes]
        
        for i, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].dropna()
                if len(data) > 0:
                    # Use DPSS (Discrete Prolate Spheroidal Sequences) windows for multi-taper
                    # Number of tapers = 2*NW - 1, where NW is the time-bandwidth product
                    NW = 4  # Time-bandwidth product
                    K = 2 * NW - 1  # Number of tapers
                    
                    # Create DPSS windows
                    window_length = min(nperseg, len(data)//2)
                    if window_length % 2 == 0:
                        window_length += 1  # Ensure odd length
                    
                    dpss_windows = windows.dpss(window_length, NW, Kmax=K)
                    
                    # Compute multi-taper power spectrum by averaging Welch estimates with different windows
                    f, Pxx = welch(data, fs=fs, nperseg=window_length, window='hann')
                    
                    # For multi-taper, we'll use a simplified approach with multiple Welch estimates
                    # This is not the full multi-taper method but provides similar benefits
                    for k in range(K):
                        # Use different window types to simulate multi-taper effect
                        window_types = ['hann', 'hamming', 'blackman', 'bartlett']
                        window_type = window_types[k % len(window_types)]
                        _, Pxx_k = welch(data, fs=fs, nperseg=window_length, window=window_type)
                        Pxx += Pxx_k
                    Pxx /= (K + 1)  # Average including the initial estimate
                    
                    axes[i].semilogy(f, Pxx)
                    axes[i].set_xlabel('Frequency [Hz]')
                    axes[i].set_ylabel('Power Spectral Density')
                    axes[i].set_title(f'{channel} Multi-taper Power Spectrum - {site_name}')
                    axes[i].grid(True)
                else:
                    axes[i].text(0.5, 0.5, f'No data for {channel}', ha='center', va='center', transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, f'Channel {channel} not found', ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(output_dir, f"{site_name}_multitaper_spectra.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            write_log(f"Multi-taper spectra plot saved as {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        write_log(f"Error in plot_multitaper_spectra: {e}", level="ERROR")


def plot_spectrograms(df, channels, fs, nperseg=256, save_plots=False, site_name="UnknownSite", output_dir="."):
    """Plots spectrograms for specified channels.
    
    Args:
        df (pd.DataFrame): DataFrame with channel data.
        channels (list): List of channel names to plot.
        fs (float): Sampling frequency.
        nperseg (int): Number of points per segment for FFT.
        save_plots (bool): Whether to save plots to files.
        site_name (str): Site name for plot titles and filenames.
        output_dir (str): Directory to save plots in.
    """
    try:
        fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3*len(channels)))
        if len(channels) == 1:
            axes = [axes]
        
        for i, channel in enumerate(channels):
            if channel in df.columns:
                data = df[channel].dropna()
                if len(data) > 0:
                    # Use shorter segments for spectrogram
                    nperseg_spec = min(nperseg, len(data)//4)
                    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg_spec, noverlap=nperseg_spec//2)
                    # Convert time axis to days
                    t_days = t / (24 * 3600)
                    # Plot spectrogram
                    im = axes[i].pcolormesh(t_days, f, 10 * np.log10(Sxx), shading='gouraud')
                    axes[i].set_ylabel('Frequency [Hz]')
                    axes[i].set_title(f'{channel} Spectrogram - {site_name}')
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=axes[i])
                    cbar.set_label('Power Spectral Density [dB/Hz]')
                else:
                    axes[i].text(0.5, 0.5, f'No data for {channel}', ha='center', va='center', transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, f'Channel {channel} not found', ha='center', va='center', transform=axes[i].transAxes)
        
        # Set x-axis label for the last subplot
        if len(axes) > 0:
            axes[-1].set_xlabel('Time [days]')
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(output_dir, f"{site_name}_spectrograms.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            write_log(f"Spectrograms plot saved as {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        write_log(f"Error in plot_spectrograms: {e}", level="ERROR")


### Filtering Functions ###

def design_comb_filter(fs, notch_freq=50.0, quality_factor=30.0, harmonics=None):
    """Design a comb filter to remove powerline noise and harmonics.
    
    Args:
        fs (float): Sampling frequency in Hz
        notch_freq (float): Fundamental frequency to notch (default: 50 Hz for powerline)
        quality_factor (float): Quality factor for the notch filter (higher = narrower)
        harmonics (list, optional): List of harmonic frequencies to notch. If None, uses first 5 harmonics.
    
    Returns:
        tuple: (b, a) filter coefficients
    """
    if harmonics is None:
        # Default to first 5 harmonics of powerline frequency
        harmonics = [notch_freq * i for i in range(1, 6)]  # 50, 100, 150, 200, 250 Hz
    
    try:
        # Start with a simple lowpass filter
        result = butter(4, 0.1, btype='low', fs=fs)
        if result is None or len(result) != 2:
            raise ValueError("butter function failed to return valid coefficients")
        b, a = result
        
        # Add notches for each harmonic
        for freq in harmonics:
            if freq < fs / 2:  # Only filter frequencies below Nyquist
                notch_result = iirnotch(freq, quality_factor, fs)
                if notch_result is None or len(notch_result) != 2:
                    continue
                notch_b, notch_a = notch_result
                # Convolve the filters
                b = np.convolve(b, notch_b)
                a = np.convolve(a, notch_a)
        
        return b, a
    except Exception as e:
        write_log(f"Error in design_comb_filter: {e}", level="ERROR")
        # Return simple pass-through filter
        return np.array([1.0]), np.array([1.0])

def design_bandpass_filter(fs, low_freq, high_freq, order=4):
    """Design a bandpass filter.
    
    Args:
        fs (float): Sampling frequency in Hz
        low_freq (float): Lower cutoff frequency in Hz
        high_freq (float): Upper cutoff frequency in Hz
        order (int): Filter order
    
    Returns:
        tuple: (b, a) filter coefficients
    """
    try:
        nyquist = fs / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Ensure frequencies are within valid range
        low_norm = max(0.001, min(low_norm, 0.999))
        high_norm = max(0.001, min(high_norm, 0.999))
        
        if low_norm >= high_norm:
            raise ValueError("Low frequency must be less than high frequency")
        
        result = butter(order, [low_norm, high_norm], btype='band', fs=fs)
        if result is None or len(result) != 2:
            raise ValueError("butter function failed to return valid coefficients")
        return result
    except Exception as e:
        write_log(f"Error in design_bandpass_filter: {e}", level="ERROR")
        # Return simple pass-through filter
        return np.array([1.0]), np.array([1.0])

def design_highpass_filter(fs, cutoff_freq, order=4):
    """Design a highpass filter.
    
    Args:
        fs (float): Sampling frequency in Hz
        cutoff_freq (float): Cutoff frequency in Hz
        order (int): Filter order
    
    Returns:
        tuple: (b, a) filter coefficients
    """
    try:
        nyquist = fs / 2
        cutoff_norm = cutoff_freq / nyquist
        
        # Ensure frequency is within valid range
        cutoff_norm = max(0.001, min(cutoff_norm, 0.999))
        
        result = butter(order, cutoff_norm, btype='high', fs=fs)
        if result is None or len(result) != 2:
            raise ValueError("butter function failed to return valid coefficients")
        return result
    except Exception as e:
        write_log(f"Error in design_highpass_filter: {e}", level="ERROR")
        # Return simple pass-through filter
        return np.array([1.0]), np.array([1.0])

def design_lowpass_filter(fs, cutoff_freq, order=4):
    """Design a lowpass filter.
    
    Args:
        fs (float): Sampling frequency in Hz
        cutoff_freq (float): Cutoff frequency in Hz
        order (int): Filter order
    
    Returns:
        tuple: (b, a) filter coefficients
    """
    try:
        nyquist = fs / 2
        cutoff_norm = cutoff_freq / nyquist
        
        # Ensure frequency is within valid range
        cutoff_norm = max(0.001, min(cutoff_norm, 0.999))
        
        result = butter(order, cutoff_norm, btype='low', fs=fs)
        if result is None or len(result) != 2:
            raise ValueError("butter function failed to return valid coefficients")
        return result
    except Exception as e:
        write_log(f"Error in design_lowpass_filter: {e}", level="ERROR")
        # Return simple pass-through filter
        return np.array([1.0]), np.array([1.0])

def apply_filter(data, b, a, filter_type="filtfilt"):
    """Apply a filter to the data.
    
    Args:
        data (np.ndarray): Input data
        b (np.ndarray): Numerator coefficients
        a (np.ndarray): Denominator coefficients
        filter_type (str): Type of filtering - "filtfilt" (zero-phase) or "lfilter" (causal)
    
    Returns:
        np.ndarray: Filtered data
    """
    if filter_type == "filtfilt":
        # Zero-phase filtering (forward-backward)
        return filtfilt(b, a, data)
    elif filter_type == "lfilter":
        # Causal filtering
        return lfilter(b, a, data)
    else:
        raise ValueError("filter_type must be 'filtfilt' or 'lfilter'")

def adaptive_filter(data, reference, filter_length=64, mu=0.01):
    """Apply adaptive filtering using LMS algorithm.
    
    Args:
        data (np.ndarray): Primary signal (signal + noise)
        reference (np.ndarray): Reference signal (noise only)
        filter_length (int): Length of adaptive filter
        mu (float): Step size for LMS algorithm
    
    Returns:
        np.ndarray: Filtered signal
    """
    # Ensure both signals have the same length
    min_len = min(len(data), len(reference))
    data = data[:min_len]
    reference = reference[:min_len]
    
    # Initialize filter weights
    w = np.zeros(filter_length)
    
    # Initialize output
    y = np.zeros(min_len)
    e = np.zeros(min_len)
    
    # LMS algorithm
    for n in range(filter_length, min_len):
        # Get reference signal window
        x = reference[n:n-filter_length:-1]
        
        # Calculate filter output
        y[n] = np.dot(w, x)
        
        # Calculate error
        e[n] = data[n] - y[n]
        
        # Update filter weights
        w = w + mu * e[n] * x
    
    return e

def filter_data(df, filter_config, fs):
    """Apply filtering to the data based on configuration.
    
    Args:
        df (pd.DataFrame): Input DataFrame with channel data
        filter_config (dict): Filter configuration
        fs (float): Sampling frequency in Hz
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Get filter parameters
    filter_type = filter_config.get('type', 'comb')
    channels = filter_config.get('channels', ['Bx', 'By', 'Bz', 'Ex', 'Ey'])
    filter_method = filter_config.get('method', 'filtfilt')
    
    write_log(f"Applying {filter_type} filter to channels: {channels}")
    
    if filter_type == 'comb':
        # Comb filter for powerline noise removal
        notch_freq = filter_config.get('notch_freq', 50.0)
        quality_factor = filter_config.get('quality_factor', 30.0)
        harmonics = filter_config.get('harmonics', None)
        
        b, a = design_comb_filter(fs, notch_freq, quality_factor, harmonics)
        
        for channel in channels:
            if channel in filtered_df.columns:
                filtered_df[channel] = apply_filter(filtered_df[channel].values, b, a, filter_method)
                write_log(f"Applied comb filter to {channel} (notch at {notch_freq} Hz)")
    
    elif filter_type == 'bandpass':
        # Bandpass filter
        low_freq = filter_config.get('low_freq', 0.1)
        high_freq = filter_config.get('high_freq', 10.0)
        order = filter_config.get('order', 4)
        
        b, a = design_bandpass_filter(fs, low_freq, high_freq, order)
        
        for channel in channels:
            if channel in filtered_df.columns:
                filtered_df[channel] = apply_filter(filtered_df[channel].values, b, a, filter_method)
                write_log(f"Applied bandpass filter to {channel} ({low_freq}-{high_freq} Hz)")
    
    elif filter_type == 'highpass':
        # Highpass filter
        cutoff_freq = filter_config.get('cutoff_freq', 0.1)
        order = filter_config.get('order', 4)
        
        b, a = design_highpass_filter(fs, cutoff_freq, order)
        
        for channel in channels:
            if channel in filtered_df.columns:
                filtered_df[channel] = apply_filter(filtered_df[channel].values, b, a, filter_method)
                write_log(f"Applied highpass filter to {channel} (cutoff: {cutoff_freq} Hz)")
    
    elif filter_type == 'lowpass':
        # Lowpass filter
        cutoff_freq = filter_config.get('cutoff_freq', 10.0)
        order = filter_config.get('order', 4)
        
        b, a = design_lowpass_filter(fs, cutoff_freq, order)
        
        for channel in channels:
            if channel in filtered_df.columns:
                filtered_df[channel] = apply_filter(filtered_df[channel].values, b, a, filter_method)
                write_log(f"Applied lowpass filter to {channel} (cutoff: {cutoff_freq} Hz)")
    
    elif filter_type == 'adaptive':
        # Adaptive filtering (requires reference signal)
        reference_channel = filter_config.get('reference_channel', 'rBx')
        filter_length = filter_config.get('filter_length', 64)
        mu = filter_config.get('mu', 0.01)
        
        if reference_channel in filtered_df.columns:
            for channel in channels:
                if channel in filtered_df.columns and channel != reference_channel:
                    filtered_df[channel] = adaptive_filter(
                        filtered_df[channel].values,
                        filtered_df[reference_channel].values,
                        filter_length,
                        mu
                    )
                    write_log(f"Applied adaptive filter to {channel} using {reference_channel} as reference")
        else:
            write_log(f"Reference channel {reference_channel} not found for adaptive filtering", level="WARNING")
    
    elif filter_type == 'custom':
        # Custom filter with user-provided coefficients
        b = filter_config.get('b', [1.0])
        a = filter_config.get('a', [1.0])
        
        for channel in channels:
            if channel in filtered_df.columns:
                filtered_df[channel] = apply_filter(filtered_df[channel].values, b, a, filter_method)
                write_log(f"Applied custom filter to {channel}")
    
    else:
        write_log(f"Unknown filter type: {filter_type}", level="WARNING")
        return df
    
    return filtered_df

def plot_filter_response(b, a, fs, filter_name="Filter", save_plot=False, output_dir="."):
    """Plot the frequency response of a filter.
    
    Args:
        b (np.ndarray): Numerator coefficients
        a (np.ndarray): Denominator coefficients
        fs (float): Sampling frequency in Hz
        filter_name (str): Name for the plot
        save_plot (bool): Whether to save the plot
        output_dir (str): Directory to save plots
    """
    try:
        from scipy.signal import freqz
        
        # Calculate frequency response
        w, h = freqz(b, a, worN=8000)
        f = w * fs / (2 * np.pi)
        
        # Convert to dB
        h_db = 20 * np.log10(abs(h))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Magnitude response
        ax1.semilogx(f, h_db)
        ax1.grid(True)
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Magnitude [dB]')
        ax1.set_title(f'{filter_name} - Magnitude Response')
        ax1.set_xlim(0.1, fs/2)
        ax1.set_ylim(-60, 5)
        
        # Phase response
        phase = np.unwrap(np.angle(h))
        ax2.semilogx(f, phase)
        ax2.grid(True)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Phase [radians]')
        ax2.set_title(f'{filter_name} - Phase Response')
        ax2.set_xlim(0.1, fs/2)
        
        plt.tight_layout()
        
        if save_plot:
            filename = os.path.join(output_dir, f"{filter_name.replace(' ', '_')}_response.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            write_log(f"Filter response plot saved as {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        write_log(f"Error plotting filter response: {e}", level="ERROR")


def plot_heatmaps(df, fs, save_plots=False, site_name="UnknownSite", output_dir=".", 
                  nperseg=1024, noverlap=None, coherence_thresholds=None):
    """Generate comprehensive heatmap plots for quality control and cultural noise detection.
    
    Creates three types of plots:
    1. Coherence heatmap (period vs. time, coherence as color)
    2. Window score barcode (quality indicators aligned with heatmap)
    3. Coherence histograms per frequency band
    
    Args:
        df (pd.DataFrame): DataFrame with channel data
        fs (float): Sampling frequency in Hz
        save_plots (bool): Whether to save plots to files
        site_name (str): Site name for plot titles and filenames
        output_dir (str): Directory to save plots in
        nperseg (int): Number of points per segment for FFT
        noverlap (int, optional): Number of points to overlap between segments
        coherence_thresholds (dict, optional): Custom coherence thresholds for quality scoring
    """
    try:
        from scipy.signal import coherence, spectrogram
        import matplotlib.colors as mcolors
        
        # Set default overlap if not specified
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Default coherence thresholds for quality scoring (green/amber/red)
        if coherence_thresholds is None:
            coherence_thresholds = {
                'good': 0.7,      # Green: coherence >= 0.7
                'fair': 0.5,      # Amber: 0.5 <= coherence < 0.7
                'poor': 0.5       # Red: coherence < 0.5
            }
        
        # Get start datetime from DataFrame index if available
        start_datetime = None
        if hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
            start_datetime = df.index[0]
        elif 'datetime' in df.columns:
            start_datetime = df['datetime'].iloc[0]
        
        # Define MT-specific channel pairs for coherence analysis
        mt_pairs = []
        if 'Bx' in df.columns and 'Ey' in df.columns:
            mt_pairs.append(('Bx', 'Ey', 'Bx-Ey'))
        if 'By' in df.columns and 'Ex' in df.columns:
            mt_pairs.append(('By', 'Ex', 'By-Ex'))
        
        # Add remote reference pairs if available
        if 'rBx' in df.columns:
            if 'Bx' in df.columns:
                mt_pairs.append(('Bx', 'rBx', 'Bx-rBx'))
            if 'Ex' in df.columns and 'rEx' in df.columns:
                mt_pairs.append(('Ex', 'rEx', 'Ex-rEx'))
        if 'rBy' in df.columns:
            if 'By' in df.columns:
                mt_pairs.append(('By', 'rBy', 'By-rBy'))
            if 'Ey' in df.columns and 'rEy' in df.columns:
                mt_pairs.append(('Ey', 'rEy', 'Ey-rEy'))
        
        if not mt_pairs:
            write_log(f"No valid MT channel pairs found for heatmap analysis for {site_name}", level="WARNING")
            return
        
        # Create figure with subplots for each pair
        n_pairs = len(mt_pairs)
        fig_height = 4 * n_pairs  # 4 inches per pair
        fig, axes = plt.subplots(n_pairs, 2, figsize=(12, fig_height))  # Changed from 3 to 2 columns
        
        # Handle single pair case
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for pair_idx, (ch1, ch2, pair_name) in enumerate(mt_pairs):
            if ch1 not in df.columns or ch2 not in df.columns:
                continue
            
            # Get data for this pair
            data1 = df[ch1].dropna()
            data2 = df[ch2].dropna()
            
            if len(data1) == 0 or len(data2) == 0:
                continue
            
            # Ensure both channels have the same length
            min_len = min(len(data1), len(data2))
            data1 = data1[:min_len]
            data2 = data2[:min_len]
            
            # Calculate time vector
            time_vector = np.arange(len(data1)) / fs
            
            # Calculate coherence over time using sliding windows
            coherence_matrix, time_windows, freq_vector = calculate_coherence_over_time(
                data1, data2, fs, nperseg, noverlap
            )
            
            # Convert frequency to period for MT analysis
            period_vector = 1.0 / freq_vector[1:]  # Skip DC component
            coherence_matrix = coherence_matrix[1:, :]  # Skip DC component
            
            # 1. Coherence Heatmap
            ax_heatmap = axes[pair_idx, 0]
            
            # Convert time axis to days for better readability
            time_days = time_windows / (24 * 3600)  # Convert seconds to days
            im = ax_heatmap.pcolormesh(time_days, period_vector, coherence_matrix, 
                                      cmap='RdYlGn_r', vmin=0, vmax=1, shading='gouraud')
            ax_heatmap.set_xlabel('Time [days]')
            ax_heatmap.set_xlim(time_days[0], time_days[-1])
            
            ax_heatmap.set_yscale('log')
            ax_heatmap.set_ylabel('Period [s]')
            ax_heatmap.set_title(f'{pair_name} Coherence Heatmap - {site_name}')
            ax_heatmap.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_heatmap)
            cbar.set_label('Coherence')
            
            # 2. Window Score Barcode (DISABLED)
            # ax_barcode = axes[pair_idx, 1]
            # window_scores = calculate_window_scores(coherence_matrix, coherence_thresholds)
            # plot_window_barcode(ax_barcode, time_windows, window_scores, pair_name, site_name, start_datetime)
            
            # 3. Coherence Histograms
            ax_hist = axes[pair_idx, 1]
            plot_coherence_histograms(ax_hist, coherence_matrix, period_vector, 
                                    coherence_thresholds, pair_name, site_name)
        
        plt.tight_layout()
        
        if save_plots:
            filename = os.path.join(output_dir, f"{site_name}_heatmaps.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            write_log(f"Heatmap plots saved as {filename}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        write_log(f"Error in plot_heatmaps: {e}", level="ERROR")
        write_log(f"Traceback: {traceback.format_exc()}", level="ERROR")


def calculate_coherence_over_time(data1, data2, fs, nperseg, noverlap):
    """Calculate coherence over time using sliding windows.
    
    Args:
        data1 (np.ndarray): First channel data
        data2 (np.ndarray): Second channel data
        fs (float): Sampling frequency
        nperseg (int): Number of points per segment
        noverlap (int): Number of points to overlap
        
    Returns:
        tuple: (coherence_matrix, time_windows, freq_vector)
    """
    from scipy.signal import coherence
    
    # Calculate number of windows
    step = nperseg - noverlap
    n_windows = (len(data1) - nperseg) // step + 1
    
    # Initialize arrays
    coherence_matrix = []
    time_windows = []
    
    # Calculate coherence for each window
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + nperseg
        
        if end_idx > len(data1):
            break
        
        # Extract window data
        window1 = data1[start_idx:end_idx]
        window2 = data2[start_idx:end_idx]
        
        # Calculate coherence for this window
        f, Cxy = coherence(window1, window2, fs=fs, nperseg=min(nperseg//2, len(window1)//2))
        
        coherence_matrix.append(Cxy)
        time_windows.append(start_idx / fs)  # Center time of window
    
    # Convert to numpy array
    coherence_matrix = np.array(coherence_matrix).T  # Transpose for (freq, time) format
    time_windows = np.array(time_windows)
    
    return coherence_matrix, time_windows, f


def calculate_window_scores(coherence_matrix, thresholds):
    """Calculate quality scores for each time window based on coherence.
    
    Args:
        coherence_matrix (np.ndarray): Coherence matrix (freq, time)
        thresholds (dict): Coherence thresholds for quality scoring
        
    Returns:
        np.ndarray: Quality scores (0=poor, 1=fair, 2=good)
    """
    # Calculate mean coherence across frequency bands for each time window
    mean_coherence = np.mean(coherence_matrix, axis=0)
    
    # Assign quality scores
    scores = np.zeros(len(mean_coherence), dtype=int)
    scores[mean_coherence >= thresholds['good']] = 2  # Good (green)
    scores[(mean_coherence >= thresholds['fair']) & (mean_coherence < thresholds['good'])] = 1  # Fair (amber)
    scores[mean_coherence < thresholds['poor']] = 0  # Poor (red)
    
    return scores


def plot_window_barcode(ax, time_windows, window_scores, pair_name, site_name, start_datetime=None):
    """Plot window score barcode showing quality over time.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        time_windows (np.ndarray): Time points for windows (in seconds from start)
        window_scores (np.ndarray): Quality scores (0=poor, 1=fair, 2=good)
        pair_name (str): Channel pair name
        site_name (str): Site name
        start_datetime (datetime, optional): Start datetime (not used, kept for compatibility)
    """
    # Define colors for quality levels
    colors = ['red', 'orange', 'green']
    
    # Calculate bar width based on time window spacing
    if len(time_windows) > 1:
        bar_width = time_windows[1] - time_windows[0]
    else:
        bar_width = 1.0
    
    # Create barcode plot
    for i, (time, score) in enumerate(zip(time_windows, window_scores)):
        if score < len(colors):  # Ensure score is valid
            ax.bar(time, 1, width=bar_width, color=colors[score], alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    # Convert time axis to days for better readability
    time_days = time_windows / (24 * 3600)  # Convert seconds to days
    ax.set_xlabel('Time [days]')
    ax.set_xlim(time_days[0], time_days[-1])
    
    ax.set_ylabel('Quality')
    ax.set_title(f'{pair_name} Window Quality Barcode - {site_name}')
    ax.set_ylim(0, 1.2)
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.8, label='Good (≥0.8)'),
        Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.8, label='Fair (0.6-0.8)'),
        Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.8, label='Poor (<0.6)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def plot_coherence_histograms(ax, coherence_matrix, period_vector, thresholds, pair_name, site_name):
    """Plot coherence histograms for different frequency bands.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        coherence_matrix (np.ndarray): Coherence matrix (freq, time)
        period_vector (np.ndarray): Period values
        thresholds (dict): Coherence thresholds
        pair_name (str): Channel pair name
        site_name (str): Site name
    """
    # Define period bands for MT analysis
    period_bands = [
        (0.001, 0.01, 'Ultra-high freq (0.001-0.01s)'),
        (0.01, 0.1, 'High freq (0.01-0.1s)'),
        (0.1, 1.0, 'Mid freq (0.1-1.0s)'),
        (1.0, 10.0, 'Low freq (1.0-10.0s)'),
        (10.0, 100.0, 'Ultra-low freq (10-100s)')
    ]
    
    # Plot histograms for each band
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (min_period, max_period, band_name) in enumerate(period_bands):
        # Find indices for this period band
        band_mask = (period_vector >= min_period) & (period_vector < max_period)
        
        if np.any(band_mask):
            # Extract coherence values for this band
            band_coherence = coherence_matrix[band_mask, :].flatten()
            
            # Plot histogram
            ax.hist(band_coherence, bins=20, alpha=0.6, color=colors[i], 
                   label=band_name, density=True)
    
    # Add threshold lines
    for threshold_name, threshold_value in thresholds.items():
        ax.axvline(threshold_value, color='black', linestyle='--', alpha=0.7,
                  label=f'{threshold_name.capitalize()} ({threshold_value})')
    
    ax.set_xlabel('Coherence')
    ax.set_ylabel('Density')
    ax.set_title(f'{pair_name} Coherence Histograms - {site_name}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def detect_cultural_noise(coherence_matrix, time_windows, period_vector, 
                         min_coherence_threshold=0.3, min_duration_seconds=60):
    """Detect potential cultural noise patterns in coherence data.
    
    Args:
        coherence_matrix (np.ndarray): Coherence matrix (freq, time)
        time_windows (np.ndarray): Time points for windows
        period_vector (np.ndarray): Period values
        min_coherence_threshold (float): Minimum coherence threshold for noise detection
        min_duration_seconds (float): Minimum duration for noise detection
        
    Returns:
        dict: Dictionary containing detected noise patterns
    """
    noise_patterns = {
        'low_coherence_regions': [],
        'vertical_stripes': [],
        'periodic_patterns': []
    }
    
    # Detect low coherence regions (potential cultural noise)
    low_coherence_mask = coherence_matrix < min_coherence_threshold
    
    # Find continuous low coherence regions in time
    for freq_idx in range(coherence_matrix.shape[0]):
        freq_low_mask = low_coherence_mask[freq_idx, :]
        
        # Find continuous regions
        regions = find_continuous_regions(freq_low_mask)
        
        for start_idx, end_idx in regions:
            start_time = time_windows[start_idx]
            end_time = time_windows[end_idx]
            duration = end_time - start_time
            
            if duration >= min_duration_seconds:
                period = period_vector[freq_idx]
                noise_patterns['low_coherence_regions'].append({
                    'period': period,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'frequency_idx': freq_idx
                })
    
    # Detect vertical stripes (cultural cycles)
    # Look for time periods with consistently low coherence across multiple frequencies
    time_low_coherence = np.mean(low_coherence_mask, axis=0)
    vertical_regions = find_continuous_regions(time_low_coherence > 0.5)  # 50% of frequencies show low coherence
    
    for start_idx, end_idx in vertical_regions:
        start_time = time_windows[start_idx]
        end_time = time_windows[end_idx]
        duration = end_time - start_time
        
        if duration >= min_duration_seconds:
            noise_patterns['vertical_stripes'].append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'severity': np.mean(time_low_coherence[start_idx:end_idx])
            })
    
    return noise_patterns


def find_continuous_regions(mask):
    """Find continuous regions where mask is True.
    
    Args:
        mask (np.ndarray): Boolean mask
        
    Returns:
        list: List of (start_idx, end_idx) tuples
    """
    regions = []
    in_region = False
    start_idx = None
    
    for i, value in enumerate(mask):
        if value and not in_region:
            in_region = True
            start_idx = i
        elif not value and in_region:
            regions.append((start_idx, i - 1))
            in_region = False
    
    # Handle case where region extends to end
    if in_region:
        regions.append((start_idx, len(mask) - 1))
    
    return regions


def generate_heatmap_report(coherence_matrix, time_windows, period_vector, 
                           noise_patterns, pair_name, site_name):
    """Generate a text report summarizing heatmap analysis results.
    
    Args:
        coherence_matrix (np.ndarray): Coherence matrix
        time_windows (np.ndarray): Time points
        period_vector (np.ndarray): Period values
        noise_patterns (dict): Detected noise patterns
        pair_name (str): Channel pair name
        pair_name (str): Site name
        
    Returns:
        str: Formatted report text
    """
    report = f"""
{'='*80}
HEATMAP ANALYSIS REPORT - {pair_name} - {site_name}
{'='*80}

DATA SUMMARY:
{'-'*40}
Total time windows: {len(time_windows)}
Time range: {time_windows[0]:.1f}s - {time_windows[-1]:.1f}s
Period range: {period_vector[0]:.3f}s - {period_vector[-1]:.1f}s
Frequency range: {1/period_vector[-1]:.1f} Hz - {1/period_vector[0]:.1f} Hz

COHERENCE STATISTICS:
{'-'*40}
Mean coherence: {np.mean(coherence_matrix):.3f}
Median coherence: {np.median(coherence_matrix):.3f}
Std coherence: {np.std(coherence_matrix):.3f}
Min coherence: {np.min(coherence_matrix):.3f}
Max coherence: {np.max(coherence_matrix):.3f}

QUALITY ASSESSMENT:
{'-'*40}"""
    
    # Calculate quality percentages
    mean_coherence = np.mean(coherence_matrix, axis=0)
    good_pct = np.sum(mean_coherence >= 0.8) / len(mean_coherence) * 100
    fair_pct = np.sum((mean_coherence >= 0.6) & (mean_coherence < 0.8)) / len(mean_coherence) * 100
    poor_pct = np.sum(mean_coherence < 0.6) / len(mean_coherence) * 100
    
    report += f"""
Good quality windows (≥0.8): {good_pct:.1f}%
Fair quality windows (0.6-0.8): {fair_pct:.1f}%
Poor quality windows (<0.6): {poor_pct:.1f}%

NOISE DETECTION RESULTS:
{'-'*40}"""
    
    if noise_patterns['low_coherence_regions']:
        report += f"\nLow coherence regions detected: {len(noise_patterns['low_coherence_regions'])}"
        for i, region in enumerate(noise_patterns['low_coherence_regions'][:5]):  # Show first 5
            report += f"\n  Region {i+1}: Period={region['period']:.3f}s, "
            report += f"Time={region['start_time']:.1f}s-{region['end_time']:.1f}s, "
            report += f"Duration={region['duration']:.1f}s"
        if len(noise_patterns['low_coherence_regions']) > 5:
            report += f"\n  ... and {len(noise_patterns['low_coherence_regions']) - 5} more regions"
    else:
        report += "\nNo significant low coherence regions detected"
    
    if noise_patterns['vertical_stripes']:
        report += f"\n\nVertical stripes (cultural cycles) detected: {len(noise_patterns['vertical_stripes'])}"
        for i, stripe in enumerate(noise_patterns['vertical_stripes'][:3]):  # Show first 3
            report += f"\n  Stripe {i+1}: Time={stripe['start_time']:.1f}s-{stripe['end_time']:.1f}s, "
            report += f"Duration={stripe['duration']:.1f}s, Severity={stripe['severity']:.2f}"
        if len(noise_patterns['vertical_stripes']) > 3:
            report += f"\n  ... and {len(noise_patterns['vertical_stripes']) - 3} more stripes"
    else:
        report += "\nNo significant vertical stripes detected"
    
    report += f"\n\n{'='*80}\n"
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MT ASCII data files for a given site.")
    parser.add_argument("--site_name", required=True, help="Site name (directory containing ASCII data files).")
    parser.add_argument("--param_file", default="config/recorder.ini", help="Path to the parameter file (default: config/recorder.ini)")
    parser.add_argument("--average", action="store_true", help="Average values over intervals.")
    parser.add_argument("--perform_freq_analysis", nargs="?", const="W", metavar="METHODS",
                        help="Perform frequency analysis. Use W for Welch, M for Multi-taper, S for Spectrogram. "
                             "Combine them: 'WMS' for all three, 'WM' for Welch + Multi-taper, etc. Default: W")
    parser.add_argument("--plot_timeseries", action="store_true", help="Display physical channel plots.")
    parser.add_argument("--apply_smoothing", action="store_true", help="Apply smoothing to the data.")
    parser.add_argument("--smoothing_window", type=int, default=2500, help="Window size for smoothing.")
    parser.add_argument("--threshold_factor", type=float, default=10.0, help="Threshold multiplier for outlier detection.")
    parser.add_argument("--plot_boundaries", action="store_true", help="Plot file boundaries.")
    parser.add_argument("--plot_smoothed_windows", action="store_true", help="Shade smoothed windows.")
    parser.add_argument("--plot_coherence", action="store_true", help="Plot power spectra and coherence.")
    parser.add_argument("--log_first_rows", action="store_true", help="Log the first 5 rows from each file.")
    parser.add_argument("--apply_drift_correction", action="store_true", help="Apply drift correction to the data.")
    parser.add_argument("--apply_rotation", action="store_true", help="Apply rotation correction based on the 'erotate' parameter.")
    parser.add_argument("--plot_drift", action="store_true", help="Plot drift-corrected data in timeseries (default: False)")
    parser.add_argument("--plot_rotation", action="store_true", help="Plot rotated data in timeseries (default: False)")
    parser.add_argument("--plot_tilt", action="store_true", help="Plot tilt-corrected data in timeseries (default: False)")
    parser.add_argument("--timezone", nargs="+", default=["UTC"], 
                        help="Timezone(s) for the data. Use one timezone for both sites (e.g., 'Australia/Adelaide') or two timezones for main and remote sites (e.g., 'Australia/Adelaide' 'UTC')")
    parser.add_argument("--plot_original_data", action="store_true", help="Plot original data alongside corrected data.")
    parser.add_argument("--save_raw_data", action="store_true", help="Save raw data to file.")
    parser.add_argument("--remote_reference", help="Remote reference site name for processing.")
    parser.add_argument("--apply_filtering", action="store_true", help="Apply frequency filtering to the data.")
    parser.add_argument("--filter_type", choices=["comb", "bandpass", "highpass", "lowpass", "adaptive"], 
                        default="comb", help="Type of filter to apply.")
    parser.add_argument("--filter_channels", nargs="+", help="Specific channels to filter (default: all channels).")
    parser.add_argument("--filter_notch_freq", type=float, default=50.0, help="Notch frequency for comb filter (Hz).")
    parser.add_argument("--filter_quality_factor", type=float, default=30.0, help="Quality factor for comb filter.")
    parser.add_argument("--filter_harmonics", nargs="+", type=float, help="Specific harmonics to notch (default: auto-detect).")
    parser.add_argument("--filter_low_freq", type=float, help="Low frequency cutoff for bandpass filter (Hz).")
    parser.add_argument("--filter_high_freq", type=float, help="High frequency cutoff for bandpass filter (Hz).")
    parser.add_argument("--filter_cutoff_freq", type=float, help="Cutoff frequency for highpass/lowpass filter (Hz).")
    parser.add_argument("--filter_order", type=int, default=4, help="Filter order (higher = sharper cutoff).")
    parser.add_argument("--filter_reference_channel", help="Reference channel for adaptive filtering.")
    parser.add_argument("--filter_length", type=int, default=64, help="Filter length for adaptive filtering.")
    parser.add_argument("--filter_mu", type=float, default=0.01, help="Step size for adaptive filtering.")
    parser.add_argument("--filter_method", choices=["filtfilt", "lfilter"], default="filtfilt", 
                        help="Filtering method (filtfilt=zero-phase, lfilter=causal).")
    parser.add_argument("--plot_heatmaps", action="store_true", help="Generate coherence heatmaps for quality control.")
    parser.add_argument("--heatmap_nperseg", type=int, default=1024, help="FFT segment length for heatmaps.")
    parser.add_argument("--heatmap_noverlap", type=int, help="Overlap between FFT segments for heatmaps.")
    parser.add_argument("--heatmap_thresholds", help="Comma-separated coherence thresholds for heatmaps (e.g., '0.9,0.7,0.7').")
    parser.add_argument("--smoothing_method", choices=["median", "adaptive"], default="median", 
                        help="Smoothing method to use.")
    parser.add_argument("--sens_start", type=int, default=0, help="Start sensor index for processing.")
    parser.add_argument("--sens_end", type=int, default=5000, help="End sensor index for processing.")
    parser.add_argument("--skip_minutes", nargs=2, type=int, default=[0, 0], 
                        help="Skip first and last N minutes of data (e.g., --skip_minutes 10 20).")
    parser.add_argument("--tilt_correction", action="store_true", help="Apply tilt correction to make mean(By) = 0.")
    parser.add_argument("--decimate", nargs="+", type=int, help="Decimation factors to apply (e.g., --decimate 2 5 10).")
    parser.add_argument("--run_lemimt", action="store_true", help="Run lemimt processing after data processing.")
    parser.add_argument("--lemimt_path", default="lemimt.exe", help="Path to lemimt executable.")
    
    args = parser.parse_args()
    
    # Set up logging
    set_log_level("INFO")
    set_batch_mode(False)
    set_site_name(args.site_name)
    
    write_log(f"Starting processing for site: {args.site_name}")
    
    # Determine decimation factors to process
    if args.decimate:
        decimation_factors = [1] + args.decimate  # Always include original (factor 1)
        write_log(f"Processing with decimation factors: {decimation_factors}")
    else:
        decimation_factors = [1]  # Just process original data
        write_log("Processing original data only (no decimation)")

    # --- Two-letter identifier system ---
    # Determine the run letter for this script run (use first decimation's output as base)
    sample_interval = 0.1  # Default, will be updated below
    # Build base_site_name for identifier search
    base_site_name = f"MT_{args.site_name}"
    if args.remote_reference:
        base_site_name += f"_RR-{args.remote_reference}"
    # Use the highest sample rate for the base name (original data)
    if hasattr(args, 'decimate') and args.decimate:
        sample_rate = 1.0 / sample_interval
        freq_str = f"_{int(sample_rate)}Hz" if sample_rate >= 1 else f"_{sample_rate:.1f}Hz"
        base_site_name += freq_str
    else:
        base_site_name += "_10Hz"
    run_letter = get_next_run_letter(base_site_name, ".")
    # --- End identifier setup ---

    # Process each decimation factor
    for file_index, decimation_factor in enumerate(decimation_factors):
        write_log(f"Processing {args.site_name} with decimation factor: {decimation_factor}")
        if args.remote_reference:
            write_log(f"Using remote reference: {args.remote_reference}")
        
        # Create processor for this decimation factor
        processor = ProcessASCII(
            input_dir=args.site_name,  # Use site_name as directory
            param_file=args.param_file,
            average=args.average,
            perform_freq_analysis=args.perform_freq_analysis,
            plot_timeseries=args.plot_timeseries,
            apply_smoothing=args.apply_smoothing,
            smoothing_window=args.smoothing_window,
            threshold_factor=args.threshold_factor,
            plot_boundaries=args.plot_boundaries,
            plot_smoothed_windows=args.plot_smoothed_windows,
            plot_coherence=args.plot_coherence,
            log_first_rows=args.log_first_rows,
            smoothing_method=args.smoothing_method,
            sens_start=args.sens_start,
            sens_end=args.sens_end,
            skip_minutes=args.skip_minutes,
            apply_drift_correction=args.apply_drift_correction,
            apply_rotation=args.apply_rotation,
            plot_drift=args.plot_drift,
            plot_rotation=args.plot_rotation,
            plot_tilt=args.plot_tilt,
            timezone=args.timezone,
            plot_original_data=args.plot_original_data,
            save_raw_data=args.save_raw_data,
            save_processed_data=True,  # Always save processed data
            remote_reference=args.remote_reference,
            apply_filtering=args.apply_filtering,
            filter_type=args.filter_type,
            filter_channels=args.filter_channels,
            filter_params={
                'notch_freq': args.filter_notch_freq,
                'quality_factor': args.filter_quality_factor,
                'harmonics': args.filter_harmonics,
                'low_freq': args.filter_low_freq,
                'high_freq': args.filter_high_freq,
                'cutoff_freq': args.filter_cutoff_freq,
                'order': args.filter_order,
                'reference_channel': args.filter_reference_channel,
                'filter_length': args.filter_length,
                'mu': args.filter_mu,
                'method': args.filter_method
            },
            plot_heatmaps=args.plot_heatmaps,
            heatmap_nperseg=args.heatmap_nperseg,
            heatmap_noverlap=args.heatmap_noverlap,
            heatmap_thresholds=args.heatmap_thresholds
        )
        
        # Set tilt correction to always apply to all Bx/By channels
        processor.tilt_correction = args.tilt_correction
        
        # Always save plots when plotting
        processor.save_plots = args.plot_timeseries
        
        # Store decimation factor for use in processing
        processor.decimation_factor = decimation_factor
        
        # Set lemimt parameters
        processor.run_lemimt = args.run_lemimt
        processor.lemimt_path = args.lemimt_path
        
        # Process the data
        processor.process_all_files()
        
        # Run lemimt processing if requested
        if args.run_lemimt:
            # Determine the filename based on decimation factor using new naming convention
            cpu_prefix = get_cpu_prefix()
            if decimation_factor == 1:
                processed_file = os.path.join(".", f"{cpu_prefix}_{args.site_name}_10Hz_Process.txt")
            else:
                sample_interval = processor.metadata.get("sample_interval", 0.1)
                suffix = get_sample_rate_suffix(sample_interval * decimation_factor)
                processed_file = os.path.join(".", f"{cpu_prefix}_{args.site_name}{suffix}_Process.txt")
            # Check if the file exists, if not try alternative naming patterns
            if not os.path.exists(processed_file):
                import glob
                pattern = os.path.join(".", f"{cpu_prefix}_{args.site_name}*_Process.txt")
                matching_files = glob.glob(pattern)
                if matching_files:
                    processed_file = matching_files[0]
                    write_log(f"Found processed file for lemimt: {processed_file}")
                else:
                    write_log(f"No processed files found matching pattern: {pattern}", level="WARNING")
                    continue
            if os.path.exists(processed_file):
                write_log(f"Running lemimt processing on: {processed_file}")
                try:
                    gps_data = processor.gps_data if hasattr(processor, 'gps_data') and processor.gps_data else {}
                    sample_interval = processor.metadata.get("sample_interval", 0.1)
                    has_remote_reference = processor.remote_reference is not None
                    remote_reference_site = processor.remote_reference
                    if gps_data and 'latitude' in gps_data and 'longitude' in gps_data:
                        write_log(f"Using GPS coordinates for config: Lat={gps_data['latitude']:.6f}°, Lon={gps_data['longitude']:.6f}°")
                    else:
                        write_log(f"WARNING: No valid GPS data available for config generation, using defaults", level="WARNING")
                        write_log(f"GPS data available: {list(gps_data.keys()) if gps_data else 'None'}")
                    config_file = generate_processing_config(
                        site_name=args.site_name,
                        gps_data=gps_data,
                        sample_interval=sample_interval,
                        has_remote_reference=has_remote_reference,
                        remote_reference_site=remote_reference_site,
                        output_file_path=processed_file,
                        run_index=run_letter,
                        file_index=file_index
                    )
                    write_log(f"Generated config file for lemimt: {config_file}")
                except Exception as e:
                    write_log(f"Warning: Could not generate config file: {e}", level="WARNING")
                success = processor.run_lemimt_processing(processed_file)
                if success:
                    write_log(f"lemimt processing completed successfully for {processed_file}")
                else:
                    write_log(f"lemimt processing failed for {processed_file}", level="WARNING")
            else:
                write_log(f"Processed file not found for lemimt processing: {processed_file}", level="WARNING")
        write_log(f"Completed processing for decimation factor: {decimation_factor}")
    write_log("All decimation processing completed")

# NOTES ON PROCESSING.TXT CONFIGURATION:
# - Use EDL_Batch.py --input_config to process all sites from Processing.txt
# - The file should be in the format: Site, xarm, yarm, [remote_ref1; remote_ref2]
# - Multiple remote references in brackets will create multiple processing runs
# - Dipole lengths (xarm, yarm) are automatically applied from the configuration
# - Remote references are automatically loaded and processed
# - Each remote reference creates a separate output file with unique CPU prefix
# - EDL_Process.py is focused on single-site processing only

# EXAMPLE USAGE:
# 
# Single site processing:
# python EDL_Process.py --site_name SiteA-50m --plot_timeseries --tilt_correction
# python EDL_Process.py --site_name SiteB-50m --remote_reference SiteD-10m --plot_timeseries
# python EDL_Process.py --site_name SiteC-10m --apply_smoothing --plot_timeseries --run_lemimt
# 
# Batch processing (multiple sites):
# python EDL_Batch.py --input_config --max_workers 4 --plot_timeseries --tilt_correction
# python EDL_Batch.py --input_config --max_workers 2 --run_lemimt
# 
# Advanced processing:
# python EDL_Process.py --site_name SiteA-50m --plot_timeseries --apply_filtering --filter_type comb --filter_notch_freq 50.0
# python EDL_Process.py --site_name SiteB-50m --plot_heatmaps --heatmap_thresholds "0.9,0.7,0.5"
# python EDL_Process.py --site_name SiteC-10m --decimate 2 5 10 --plot_timeseries
# 
# Processing.txt format example:
# SiteA-50m,100.0,100.0,SiteD-10m
# SiteB-50m,100.0,100.0,[SiteC-10m; SiteD-10m]
# SiteC-10m,50.0,50.0,
# SiteD-10m,50.0,50.0,SiteA-50m