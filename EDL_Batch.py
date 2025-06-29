#!/usr/bin/env python3
"""
EDL Batch Processing Script

This script processes multiple EDL sites in parallel using multiprocessing.
For each site, a ProcessASCII instance (from EDL_Process.py) is created and run in parallel
to handle the processing of ASCII data files.

Usage:
    python EDL_Batch.py --sites HDD5449,HDD5456,HDD5470 --parent_dir . --apply_drift_correction --apply_rotation --tilt_correction --plot_data --save_plots

Author: [Your Name]
Date: [Date]
"""

import argparse
import os
import sys
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the processing module
from EDL_Process import ProcessASCII, write_log, write_batch_log, set_batch_mode, set_log_level, create_batch_summary

def process_single_site(site, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                        plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        save_processed_data, remote_reference=None, apply_filtering=False,
                        filter_type="comb", filter_channels=None, filter_params=None,
                        plot_heatmaps=False, heatmap_nperseg=1024, heatmap_noverlap=None, heatmap_thresholds=None):
    """
    Processes a single site folder.
    
    Args:
        site (str): Site folder name.
        parent_dir (str): Parent directory containing site folders.
        apply_drift_correction (bool): Whether to apply drift correction.
        apply_rotation (bool): Whether to apply rotation.
        tilt_correction (bool): Whether to apply tilt correction.
        apply_smoothing (bool): Whether to apply smoothing.
        smoothing_method (str): Smoothing method to use.
        smoothing_window (int): Window size for smoothing.
        threshold_factor (float): Threshold factor for outlier detection.
        plot_data (bool): Whether to plot data.
        plot_boundaries (bool): Whether to plot boundaries.
        plot_smoothed_windows (bool): Whether to plot smoothed windows.
        plot_coherence (bool): Whether to plot coherence.
        perform_freq_analysis (bool): Whether to perform frequency analysis.
        log_first_rows (bool): Whether to log first rows.
        sens_start (int): Start sample for sensitivity test.
        sens_end (int): End sample for sensitivity test.
        skip_minutes (list): Minutes to skip from start and end.
        save_plots (bool): Whether to save plots.
        plot_drift (bool): Whether to plot drift correction.
        plot_rotation (bool): Whether to plot rotation.
        plot_tilt (bool): Whether to plot tilt correction.
        timezone (str): Timezone for the data.
        plot_original_data (bool): Whether to plot original data.
        save_processed_data (bool): Whether to save processed data.
        remote_reference (str, optional): Remote reference site name.
        apply_filtering (bool): Whether to apply frequency filtering.
        filter_type (str): Type of filter to apply.
        filter_channels (list, optional): Channels to filter.
        filter_params (dict, optional): Additional filter parameters.
        plot_heatmaps (bool): Whether to generate heatmap plots.
        heatmap_nperseg (int): Number of points per segment for heatmap FFT.
        heatmap_noverlap (int, optional): Number of points to overlap between heatmap segments.
        heatmap_thresholds (dict, optional): Custom coherence thresholds for heatmap quality scoring.
    
    Returns:
        dict: Results dictionary.
    """
    try:
        site_dir = os.path.join(parent_dir, site)
        if not os.path.exists(site_dir):
            write_batch_log(f"Site directory not found: {site_dir}", level="ERROR")
            return {"status": "FAILED", "error": f"Directory not found: {site_dir}"}
        
        write_batch_log(f"Processing site: {site}")
        
        # Create processor
        processor = ProcessASCII(
            input_dir=site_dir,
            param_file="config/recorder.ini",
            average=False,
            perform_freq_analysis=perform_freq_analysis,
            plot_data=plot_data,
            apply_smoothing=apply_smoothing,
            smoothing_window=smoothing_window,
            threshold_factor=threshold_factor,
            plot_boundaries=plot_boundaries,
            plot_smoothed_windows=plot_smoothed_windows,
            plot_coherence=plot_coherence,
            log_first_rows=log_first_rows,
            smoothing_method=smoothing_method,
            sens_start=sens_start,
            sens_end=sens_end,
            skip_minutes=skip_minutes,
            apply_drift_correction=apply_drift_correction,
            apply_rotation=apply_rotation,
            plot_drift=plot_drift,
            plot_rotation=plot_rotation,
            plot_tilt=plot_tilt,
            timezone=timezone,
            plot_original_data=plot_original_data,
            save_raw_data=False,
            save_processed_data=save_processed_data,
            remote_reference=remote_reference,
            apply_filtering=apply_filtering,
            filter_type=filter_type,
            filter_channels=filter_channels,
            filter_params=filter_params,
            plot_heatmaps=plot_heatmaps,
            heatmap_nperseg=heatmap_nperseg,
            heatmap_noverlap=heatmap_noverlap,
            heatmap_thresholds=heatmap_thresholds
        )
        
        processor.tilt_correction = tilt_correction
        processor.save_plots = save_plots
        
        # Process the site
        processor.process_all_files()
        
        return processor.results
        
    except Exception as e:
        error_msg = f"Error processing site {site}: {e}"
        write_batch_log(error_msg, level="ERROR")
        return {"status": "FAILED", "error": str(e)}


def batch_process_sites(sites, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                        max_workers, plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        save_processed_data, remote_reference=None, apply_filtering=False,
                        filter_type="comb", filter_channels=None, filter_params=None,
                        plot_heatmaps=False, heatmap_nperseg=1024, heatmap_noverlap=None, heatmap_thresholds=None):
    """
    Processes multiple MT sites in parallel.
    
    Args:
        sites (list): List of site folder names.
        parent_dir (str): Parent directory containing site folders.
        apply_drift_correction (bool): Whether to apply drift correction.
        apply_rotation (bool): Whether to apply rotation.
        tilt_correction (bool): Whether to apply tilt correction.
        apply_smoothing (bool): Whether to apply smoothing.
        smoothing_method (str): Smoothing method to use.
        smoothing_window (int): Window size for smoothing.
        threshold_factor (float): Threshold factor for outlier detection.
        plot_data (bool): Whether to plot data.
        plot_boundaries (bool): Whether to plot boundaries.
        plot_smoothed_windows (bool): Whether to plot smoothed windows.
        plot_coherence (bool): Whether to plot coherence.
        perform_freq_analysis (bool): Whether to perform frequency analysis.
        log_first_rows (bool): Whether to log first rows.
        sens_start (int): Start sample for sensitivity test.
        sens_end (int): End sample for sensitivity test.
        skip_minutes (list): Minutes to skip from start and end.
        save_plots (bool): Whether to save plots.
        max_workers (int): Maximum number of parallel workers.
        plot_drift (bool): Whether to plot drift correction.
        plot_rotation (bool): Whether to plot rotation.
        plot_tilt (bool): Whether to plot tilt correction.
        timezone (str): Timezone for the data.
        plot_original_data (bool): Whether to plot original data.
        save_processed_data (bool): Whether to save processed data.
        remote_reference (str, optional): Remote reference site name.
        apply_filtering (bool): Whether to apply frequency filtering.
        filter_type (str): Type of filter to apply.
        filter_channels (list, optional): Channels to filter.
        filter_params (dict, optional): Additional filter parameters.
        plot_heatmaps (bool): Whether to generate heatmap plots.
        heatmap_nperseg (int): Number of points per segment for heatmap FFT.
        heatmap_noverlap (int, optional): Number of points to overlap between heatmap segments.
        heatmap_thresholds (dict, optional): Custom coherence thresholds for heatmap quality scoring.
    
    Returns:
        dict: Dictionary mapping site names to results.
    """
    start_time = datetime.datetime.now()
    write_batch_log(f"Starting batch processing of {len(sites)} sites at {start_time}")
    
    results = {}
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_site = {}
        for site in sites:
            future = executor.submit(
                process_single_site,
                site, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                apply_smoothing, smoothing_method, smoothing_window,
                threshold_factor, plot_data, plot_boundaries,
                plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                save_processed_data, remote_reference, apply_filtering,
                filter_type, filter_channels, filter_params,
                plot_heatmaps, heatmap_nperseg, heatmap_noverlap, heatmap_thresholds
            )
            future_to_site[future] = site
        
        # Collect results as they complete
        for future in as_completed(future_to_site):
            site = future_to_site[future]
            try:
                result = future.result()
                results[site] = result
                status = result.get('status', 'UNKNOWN')
                write_batch_log(f"Completed {site}: {status}")
            except Exception as e:
                error_msg = f"Exception occurred while processing {site}: {e}"
                write_batch_log(error_msg, level="ERROR")
                results[site] = {'status': 'FAILED', 'error': str(e)}
    
    # Create and display batch summary
    create_batch_summary(sites, results, start_time)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process multiple MT sites contained in individual folders in parallel."
    )
    parser.add_argument("--parent_dir", required=True,
                        help="Parent directory containing MT site folders.")
    parser.add_argument("--sites", nargs="+", required=True,
                        help="List of site folder names to process.")
    parser.add_argument("--apply_drift_correction", action="store_true",
                        help="Apply drift correction to the data.")
    parser.add_argument("--apply_rotation", action="store_true",
                        help="Apply rotation correction based on the 'erotate' parameter.")
    parser.add_argument("--tilt_correction", action="store_true",
                        help="Apply tilt correction so that mean(By) is zero.")
    parser.add_argument("--apply_smoothing", action="store_true",
                        help="Apply smoothing to the data.")
    # Only "median" and "adaptive" are allowed smoothing methods now.
    parser.add_argument("--smoothing_method", default="median",
                        choices=["median", "adaptive"],
                        help="Smoothing method to use: 'median' for median/MAD or 'adaptive' for adaptive median filtering.")
    parser.add_argument("--smoothing_window", type=int, default=2500,
                        help="Window size for smoothing.")
    parser.add_argument("--threshold_factor", type=float, default=10.0,
                        help="Threshold multiplier for outlier detection (for median method).")
    parser.add_argument("--plot_data", action="store_true",
                        help="Plot physical channel data.")
    parser.add_argument("--plot_boundaries", action="store_true",
                        help="Plot file boundaries on the plots.")
    parser.add_argument("--plot_smoothed_windows", action="store_true",
                        help="Shade smoothed windows in the plots.")
    parser.add_argument("--plot_coherence", action="store_true",
                        help="Plot power spectra and coherence.")
    parser.add_argument("--perform_freq_analysis", action="store_true",
                        help="Perform frequency analysis on the data.")
    parser.add_argument("--log_first_rows", action="store_true",
                        help="Log the first 5 rows of data from each binary file.")
    parser.add_argument("--sens_start", type=int, default=0,
                        help="(Unused) Start sample index for sensitivity test.")
    parser.add_argument("--sens_end", type=int, default=5000,
                        help="(Unused) End sample index for sensitivity test.")
    parser.add_argument("--skip_minutes", nargs=2, type=int, default=[0, 0], metavar=('START', 'END'),
                        help="Minutes to skip from start and end of data (e.g., '30 20' skips first 30 and last 20 minutes)")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save plots to files instead of displaying them.")
    parser.add_argument("--save_processed_data", action="store_true", default=False,
                        help="Save processed data to file")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of parallel processes to use.")
    parser.add_argument("--plot_drift", action="store_true",
                        help="Whether to plot drift-corrected data in timeseries.")
    parser.add_argument("--plot_rotation", action="store_true",
                        help="Whether to plot rotated data in timeseries.")
    parser.add_argument("--plot_tilt", action="store_true",
                        help="Whether to plot tilt-corrected data in timeseries.")
    parser.add_argument("--timezone", default="UTC",
                        help="Timezone for the data (default: UTC).")
    parser.add_argument("--plot_original_data", action="store_true",
                        help="Plot original data alongside corrected data (default: False).")
    parser.add_argument("--remote_reference", type=str, default=None,
                        help="Site name to use as remote reference (only Bx and By will be loaded and merged as rBx, rBy)")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level for output (default: INFO)")
    parser.add_argument("--apply_filtering", action="store_true", default=False,
                        help="Apply frequency filtering to remove unwanted frequency components")
    parser.add_argument("--filter_type", type=str, default="comb", 
                        choices=["comb", "bandpass", "highpass", "lowpass", "adaptive", "custom"],
                        help="Type of filter to apply (default: comb for powerline noise removal)")
    parser.add_argument("--filter_channels", nargs="+", default=None,
                        help="Channels to filter (default: all main channels Bx, By, Bz, Ex, Ey)")
    parser.add_argument("--filter_notch_freq", type=float, default=50.0,
                        help="Notch frequency for comb filter (default: 50 Hz for powerline)")
    parser.add_argument("--filter_quality_factor", type=float, default=30.0,
                        help="Quality factor for comb filter (higher = narrower notch, default: 30)")
    parser.add_argument("--filter_harmonics", nargs="+", type=float, default=None,
                        help="Specific harmonic frequencies to notch (default: first 5 harmonics)")
    parser.add_argument("--filter_low_freq", type=float, default=0.1,
                        help="Lower cutoff frequency for bandpass filter (default: 0.1 Hz)")
    parser.add_argument("--filter_high_freq", type=float, default=10.0,
                        help="Upper cutoff frequency for bandpass filter (default: 10.0 Hz)")
    parser.add_argument("--filter_cutoff_freq", type=float, default=0.1,
                        help="Cutoff frequency for highpass/lowpass filters (default: 0.1 Hz)")
    parser.add_argument("--filter_order", type=int, default=4,
                        help="Filter order for Butterworth filters (default: 4)")
    parser.add_argument("--filter_reference_channel", type=str, default="rBx",
                        help="Reference channel for adaptive filtering (default: rBx)")
    parser.add_argument("--filter_length", type=int, default=64,
                        help="Filter length for adaptive filtering (default: 64)")
    parser.add_argument("--filter_mu", type=float, default=0.01,
                        help="Step size for adaptive filtering (default: 0.01)")
    parser.add_argument("--filter_method", type=str, default="filtfilt",
                        choices=["filtfilt", "lfilter"],
                        help="Filtering method: filtfilt (zero-phase) or lfilter (causal, default: filtfilt)")
    parser.add_argument("--plot_heatmaps", action="store_true", default=False,
                        help="Generate heatmap plots for quality control and cultural noise detection")
    parser.add_argument("--heatmap_nperseg", type=int, default=1024,
                        help="Number of points per segment for heatmap FFT (default: 1024)")
    parser.add_argument("--heatmap_noverlap", type=int, default=None,
                        help="Number of points to overlap between heatmap segments (default: nperseg//2)")
    parser.add_argument("--heatmap_thresholds", type=str, default=None,
                        help="Custom coherence thresholds for heatmap quality scoring (format: 'good,fair,poor' e.g., '0.8,0.6,0.6')")

    args = parser.parse_args()
    
    # Set up logging
    set_batch_mode(True)
    set_log_level(args.log_level)
    
    # Prepare filter parameters
    filter_params = {
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
    }
    
    # Process sites
    batch_process_sites(
        args.sites, args.parent_dir, args.apply_drift_correction, args.apply_rotation, args.tilt_correction,
        args.apply_smoothing, args.smoothing_method, args.smoothing_window, args.threshold_factor,
        args.plot_data, args.plot_boundaries, args.plot_smoothed_windows, args.plot_coherence,
        args.perform_freq_analysis, args.log_first_rows, args.sens_start, args.sens_end,
        args.skip_minutes, args.save_plots, args.max_workers, args.plot_drift, args.plot_rotation,
        args.plot_tilt, args.timezone, args.plot_original_data, args.save_processed_data, args.remote_reference,
        args.apply_filtering, args.filter_type, args.filter_channels, filter_params,
        args.plot_heatmaps, args.heatmap_nperseg, args.heatmap_noverlap, args.heatmap_thresholds
    )

"""
To run: 

# Basic processing with rotation and tilt correction:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_rotation --tilt_correction --apply_smoothing --plot_boundaries --plot_smoothed_windows --smoothing_method median --save_plots --plot_data --plot_boundaries

# Processing with drift correction and rotation:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --max_workers 4 --apply_drift_correction --apply_rotation --tilt_correction --apply_smoothing --skip_minutes 20 --plot_data --plot_boundaries --plot_smoothed_windows --smoothing_method median --save_plots

# Processing with drift correction and rotation, plotting both uncorrected and corrected data:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --max_workers 4 --apply_drift_correction --apply_rotation --plot_drift --plot_rotation --plot_data --save_plots

# Processing with tilt correction, plotting both uncorrected and tilt-corrected data:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --max_workers 4 --tilt_correction --plot_tilt --plot_data --save_plots

# Basic processing without any corrections:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_data --save_plots

# Processing with timezone conversion (e.g., to local time):
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --timezone "Australia/Sydney" --plot_data --save_plots

# Processing with timezone conversion and all corrections:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --timezone "America/New_York" --apply_drift_correction --apply_rotation --tilt_correction --plot_drift --plot_rotation --plot_tilt --plot_data --save_plots

# Example command with all options:
# python EDL_Batch.py --input_dir . --apply_drift_correction --apply_rotation --tilt_correction --apply_smoothing --plot_boundaries --plot_smoothed_windows --smoothing_method median --skip_minutes 10 5 --save_plots --max_workers 4

# FREQUENCY FILTERING EXAMPLES
# ============================

# Basic comb filter for powerline noise removal (50 Hz and harmonics):
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type comb --plot_data --save_plots --save_processed_data

# Comb filter with custom notch frequency (60 Hz for US powerline):
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type comb --filter_notch_freq 60.0 --plot_data --save_plots --save_processed_data

# Bandpass filter for specific frequency range:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type bandpass --filter_low_freq 0.01 --filter_high_freq 1.0 --plot_data --save_plots --save_processed_data

# Highpass filter to remove DC and low-frequency drift:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type highpass --filter_cutoff_freq 0.001 --plot_data --save_plots --save_processed_data

# Lowpass filter to remove high-frequency noise:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type lowpass --filter_cutoff_freq 5.0 --plot_data --save_plots --save_processed_data

# Filter specific channels only:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type comb --filter_channels Bx By --plot_data --save_plots --save_processed_data

# Adaptive filtering using remote reference:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --remote_reference HDD5974 --apply_filtering --filter_type adaptive --filter_reference_channel rBx --plot_data --save_plots --save_processed_data

# Filtering with other corrections:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type comb --apply_drift_correction --apply_rotation --tilt_correction --plot_data --save_plots --save_processed_data

# Filtering with smoothing:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type comb --apply_smoothing --smoothing_method median --plot_data --save_plots --save_processed_data

# HEATMAP EXAMPLES
# =================

# Basic heatmap generation for quality control:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_heatmaps --save_plots --save_processed_data

# Heatmap with custom FFT parameters:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_heatmaps --heatmap_nperseg 2048 --heatmap_noverlap 1024 --save_plots --save_processed_data

# Heatmap with custom coherence thresholds:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_heatmaps --heatmap_thresholds "0.9,0.7,0.7" --save_plots --save_processed_data

# Heatmap with remote reference:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --remote_reference HDD5974 --plot_heatmaps --save_plots --save_processed_data

# Heatmap with filtering and corrections:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --apply_filtering --filter_type comb --apply_drift_correction --apply_rotation --tilt_correction --plot_heatmaps --save_plots --save_processed_data

# Heatmap with frequency analysis:
python EDL_Batch.py --parent_dir . --sites HDD5449 HDD5456 HDD5470 HDD5974 --plot_heatmaps --perform_freq_analysis WMS --plot_coherence --save_plots --save_processed_data
"""