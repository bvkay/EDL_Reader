#!/usr/bin/env python3
"""
Batch Process MT Data in Parallel

This script processes multiple magnetotelluric (MT) sites in batch mode.
Each site is assumed to be in its own folder (named after the site) contained within a parent directory.
For each site, a ProcessASCII instance (from Test_Process.py) is created and run in parallel
using a ProcessPoolExecutor. Output files and figures are named using the site folder name.

Usage example:
    python ./Test_Batch.py --parent_dir ./ --sites HDD5449 HDD5456 HDD5470 HDD5974 --param_file param.mt --rotate --tilt_correction --apply_smoothing --plot_data --plot_boundaries --plot_smoothed_windows --smoothing_method median --skip_minutes 10 5 --save_plots --max_workers 4
"""

import os
import argparse
import concurrent.futures
import datetime
from EDL_Process import ProcessASCII, write_log, write_batch_log, set_batch_mode, set_log_level, create_batch_summary

def process_single_site(site, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                        plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        save_processed_data, remote_reference=None):
    """
    Processes a single site folder.
    
    Args:
        site (str): Site folder name.
        parent_dir (str): Parent directory containing the site folders.
        apply_drift_correction (bool): Whether to apply drift correction.
        apply_rotation (bool): Whether to apply rotation.
        tilt_correction (bool): Whether to apply tilt correction.
        apply_smoothing (bool): Whether to apply smoothing.
        smoothing_method (str): Smoothing method to use ("median" or "adaptive").
        smoothing_window (int): Window size for smoothing.
        threshold_factor (float): Threshold multiplier for outlier detection.
        plot_data (bool): Whether to plot physical channel data.
        plot_boundaries (bool): Whether to show file boundaries on plots.
        plot_smoothed_windows (bool): Whether to shade smoothed windows.
        plot_coherence (bool): Whether to plot power spectra and coherence.
        perform_freq_analysis (bool): Whether to perform frequency analysis.
        log_first_rows (bool): Whether to log the first 5 rows from each file.
        sens_start (int): Start sample index for sensitivity test (unused now).
        sens_end (int): End sample index for sensitivity test (unused now).
        skip_minutes (int): Number of minutes to skip from the beginning.
        save_plots (bool): If True, plots are saved to disk.
        plot_drift (bool): Whether to plot drift-corrected data in timeseries.
        plot_rotation (bool): Whether to plot rotated data in timeseries.
        plot_tilt (bool): Whether to plot tilt-corrected data in timeseries.
        timezone (str): Timezone for the data.
        plot_original_data (bool): Whether to plot original data alongside corrected data.
        save_processed_data (bool): Whether to save processed data to file.
        remote_reference (str, optional): Site name to use as remote reference.
    
    Returns:
        dict: Processing results for the site.
    """
    site_dir = os.path.join(parent_dir, site)
    if not os.path.isdir(site_dir):
        write_log(f"Site folder not found: {site_dir}", level="ERROR")
        return {'status': 'FAILED', 'error': f'Site folder not found: {site_dir}'}

    try:
        write_log(f"Starting processing for site: {site}")
        processor = ProcessASCII(
            input_dir=site_dir,
            param_file="config/recorder.ini",  # Fixed path
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
            remote_reference=remote_reference
        )
        processor.tilt_correction = tilt_correction
        processor.save_plots = save_plots
        processor.process_all_files()
        
        # Return the results from the processor
        results = processor.results.copy()
        results['site'] = site
        write_log(f"Completed processing for site: {site}")
        return results
        
    except Exception as e:
        error_msg = f"Error processing site {site}: {e}"
        write_log(error_msg, level="ERROR")
        return {'status': 'FAILED', 'error': str(e), 'site': site}


def batch_process_sites(sites, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                        max_workers, plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        save_processed_data, remote_reference=None):
    """
    Processes multiple MT sites in parallel.
    
    Args:
        sites (list): List of site folder names.
        parent_dir (str): Parent directory containing the site folders.
        max_workers (int): Maximum number of parallel processes to use.
        plot_drift (bool): Whether to plot drift-corrected data in timeseries.
        plot_rotation (bool): Whether to plot rotated data in timeseries.
        plot_tilt (bool): Whether to plot tilt-corrected data in timeseries.
        timezone (str): Timezone for the data.
        plot_original_data (bool): Whether to plot original data alongside corrected data.
        remote_reference (str, optional): Site name to use as remote reference.
        (Other arguments are passed to process_single_site)
    
    Returns:
        dict: Results for all sites.
    """
    start_time = datetime.datetime.now()
    write_batch_log(f"Starting batch processing of {len(sites)} sites: {', '.join(sites)}")
    write_batch_log(f"Using {max_workers} parallel workers")
    
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_site, site, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                apply_smoothing, smoothing_method, smoothing_window, threshold_factor,
                plot_data, plot_boundaries, plot_smoothed_windows, plot_coherence,
                perform_freq_analysis, log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data, save_processed_data, remote_reference
            )
            for site in sites
        ]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result and 'site' in result:
                    site = result['site']
                    results[site] = result
                    completed += 1
                    
                    if result['status'] == 'SUCCESS':
                        write_batch_log(f"✓ Completed site {site} ({completed}/{len(sites)})")
                    else:
                        write_batch_log(f"✗ Failed site {site} ({completed}/{len(sites)}): {result.get('error', 'Unknown error')}")
                else:
                    write_batch_log(f"✗ Site processing returned invalid result", level="ERROR")
                    
            except Exception as e:
                write_batch_log(f"✗ Exception in site processing: {e}", level="ERROR")
    
    # Create and display summary
    summary = create_batch_summary(sites, results, start_time)
    write_batch_log("Batch processing completed")
    
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

    args = parser.parse_args()
    
    # Set up logging
    set_batch_mode(True)
    set_log_level(args.log_level)
    
    # Process sites
    batch_process_sites(
        args.sites, args.parent_dir, args.apply_drift_correction, args.apply_rotation, args.tilt_correction,
        args.apply_smoothing, args.smoothing_method, args.smoothing_window, args.threshold_factor,
        args.plot_data, args.plot_boundaries, args.plot_smoothed_windows, args.plot_coherence,
        args.perform_freq_analysis, args.log_first_rows, args.sens_start, args.sens_end,
        args.skip_minutes, args.save_plots, args.max_workers, args.plot_drift, args.plot_rotation,
        args.plot_tilt, args.timezone, args.plot_original_data, args.save_processed_data, args.remote_reference
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
"""