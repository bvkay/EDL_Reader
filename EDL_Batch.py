#!/usr/bin/env python3
"""
EDL Batch Processing Script

This script processes multiple EDL sites in parallel using multiprocessing.
For each site, a ProcessASCII instance (from EDL_Process.py) is created and run in parallel
to handle the processing of ASCII data files.

Usage:
    python EDL_Batch.py --sites HDD5449,HDD5456,HDD5470 --parent_dir . --apply_drift_correction --apply_rotation --tilt_correction --plot_data --save_plots
    
    OR using a configuration file:
    python EDL_Batch.py --config_file Processing.txt --parent_dir . --apply_drift_correction --apply_rotation --tilt_correction --plot_data --save_plots

Author: [Your Name]
Date: [Date]
"""

import argparse
import os
import sys
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the processing module
from EDL_Process import ProcessASCII, write_log, write_batch_log, write_site_log, set_batch_mode, set_log_level, create_batch_summary, read_processing_config

def process_single_site(site, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                        plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        save_processed_data, remote_reference=None, apply_filtering=False,
                        filter_type="comb", filter_channels=None, filter_params=None,
                        plot_heatmaps=False, heatmap_nperseg=1024, heatmap_noverlap=None, heatmap_thresholds=None,
                        site_config=None, run_lemimt=False, lemimt_path="lemimt.exe"):
    """Process a single site with the given parameters.
    
    Args:
        site (str): Site name to process
        parent_dir (str): Parent directory containing site folders
        apply_drift_correction (bool): Whether to apply drift correction
        apply_rotation (bool): Whether to apply rotation correction
        tilt_correction (bool): Whether to apply tilt correction
        apply_smoothing (bool): Whether to apply smoothing
        smoothing_method (str): Smoothing method to use
        smoothing_window (int): Smoothing window size
        threshold_factor (float): Threshold factor for smoothing
        plot_data (bool): Whether to plot data
        plot_boundaries (bool): Whether to plot file boundaries
        plot_smoothed_windows (bool): Whether to plot smoothed windows
        plot_coherence (bool): Whether to plot coherence
        perform_freq_analysis (str): Frequency analysis options
        log_first_rows (bool): Whether to log first rows
        sens_start (int): Sensor start index
        sens_end (int): Sensor end index
        skip_minutes (list): Minutes to skip from start and end
        save_plots (bool): Whether to save plots
        plot_drift (bool): Whether to plot drift correction
        plot_rotation (bool): Whether to plot rotation correction
        plot_tilt (bool): Whether to plot tilt correction
        timezone (str): Timezone for the data
        plot_original_data (bool): Whether to plot original data
        save_processed_data (bool): Whether to save processed data
        remote_reference (str): Remote reference site name
        apply_filtering (bool): Whether to apply filtering
        filter_type (str): Type of filter to apply
        filter_channels (list): Channels to filter
        filter_params (dict): Filter parameters
        plot_heatmaps (bool): Whether to generate heatmaps
        heatmap_nperseg (int): Number of points per segment for heatmap FFT
        heatmap_noverlap (int): Number of points to overlap between heatmap segments
        heatmap_thresholds (dict): Custom coherence thresholds for heatmap quality scoring
        site_config (dict): Site-specific configuration from Processing.txt
        run_lemimt (bool): Whether to run lemimt.exe on the processed output file
        lemimt_path (str): Full path to the lemimt.exe executable
    
    Returns:
        dict: Processing results for the site
    """
    try:
        write_site_log(f"Starting processing for site: {site}")
        
        # Use site-specific remote reference if available in config
        if site_config and 'remote_reference' in site_config:
            site_remote_ref = site_config['remote_reference']
            if site_remote_ref and site_remote_ref != 'None':
                remote_reference = site_remote_ref
                write_site_log(f"Using site-specific remote reference: {remote_reference}")
        
        # Create processor
        processor = ProcessASCII(
            input_dir=os.path.join(parent_dir, site),
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
        
        # Set lemimt parameters
        processor.run_lemimt = run_lemimt
        processor.lemimt_path = lemimt_path
        
        # Override metadata with site-specific config if available
        if site_config:
            if 'xarm' in site_config:
                processor.metadata['xarm'] = site_config['xarm']
            if 'yarm' in site_config:
                processor.metadata['yarm'] = site_config['yarm']
            write_site_log(f"Using site config: xarm={site_config.get('xarm', 'default')} m, yarm={site_config.get('yarm', 'default')} m")
        
        # Process the data
        processor.process_all_files()
        
        write_site_log(f"Successfully completed processing for site: {site}")
        return {'status': 'SUCCESS', 'site': site, 'end_time': datetime.datetime.now()}
        
    except Exception as e:
        error_msg = f"Error processing site {site}: {e}"
        write_site_log(error_msg, level="ERROR")
        return {'status': 'FAILED', 'site': site, 'error': str(e), 'end_time': datetime.datetime.now()}

def batch_process_sites(sites, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_data, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes, save_plots,
                        max_workers, plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        save_processed_data, remote_reference=None, apply_filtering=False,
                        filter_type="comb", filter_channels=None, filter_params=None,
                        plot_heatmaps=False, heatmap_nperseg=1024, heatmap_noverlap=None, heatmap_thresholds=None,
                        site_configs=None, run_lemimt=False, lemimt_path="lemimt.exe"):
    """Process multiple sites in parallel.
    
    Args:
        sites (list): List of site names to process
        parent_dir (str): Parent directory containing site folders
        apply_drift_correction (bool): Whether to apply drift correction
        apply_rotation (bool): Whether to apply rotation correction
        tilt_correction (bool): Whether to apply tilt correction
        apply_smoothing (bool): Whether to apply smoothing
        smoothing_method (str): Smoothing method to use
        smoothing_window (int): Smoothing window size
        threshold_factor (float): Threshold factor for smoothing
        plot_data (bool): Whether to plot data
        plot_boundaries (bool): Whether to plot file boundaries
        plot_smoothed_windows (bool): Whether to plot smoothed windows
        plot_coherence (bool): Whether to plot coherence
        perform_freq_analysis (str): Frequency analysis options
        log_first_rows (bool): Whether to log first rows
        sens_start (int): Sensor start index
        sens_end (int): Sensor end index
        skip_minutes (list): Minutes to skip from start and end
        save_plots (bool): Whether to save plots
        max_workers (int): Maximum number of parallel workers
        plot_drift (bool): Whether to plot drift correction
        plot_rotation (bool): Whether to plot rotation correction
        plot_tilt (bool): Whether to plot tilt correction
        timezone (str): Timezone for the data
        plot_original_data (bool): Whether to plot original data
        save_processed_data (bool): Whether to save processed data
        remote_reference (str): Remote reference site name
        apply_filtering (bool): Whether to apply filtering
        filter_type (str): Type of filter to apply
        filter_channels (list): Channels to filter
        filter_params (dict): Filter parameters
        plot_heatmaps (bool): Whether to generate heatmaps
        heatmap_nperseg (int): Number of points per segment for heatmap FFT
        heatmap_noverlap (int): Number of points to overlap between heatmap segments
        heatmap_thresholds (dict): Custom coherence thresholds for heatmap quality scoring
        site_configs (dict): Dictionary mapping site names to their configurations
        run_lemimt (bool): Whether to run lemimt.exe on the processed output file
        lemimt_path (str): Full path to the lemimt.exe executable
    
    Returns:
        dict: Dictionary mapping site names to results
    """
    start_time = datetime.datetime.now()
    write_batch_log(f"Starting batch processing of {len(sites)} sites at {start_time}")
    
    results = {}
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_site = {}
        for site in sites:
            # Get site-specific config if available
            site_config = site_configs.get(site, {}) if site_configs else {}
            
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
                plot_heatmaps, heatmap_nperseg, heatmap_noverlap, heatmap_thresholds,
                site_config, run_lemimt, lemimt_path
            )
            future_to_site[future] = site
        
        # Collect results as they complete
        for future in as_completed(future_to_site):
            site = future_to_site[future]
            try:
                result = future.result()
                results[site] = result
                if result['status'] == 'SUCCESS':
                    write_batch_log(f"Site {site} completed successfully")
                else:
                    write_batch_log(f"Site {site} failed: {result.get('error', 'Unknown error')}", level="ERROR")
            except Exception as e:
                error_msg = f"Exception occurred while processing site {site}: {e}"
                write_batch_log(error_msg, level="ERROR")
                results[site] = {'status': 'FAILED', 'site': site, 'error': str(e), 'end_time': datetime.datetime.now()}
    
    # Create and display batch summary
    summary = create_batch_summary(sites, results, start_time)
    print(summary)
    write_batch_log(summary)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple EDL sites")
    
    # Site selection - either use --sites or --config_file
    site_group = parser.add_mutually_exclusive_group(required=True)
    site_group.add_argument("--sites", nargs="+", help="List of site directories to process")
    site_group.add_argument("--config_file", help="Path to Processing.txt or similar config file")
    
    parser.add_argument("--parent_dir", default=".", help="Parent directory containing site folders")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    
    # Processing options
    parser.add_argument("--apply_drift_correction", action="store_true", help="Apply drift correction")
    parser.add_argument("--apply_rotation", action="store_true", help="Apply rotation correction")
    parser.add_argument("--tilt_correction", action="store_true", help="Apply tilt correction")
    parser.add_argument("--apply_smoothing", action="store_true", help="Apply smoothing")
    parser.add_argument("--smoothing_method", choices=["median", "adaptive"], default="median", help="Smoothing method")
    parser.add_argument("--smoothing_window", type=int, default=2500, help="Smoothing window size")
    parser.add_argument("--threshold_factor", type=float, default=10.0, help="Threshold factor for smoothing")
    
    # Plotting options
    parser.add_argument("--plot_data", action="store_true", help="Plot data")
    parser.add_argument("--plot_boundaries", action="store_true", help="Plot file boundaries")
    parser.add_argument("--plot_smoothed_windows", action="store_true", help="Plot smoothed windows")
    parser.add_argument("--plot_coherence", action="store_true", help="Plot coherence")
    parser.add_argument("--plot_drift", action="store_true", help="Plot drift correction")
    parser.add_argument("--plot_rotation", action="store_true", help="Plot rotation correction")
    parser.add_argument("--plot_tilt", action="store_true", help="Plot tilt correction")
    parser.add_argument("--plot_original_data", action="store_true", help="Plot original data")
    parser.add_argument("--save_plots", action="store_true", help="Save plots")
    
    # Data options
    parser.add_argument("--save_processed_data", action="store_true", help="Save processed data")
    parser.add_argument("--log_first_rows", action="store_true", help="Log first rows of data")
    parser.add_argument("--sens_start", type=int, default=0, help="Sensor start index")
    parser.add_argument("--sens_end", type=int, default=5000, help="Sensor end index")
    parser.add_argument("--skip_minutes", nargs=2, type=int, default=[0, 0], help="Minutes to skip from start and end")
    parser.add_argument("--timezone", default="UTC", help="Timezone for the data")
    
    # Frequency analysis
    parser.add_argument("--perform_freq_analysis", help="Frequency analysis options (W=Welch, M=Multi-taper, S=Spectrogram)")
    
    # Remote reference
    parser.add_argument("--remote_reference", help="Remote reference site name (overrides config file)")
    
    # Filtering options
    parser.add_argument("--apply_filtering", action="store_true", help="Apply frequency filtering")
    parser.add_argument("--filter_type", choices=["comb", "bandpass", "highpass", "lowpass", "adaptive"], default="comb", help="Filter type")
    parser.add_argument("--filter_channels", nargs="+", help="Channels to filter")
    parser.add_argument("--filter_notch_freq", type=float, default=50.0, help="Notch frequency for comb filter")
    parser.add_argument("--filter_quality_factor", type=float, default=30.0, help="Quality factor for comb filter")
    parser.add_argument("--filter_harmonics", nargs="+", type=float, help="Harmonic frequencies for comb filter")
    parser.add_argument("--filter_low_freq", type=float, default=0.1, help="Low frequency for bandpass filter")
    parser.add_argument("--filter_high_freq", type=float, default=10.0, help="High frequency for bandpass filter")
    parser.add_argument("--filter_cutoff_freq", type=float, default=0.1, help="Cutoff frequency for highpass/lowpass filter")
    parser.add_argument("--filter_order", type=int, default=4, help="Filter order")
    parser.add_argument("--filter_reference_channel", help="Reference channel for adaptive filtering")
    parser.add_argument("--filter_length", type=int, default=64, help="Filter length for adaptive filtering")
    parser.add_argument("--filter_mu", type=float, default=0.01, help="Step size for adaptive filtering")
    parser.add_argument("--filter_method", choices=["filtfilt", "lfilter"], default="filtfilt", help="Filtering method")
    
    # Heatmap options
    parser.add_argument("--plot_heatmaps", action="store_true", help="Generate heatmap plots for quality control")
    parser.add_argument("--heatmap_nperseg", type=int, default=1024, help="Number of points per segment for heatmap FFT")
    parser.add_argument("--heatmap_noverlap", type=int, help="Number of points to overlap between heatmap segments")
    parser.add_argument("--heatmap_thresholds", help="Custom coherence thresholds for heatmap quality scoring")
    
    # LEMIMT.EXE integration
    parser.add_argument("--run_lemimt", action="store_true", default=False,
                        help="Run lemimt.exe on the processed output file after processing is complete")
    parser.add_argument("--lemimt_path", type=str, default="lemimt.exe",
                        help="Full path to the lemimt.exe executable (default: lemimt.exe in current directory)")
    
    # Logging
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Set up logging
    set_log_level(args.log_level)
    set_batch_mode(True)
    
    # Determine sites to process
    sites = []
    site_configs = {}
    
    if args.config_file:
        # Read sites from config file
        write_batch_log(f"Reading site configuration from: {args.config_file}")
        site_configs = read_processing_config(args.config_file)
        
        if not site_configs:
            write_batch_log(f"No valid site configurations found in {args.config_file}", level="ERROR")
            return
        
        sites = list(site_configs.keys())
        
        # Check if only one site is specified
        if len(sites) == 1:
            site = sites[0]
            write_batch_log(f"Only one site found in config file: {site}")
            write_batch_log(f"Consider using single-site processing instead:")
            write_batch_log(f"  python EDL_Process.py --input_dir {site} [other options]")
            write_batch_log(f"Continuing with batch processing for single site...")
        
        write_batch_log(f"Found {len(sites)} sites in config file: {', '.join(sites)}")
        
        # Display site configurations
        for site, config in site_configs.items():
            write_batch_log(f"Site {site}: xarm={config.get('xarm', 'N/A')} m, yarm={config.get('yarm', 'N/A')} m, remote_ref={config.get('remote_reference', 'None')}")
    
    else:
        # Use manually specified sites
        sites = args.sites
        write_batch_log(f"Processing {len(sites)} manually specified sites: {', '.join(sites)}")
    
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
    
    # Parse heatmap thresholds if provided
    heatmap_thresholds = args.heatmap_thresholds
    if isinstance(heatmap_thresholds, str):
        try:
            parts = heatmap_thresholds.split(',')
            if len(parts) == 3:
                heatmap_thresholds = {
                    'good': float(parts[0]),
                    'fair': float(parts[1]),
                    'poor': float(parts[2])
                }
        except Exception as e:
            write_batch_log(f"Error parsing heatmap thresholds: {e}, using defaults", level="WARNING")
            heatmap_thresholds = None
    
    # Process the sites
    results = batch_process_sites(
        sites=sites,
        parent_dir=args.parent_dir,
        apply_drift_correction=args.apply_drift_correction,
        apply_rotation=args.apply_rotation,
        tilt_correction=args.tilt_correction,
        apply_smoothing=args.apply_smoothing,
        smoothing_method=args.smoothing_method,
        smoothing_window=args.smoothing_window,
        threshold_factor=args.threshold_factor,
        plot_data=args.plot_data,
        plot_boundaries=args.plot_boundaries,
        plot_smoothed_windows=args.plot_smoothed_windows,
        plot_coherence=args.plot_coherence,
        perform_freq_analysis=args.perform_freq_analysis,
        log_first_rows=args.log_first_rows,
        sens_start=args.sens_start,
        sens_end=args.sens_end,
        skip_minutes=args.skip_minutes,
        save_plots=args.save_plots,
        max_workers=args.max_workers,
        plot_drift=args.plot_drift,
        plot_rotation=args.plot_rotation,
        plot_tilt=args.plot_tilt,
        timezone=args.timezone,
        plot_original_data=args.plot_original_data,
        save_processed_data=args.save_processed_data,
        remote_reference=args.remote_reference,
        apply_filtering=args.apply_filtering,
        filter_type=args.filter_type,
        filter_channels=args.filter_channels,
        filter_params=filter_params,
        plot_heatmaps=args.plot_heatmaps,
        heatmap_nperseg=args.heatmap_nperseg,
        heatmap_noverlap=args.heatmap_noverlap,
        heatmap_thresholds=heatmap_thresholds,
        site_configs=site_configs,
        run_lemimt=args.run_lemimt,
        lemimt_path=args.lemimt_path
    )
    
    # Print final summary
    successful = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
    failed = len(results) - successful
    write_batch_log(f"Batch processing completed. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main()

"""
To run: 

# CONFIGURATION FILE EXAMPLES
# ===========================

# Using Processing.txt for batch processing (recommended):
python EDL_Batch.py --config_file Processing.txt --parent_dir . --apply_rotation --tilt_correction --plot_data --save_plots

# Using Processing.txt with all corrections:
python EDL_Batch.py --config_file Processing.txt --parent_dir . --apply_drift_correction --apply_rotation --tilt_correction --plot_data --save_plots

# Using custom config file:
python EDL_Batch.py --config_file my_sites.txt --parent_dir . --plot_data --save_plots

# MANUAL SITE SPECIFICATION EXAMPLES
# ==================================

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

# FREQUENCY FILTERING EXAMPLES
# ============================

# Basic comb filter for powerline noise removal (50 Hz and harmonics):
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type comb --plot_data --save_plots --save_processed_data

# Comb filter with custom notch frequency (60 Hz for US powerline):
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type comb --filter_notch_freq 60.0 --plot_data --save_plots --save_processed_data

# Bandpass filter for specific frequency range:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type bandpass --filter_low_freq 0.01 --filter_high_freq 1.0 --plot_data --save_plots --save_processed_data

# Highpass filter to remove DC and low-frequency drift:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type highpass --filter_cutoff_freq 0.001 --plot_data --save_plots --save_processed_data

# Lowpass filter to remove high-frequency noise:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type lowpass --filter_cutoff_freq 5.0 --plot_data --save_plots --save_processed_data

# Filter specific channels only:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type comb --filter_channels Bx By --plot_data --save_plots --save_processed_data

# Adaptive filtering using remote reference:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type adaptive --filter_reference_channel rBx --plot_data --save_plots --save_processed_data

# Filtering with other corrections:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type comb --apply_drift_correction --apply_rotation --tilt_correction --plot_data --save_plots --save_processed_data

# Filtering with smoothing:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type comb --apply_smoothing --smoothing_method median --plot_data --save_plots --save_processed_data

# HEATMAP EXAMPLES
# =================

# Basic heatmap generation for quality control:
python EDL_Batch.py --config_file Processing.txt --plot_heatmaps --save_plots --save_processed_data

# Heatmap with custom FFT parameters:
python EDL_Batch.py --config_file Processing.txt --plot_heatmaps --heatmap_nperseg 2048 --heatmap_noverlap 1024 --save_plots --save_processed_data

# Heatmap with custom coherence thresholds:
python EDL_Batch.py --config_file Processing.txt --plot_heatmaps --heatmap_thresholds "0.9,0.7,0.7" --save_plots --save_processed_data

# Heatmap with remote reference (from Processing.txt):
python EDL_Batch.py --config_file Processing.txt --plot_heatmaps --save_plots --save_processed_data

# Heatmap with filtering and corrections:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type comb --apply_drift_correction --apply_rotation --tilt_correction --plot_heatmaps --save_plots --save_processed_data

# Heatmap with frequency analysis:
python EDL_Batch.py --config_file Processing.txt --plot_heatmaps --perform_freq_analysis WMS --plot_coherence --save_plots --save_processed_data

# FREQUENCY ANALYSIS EXAMPLES
# ===========================

# Welch power spectra analysis:
python EDL_Batch.py --config_file Processing.txt --perform_freq_analysis W --plot_data --save_plots --save_processed_data

# Multi-taper spectral analysis:
python EDL_Batch.py --config_file Processing.txt --perform_freq_analysis M --plot_data --save_plots --save_processed_data

# Spectrogram analysis:
python EDL_Batch.py --config_file Processing.txt --perform_freq_analysis S --plot_data --save_plots --save_processed_data

# All frequency analysis methods:
python EDL_Batch.py --config_file Processing.txt --perform_freq_analysis WMS --plot_data --save_plots --save_processed_data

# LEMIMT.EXE INTEGRATION EXAMPLES
# ===============================

# Process and run lemimt.exe (Windows only):
python EDL_Batch.py --config_file Processing.txt --plot_data --save_plots --save_processed_data --run_lemimt

# Process with lemimt and cleanup:
python EDL_Batch.py --config_file Processing.txt --plot_data --save_plots --save_processed_data --run_lemimt --cleanup_processed_files

# COMPREHENSIVE WORKFLOW EXAMPLES
# ===============================

# Complete processing pipeline with all features:
python EDL_Batch.py --config_file Processing.txt --apply_drift_correction --apply_rotation --tilt_correction --apply_filtering --filter_type comb --apply_smoothing --smoothing_method median --perform_freq_analysis WMS --plot_heatmaps --plot_data --save_plots --save_processed_data --run_lemimt

# Quality control focused processing:
python EDL_Batch.py --config_file Processing.txt --apply_filtering --filter_type comb --plot_heatmaps --perform_freq_analysis WMS --plot_data --save_plots --save_processed_data

# Production processing with timezone and all corrections:
python EDL_Batch.py --config_file Processing.txt --timezone "Australia/Sydney" --apply_drift_correction --apply_rotation --tilt_correction --apply_filtering --filter_type comb --plot_data --save_plots --save_processed_data --run_lemimt
"""