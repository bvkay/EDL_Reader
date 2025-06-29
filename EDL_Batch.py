#!/usr/bin/env python3
"""
Batch Process MT Data in Parallel

This script processes multiple magnetotelluric (MT) sites in batch mode using Processing.txt configuration.
Each site is assumed to be in its own folder (named after the site) contained within a parent directory.
For each site, a ProcessASCII instance (from EDL_Process.py) is created and run in parallel
using a ThreadPoolExecutor. Output files and figures are named using the site folder name.

Usage example:
    python EDL_Batch.py --input_config --max_workers 4 --plot_timeseries --tilt_correction
    python EDL_Batch.py --input_config --max_workers 2 --run_lemimt
"""

import os
import argparse
import concurrent.futures
import datetime
import sys
import subprocess
from EDL_Process import ProcessASCII, write_log, write_batch_log, set_batch_mode, set_log_level, create_batch_summary, read_processing_config

def process_single_site(site, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_timeseries, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes,
                        plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        remote_reference=None, apply_filtering=False,
                        filter_type="comb", filter_channels=None, filter_params=None,
                        plot_heatmaps=False, heatmap_nperseg=1024, heatmap_noverlap=None, heatmap_thresholds=None,
                        run_lemimt=False, lemimt_path="lemimt.exe"):
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
        plot_timeseries (bool): Whether to plot data.
        plot_boundaries (bool): Whether to plot boundaries.
        plot_smoothed_windows (bool): Whether to plot smoothed windows.
        plot_coherence (bool): Whether to plot coherence.
        perform_freq_analysis (bool): Whether to perform frequency analysis.
        log_first_rows (bool): Whether to log first rows.
        sens_start (int): Start sample for sensitivity test.
        sens_end (int): End sample for sensitivity test.
        skip_minutes (list): Minutes to skip from start and end.
        plot_drift (bool): Whether to plot drift correction.
        plot_rotation (bool): Whether to plot rotation.
        plot_tilt (bool): Whether to plot tilt correction.
        timezone (str): Timezone for the data.
        plot_original_data (bool): Whether to plot original data.
        remote_reference (str, optional): Remote reference site name.
        apply_filtering (bool): Whether to apply frequency filtering.
        filter_type (str): Type of filter to apply.
        filter_channels (list, optional): Channels to filter.
        filter_params (dict, optional): Additional filter parameters.
        plot_heatmaps (bool): Whether to generate heatmap plots.
        heatmap_nperseg (int): Number of points per segment for heatmap FFT.
        heatmap_noverlap (int, optional): Number of points to overlap between heatmap segments.
        heatmap_thresholds (dict, optional): Custom coherence thresholds for heatmap quality scoring.
        run_lemimt (bool): Whether to run lemimt.exe on the processed output file.
        lemimt_path (str): Full path to the lemimt.exe executable.
    
    Returns:
        dict: Results dictionary.
    """
    try:
        site_dir = os.path.join(parent_dir, site)
        if not os.path.exists(site_dir):
            write_batch_log(f"Site directory not found: {site_dir}", level="ERROR")
            return {"status": "FAILED", "error": f"Directory not found: {site_dir}"}
        
        write_batch_log(f"Processing site: {site}")
        
        # Create processor instance
        processor = ProcessASCII(
            input_dir=os.path.join(parent_dir, site),
            param_file="config/recorder.ini",
            average=False,
            perform_freq_analysis=perform_freq_analysis,
            plot_timeseries=plot_timeseries,
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
            save_processed_data=True,  # Always save processed data
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
        
        # Always save plots when plotting
        processor.save_plots = plot_timeseries
        
        processor.tilt_correction = tilt_correction
        
        # Process the site
        processor.process_all_files()
        
        return processor.results
        
    except Exception as e:
        error_msg = f"Error processing site {site}: {e}"
        write_batch_log(error_msg, level="ERROR")
        return {"status": "FAILED", "error": str(e)}


def batch_process_sites(sites, parent_dir, apply_drift_correction, apply_rotation, tilt_correction,
                        apply_smoothing, smoothing_method, smoothing_window,
                        threshold_factor, plot_timeseries, plot_boundaries,
                        plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                        log_first_rows, sens_start, sens_end, skip_minutes,
                        max_workers, plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                        remote_reference=None, apply_filtering=False,
                        filter_type="comb", filter_channels=None, filter_params=None,
                        plot_heatmaps=False, heatmap_nperseg=1024, heatmap_noverlap=None, heatmap_thresholds=None,
                        run_lemimt=False, lemimt_path="lemimt.exe"):
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
        plot_timeseries (bool): Whether to plot data.
        plot_boundaries (bool): Whether to plot boundaries.
        plot_smoothed_windows (bool): Whether to plot smoothed windows.
        plot_coherence (bool): Whether to plot coherence.
        perform_freq_analysis (bool): Whether to perform frequency analysis.
        log_first_rows (bool): Whether to log first rows.
        sens_start (int): Start sample for sensitivity test.
        sens_end (int): End sample for sensitivity test.
        skip_minutes (list): Minutes to skip from start and end.
        max_workers (int): Maximum number of parallel workers.
        plot_drift (bool): Whether to plot drift correction.
        plot_rotation (bool): Whether to plot rotation.
        plot_tilt (bool): Whether to plot tilt correction.
        timezone (str): Timezone for the data.
        plot_original_data (bool): Whether to plot original data.
        remote_reference (str, optional): Remote reference site name.
        apply_filtering (bool): Whether to apply frequency filtering.
        filter_type (str): Type of filter to apply.
        filter_channels (list, optional): Channels to filter.
        filter_params (dict, optional): Additional filter parameters.
        plot_heatmaps (bool): Whether to generate heatmap plots.
        heatmap_nperseg (int): Number of points per segment for heatmap FFT.
        heatmap_noverlap (int, optional): Number of points to overlap between heatmap segments.
        heatmap_thresholds (dict, optional): Custom coherence thresholds for heatmap quality scoring.
        run_lemimt (bool): Whether to run lemimt.exe on the processed output file.
        lemimt_path (str): Full path to the lemimt.exe executable.
    
    Returns:
        dict: Dictionary mapping site names to results.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
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
                threshold_factor, plot_timeseries, plot_boundaries,
                plot_smoothed_windows, plot_coherence, perform_freq_analysis,
                log_first_rows, sens_start, sens_end, skip_minutes,
                plot_drift, plot_rotation, plot_tilt, timezone, plot_original_data,
                remote_reference, apply_filtering,
                filter_type, filter_channels, filter_params,
                plot_heatmaps, heatmap_nperseg, heatmap_noverlap, heatmap_thresholds,
                run_lemimt, lemimt_path
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


def process_single_task(task, args, worker_num=None):
    """Process a single task with the given arguments."""
    try:
        # Call EDL_Process.py for each task
        # Set environment variable for worker number
        env = os.environ.copy()
        env['WORKER_NUM'] = str(worker_num) if worker_num else '1'
        
        # Build command for EDL_Process.py
        cmd = [sys.executable, "EDL_Process.py", "--site_name", task['site_name']]
        
        # Add all the processing arguments
        if args.apply_drift_correction:
            cmd.append("--apply_drift_correction")
        if args.apply_rotation:
            cmd.append("--apply_rotation")
        if args.tilt_correction:
            cmd.append("--tilt_correction")
        if args.apply_smoothing:
            cmd.extend(["--apply_smoothing", "--smoothing_method", args.smoothing_method,
                       "--smoothing_window", str(args.smoothing_window),
                       "--threshold_factor", str(args.threshold_factor)])
        if args.plot_timeseries:
            cmd.append("--plot_timeseries")
        if args.plot_boundaries:
            cmd.append("--plot_boundaries")
        if args.plot_smoothed_windows:
            cmd.append("--plot_smoothed_windows")
        if args.plot_coherence:
            cmd.append("--plot_coherence")
        if args.perform_freq_analysis:
            cmd.extend(["--perform_freq_analysis", args.perform_freq_analysis])
        if args.log_first_rows:
            cmd.append("--log_first_rows")
        if args.sens_start != 0 or args.sens_end != 5000:
            cmd.extend(["--sens_start", str(args.sens_start), "--sens_end", str(args.sens_end)])
        if args.skip_minutes != [0, 0]:
            cmd.extend(["--skip_minutes"] + [str(x) for x in args.skip_minutes])
        if args.plot_drift:
            cmd.append("--plot_drift")
        if args.plot_rotation:
            cmd.append("--plot_rotation")
        if args.plot_tilt:
            cmd.append("--plot_tilt")
        if args.timezone != ["UTC"]:
            cmd.extend(["--timezone"] + args.timezone)
        if args.plot_original_data:
            cmd.append("--plot_original_data")
        
        # IMPORTANT: Use remote reference from Processing.txt, not command line
        if task['remote_reference']:
            cmd.extend(["--remote_reference", task['remote_reference']])
        elif args.remote_reference:
            # Only use command line remote reference if not specified in Processing.txt
            cmd.extend(["--remote_reference", args.remote_reference])
        
        if args.apply_filtering:
            cmd.extend(["--apply_filtering", "--filter_type", args.filter_type])
            if args.filter_channels:
                cmd.extend(["--filter_channels"] + args.filter_channels)
            if args.filter_notch_freq != 50.0:
                cmd.extend(["--filter_notch_freq", str(args.filter_notch_freq)])
            if args.filter_quality_factor != 30.0:
                cmd.extend(["--filter_quality_factor", str(args.filter_quality_factor)])
            if args.filter_harmonics:
                cmd.extend(["--filter_harmonics"] + [str(x) for x in args.filter_harmonics])
            if args.filter_low_freq:
                cmd.extend(["--filter_low_freq", str(args.filter_low_freq)])
            if args.filter_high_freq:
                cmd.extend(["--filter_high_freq", str(args.filter_high_freq)])
            if args.filter_cutoff_freq:
                cmd.extend(["--filter_cutoff_freq", str(args.filter_cutoff_freq)])
            if args.filter_order != 4:
                cmd.extend(["--filter_order", str(args.filter_order)])
            if args.filter_reference_channel:
                cmd.extend(["--filter_reference_channel", args.filter_reference_channel])
            if args.filter_length != 64:
                cmd.extend(["--filter_length", str(args.filter_length)])
            if args.filter_mu != 0.01:
                cmd.extend(["--filter_mu", str(args.filter_mu)])
            if args.filter_method != "filtfilt":
                cmd.extend(["--filter_method", args.filter_method])
        if args.plot_heatmaps:
            cmd.append("--plot_heatmaps")
            if args.heatmap_nperseg != 1024:
                cmd.extend(["--heatmap_nperseg", str(args.heatmap_nperseg)])
            if args.heatmap_noverlap:
                cmd.extend(["--heatmap_noverlap", str(args.heatmap_noverlap)])
            if args.heatmap_thresholds:
                cmd.extend(["--heatmap_thresholds", args.heatmap_thresholds])
        if args.decimate:
            cmd.extend(["--decimate"] + [str(x) for x in args.decimate])
        if args.run_lemimt:
            cmd.extend(["--run_lemimt", "--lemimt_path", args.lemimt_path])
        
        # Run the command with environment variable
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            return {
                'site': task['site_name'],
                'remote_reference': task['remote_reference'],
                'status': 'success',
                'output': result.stdout
            }
        else:
            return {
                'site': task['site_name'],
                'remote_reference': task['remote_reference'],
                'status': 'failed',
                'error': result.stderr
            }
            
    except Exception as e:
        return {
            'site': task['site_name'],
            'remote_reference': task['remote_reference'],
            'status': 'exception',
            'error': str(e)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process multiple MT sites using Processing.txt configuration.")
    parser.add_argument("--input_config", action="store_true", help="Use Processing.txt configuration file for site parameters and remote references.")
    parser.add_argument("--parent_dir", default=".", help="Parent directory containing site folders.")
    parser.add_argument("--apply_drift_correction", action="store_true", help="Apply drift correction to the data.")
    parser.add_argument("--apply_rotation", action="store_true", help="Apply rotation correction based on the 'erotate' parameter.")
    parser.add_argument("--tilt_correction", action="store_true", help="Apply tilt correction to make mean(By) = 0.")
    parser.add_argument("--apply_smoothing", action="store_true", help="Apply smoothing to the data.")
    parser.add_argument("--smoothing_method", choices=["median", "adaptive"], default="median", help="Smoothing method to use.")
    parser.add_argument("--smoothing_window", type=int, default=2500, help="Window size for smoothing.")
    parser.add_argument("--threshold_factor", type=float, default=10.0, help="Threshold multiplier for outlier detection.")
    parser.add_argument("--plot_timeseries", action="store_true", help="Display physical channel plots.")
    parser.add_argument("--plot_boundaries", action="store_true", help="Plot file boundaries.")
    parser.add_argument("--plot_smoothed_windows", action="store_true", help="Shade smoothed windows.")
    parser.add_argument("--plot_coherence", action="store_true", help="Plot power spectra and coherence.")
    parser.add_argument("--perform_freq_analysis", nargs="?", const="W", metavar="METHODS",
                        help="Perform frequency analysis. Use W for Welch, M for Multi-taper, S for Spectrogram. "
                             "Combine them: 'WMS' for all three, 'WM' for Welch + Multi-taper, etc. Default: W")
    parser.add_argument("--log_first_rows", action="store_true", help="Log the first 5 rows from each file.")
    parser.add_argument("--sens_start", type=int, default=0, help="Start sensor index for processing.")
    parser.add_argument("--sens_end", type=int, default=5000, help="End sensor index for processing.")
    parser.add_argument("--skip_minutes", nargs=2, type=int, default=[0, 0], 
                        help="Skip first and last N minutes of data (e.g., --skip_minutes 10 20).")
    parser.add_argument("--plot_drift", action="store_true", help="Plot drift-corrected data in timeseries.")
    parser.add_argument("--plot_rotation", action="store_true", help="Plot rotated data in timeseries.")
    parser.add_argument("--plot_tilt", action="store_true", help="Plot tilt-corrected data in timeseries.")
    parser.add_argument("--timezone", nargs="+", default=["UTC"], 
                        help="Timezone(s) for the data. Use one timezone for both sites (e.g., 'Australia/Adelaide') or two timezones for main and remote sites (e.g., 'Australia/Adelaide' 'UTC')")
    parser.add_argument("--plot_original_data", action="store_true", help="Plot original data alongside corrected data.")
    parser.add_argument("--remote_reference", help="Remote reference site name for processing (overrides Processing.txt if specified).")
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
    parser.add_argument("--decimate", nargs="+", type=int, help="Decimation factors to apply (e.g., --decimate 2 5 10).")
    parser.add_argument("--run_lemimt", action="store_true", help="Run lemimt processing after data processing.")
    parser.add_argument("--lemimt_path", default="lemimt.exe", help="Path to lemimt executable.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers.")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_config:
        parser.error("--input_config is required. This script processes all sites from Processing.txt configuration file.")
    
    # Set up logging
    from EDL_Process import set_log_level, set_batch_mode, write_batch_log
    set_log_level("INFO")
    set_batch_mode(True)
    
    write_batch_log("Starting batch processing with Processing.txt configuration")
    
    # Load sites from Processing.txt
    from EDL_Process import read_processing_config
    processing_config = read_processing_config()
    write_batch_log(f"Loaded Processing.txt configuration with {len(processing_config)} sites")
    
    # Create processing tasks for each site and its remote references
    processing_tasks = []
    for site_name, site_config in processing_config.items():
        # Handle multiple remote references
        remote_refs = []
        if site_config.get('remote_reference'):
            remote_ref_str = site_config['remote_reference']
            if remote_ref_str.startswith('[') and remote_ref_str.endswith(']'):
                # Multiple remote references
                remote_refs = [ref.strip() for ref in remote_ref_str[1:-1].split(';')]
            else:
                # Single remote reference
                remote_refs = [remote_ref_str.strip()]
        
        # If no remote references in config, use command line argument
        if not remote_refs and args.remote_reference:
            remote_refs = [args.remote_reference]
        
        # Process with each remote reference (or no remote reference)
        for remote_ref in remote_refs + [None] if remote_refs else [None]:
            processing_tasks.append({
                'site_name': site_name,
                'xarm': site_config['xarm'],
                'yarm': site_config['yarm'],
                'remote_reference': remote_ref
            })
    
    write_batch_log(f"Created {len(processing_tasks)} processing tasks from Processing.txt")
    
    # Process each task with parallel processing
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks with worker numbers
        future_to_task = {}
        for i, task in enumerate(processing_tasks):
            # Assign worker number (1-based)
            worker_num = (i % args.max_workers) + 1
            future = executor.submit(
                process_single_task,
                task, args, worker_num
            )
            future_to_task[future] = task
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                status = result.get('status', 'UNKNOWN')
                write_batch_log(f"✓ Completed {task['site_name']}" + 
                              (f" with remote reference {task['remote_reference']}" if task['remote_reference'] else "") +
                              f": {status}")
            except Exception as e:
                write_batch_log(f"✗ Exception processing {task['site_name']}: {str(e)}", level="ERROR")
                results.append({
                    'site': task['site_name'],
                    'remote_reference': task['remote_reference'],
                    'status': 'exception',
                    'error': str(e)
                })
    
    # Create summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    write_batch_log(f"Batch processing completed: {successful} successful, {failed} failed")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Parallel workers used: {args.max_workers}")
    print(f"{'='*80}")
    
    if failed > 0:
        print(f"\nFAILED TASKS:")
        for result in results:
            if result['status'] != 'success':
                print(f"  - {result['site']}" + 
                      (f" (RR: {result['remote_reference']})" if result['remote_reference'] else "") +
                      f": {result['error']}")
    
    write_batch_log("Batch processing finished")

"""
To run: 

# BASIC BATCH PROCESSING EXAMPLES
# ===============================

# Basic processing with rotation and tilt correction:
python EDL_Batch.py --input_config --apply_rotation --tilt_correction --apply_smoothing --plot_boundaries --plot_smoothed_windows --plot_timeseries

# Processing with drift correction and rotation:
python EDL_Batch.py --input_config --apply_drift_correction --apply_rotation --tilt_correction --apply_smoothing --skip_minutes 20 --plot_timeseries --plot_boundaries --plot_smoothed_windows --smoothing_method median

# Processing with drift correction and rotation, plotting both uncorrected and corrected data:
python EDL_Batch.py --input_config --apply_drift_correction --apply_rotation --plot_drift --plot_rotation --plot_timeseries

# Processing with tilt correction, plotting both uncorrected and tilt-corrected data:
python EDL_Batch.py --input_config --tilt_correction --plot_tilt --plot_timeseries

# Basic processing without any corrections:
python EDL_Batch.py --input_config --plot_timeseries

# Processing with timezone conversion (e.g., to local time):
python EDL_Batch.py --input_config --timezone "Australia/Sydney" --plot_timeseries

# Processing with timezone conversion and all corrections:
python EDL_Batch.py --input_config --timezone "America/New_York" --apply_drift_correction --apply_rotation --tilt_correction --plot_drift --plot_rotation --plot_tilt --plot_timeseries

# ADVANCED BATCH PROCESSING EXAMPLES
# ==================================

# Full processing pipeline with all corrections, smoothing, and analysis:
python EDL_Batch.py --input_config --plot_timeseries --apply_drift_correction --apply_rotation --tilt_correction --apply_smoothing --smoothing_method adaptive --perform_freq_analysis WMS --plot_coherence --plot_drift --plot_rotation --plot_tilt --plot_original_data --timezone "Australia/Sydney" --skip_minutes 5

# Processing with custom smoothing parameters:
python EDL_Batch.py --input_config --plot_timeseries --apply_smoothing --smoothing_window 500 --threshold_factor 3.0 --smoothing_method median --plot_boundaries --plot_smoothed_windows

# Processing with all corrections but no smoothing:
python EDL_Batch.py --input_config --plot_timeseries --apply_drift_correction --apply_rotation --tilt_correction --plot_drift --plot_rotation --plot_tilt --plot_original_data

# FREQUENCY FILTERING EXAMPLES
# ============================

# Basic comb filter for powerline noise removal (50 Hz and harmonics):
python EDL_Batch.py --input_config --apply_filtering --filter_type comb --plot_timeseries

# Comb filter with custom notch frequency (60 Hz for US powerline):
python EDL_Batch.py --input_config --apply_filtering --filter_type comb --filter_notch_freq 60.0 --plot_timeseries

# Bandpass filter for specific frequency range:
python EDL_Batch.py --input_config --apply_filtering --filter_type bandpass --filter_low_freq 0.01 --filter_high_freq 1.0 --plot_timeseries

# Highpass filter to remove DC and low-frequency drift:
python EDL_Batch.py --input_config --apply_filtering --filter_type highpass --filter_cutoff_freq 0.001 --plot_timeseries

# Lowpass filter to remove high-frequency noise:
python EDL_Batch.py --input_config --apply_filtering --filter_type lowpass --filter_cutoff_freq 5.0 --plot_timeseries

# Filter specific channels only:
python EDL_Batch.py --input_config --apply_filtering --filter_type comb --filter_channels Bx By --plot_timeseries

# Adaptive filtering using remote reference (requires remote reference in Processing.txt):
python EDL_Batch.py --input_config --apply_filtering --filter_type adaptive --filter_reference_channel rBx --plot_timeseries

# Filtering with other corrections:
python EDL_Batch.py --input_config --apply_filtering --filter_type comb --apply_drift_correction --apply_rotation --tilt_correction --plot_timeseries

# Filtering with smoothing:
python EDL_Batch.py --input_config --apply_filtering --filter_type comb --apply_smoothing --smoothing_method median --plot_timeseries

# HEATMAP EXAMPLES
# =================

# Basic heatmap generation for quality control:
python EDL_Batch.py --input_config --plot_heatmaps

# Heatmap with custom FFT parameters:
python EDL_Batch.py --input_config --plot_heatmaps --heatmap_nperseg 2048 --heatmap_noverlap 1024

# Heatmap with custom coherence thresholds:
python EDL_Batch.py --input_config --plot_heatmaps --heatmap_thresholds "0.9,0.7,0.7"

# Heatmap with remote reference (from Processing.txt):
python EDL_Batch.py --input_config --plot_heatmaps

# Heatmap with filtering and corrections:
python EDL_Batch.py --input_config --apply_filtering --filter_type comb --apply_drift_correction --apply_rotation --tilt_correction --plot_heatmaps

# Heatmap with frequency analysis:
python EDL_Batch.py --input_config --plot_heatmaps --perform_freq_analysis WMS --plot_coherence

# DECIMATION EXAMPLES
# ===================

# Process with 2x decimation (5 Hz output):
python EDL_Batch.py --input_config --plot_timeseries --decimate 2

# Process with multiple decimation factors (10 Hz, 5 Hz, 2 Hz, 1 Hz):
python EDL_Batch.py --input_config --plot_timeseries --decimate 2 5 10

# Decimation with corrections:
python EDL_Batch.py --input_config --plot_timeseries --apply_drift_correction --apply_rotation --tilt_correction --decimate 2 5

# LEMIMT PROCESSING EXAMPLES
# ==========================

# Basic lemimt processing (requires Windows):
python EDL_Batch.py --input_config --run_lemimt

# lemimt processing with custom executable path:
python EDL_Batch.py --input_config --run_lemimt --lemimt_path "C:/Program Files/lemimt/lemimt.exe"

# lemimt processing with full processing pipeline:
python EDL_Batch.py --input_config --plot_timeseries --apply_drift_correction --apply_rotation --tilt_correction --run_lemimt

# lemimt processing with decimation:
python EDL_Batch.py --input_config --run_lemimt --decimate 2 5

# PARALLEL PROCESSING EXAMPLES
# ============================

# Process with custom number of parallel workers:
python EDL_Batch.py --input_config --plot_timeseries --max_workers 8

# Process with single worker (sequential processing):
python EDL_Batch.py --input_config --plot_timeseries --max_workers 1

# NOTES ON PROCESSING.TXT CONFIGURATION:
# - The file should be in the format: Site, xarm, yarm, [remote_ref1; remote_ref2]
# - Multiple remote references in brackets will create multiple processing runs
# - Dipole lengths (xarm, yarm) are automatically applied from the configuration
# - Remote references are automatically loaded and processed
# - Each remote reference creates a separate output file with unique CPU prefix
# - All sites in Processing.txt will be processed automatically
# - Use --remote_reference to override Processing.txt remote references for all sites
"""