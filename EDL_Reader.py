import os
import numpy as np
import pandas as pd
import datetime
import glob

LOG_FILE = "process_ascii.log"

def write_log(message, level="INFO"):
    """
    Writes log messages to the log file.

    Args:
        message (str): The log message to be written.
        level (str, optional): Log level (INFO, WARNING, ERROR). Defaults to "INFO".
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {level} - {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_message)

class ASCIIReader: 
    """
    Reads and processes ASCII EDL magnetotelluric data files.
    """
    
    def __init__(self, data_dir, metadata, average=False, log_first_rows=False):
        """
        Initializes the ASCII reader.
        
        Args:
            data_dir (str): Path to the directory containing ASCII data files.
            metadata (dict): Metadata dictionary from the process script.
            average (bool): Whether to average values over intervals.
            log_first_rows (bool): If True, log the first 5 rows of data.
        """
        self.data_dir = data_dir
        self.metadata = metadata
        self.average = average
        self.log_first_rows = log_first_rows
        self.channels = ['BX', 'BY', 'BZ', 'EX', 'EY']
        self.data = None
    
    def get_data_files(self, specific_channels=None):
        """
        Gets data files for specified channels in the data directory and its subdirectories.
        Uses the station_long_identifier from metadata to find files with the correct naming pattern.
        
        Args:
            specific_channels (list, optional): List of specific channels to load. If None, loads all channels.
        
        Returns:
            dict: Dictionary mapping channel names to file paths.
        """
        files = {}
        station_identifier = self.metadata.get('station_long_identifier', 'EDL_')
        
        # Use specific channels if provided, otherwise use all channels
        channels_to_load = specific_channels if specific_channels else self.channels
        
        for channel in channels_to_load:
            # Look for files with the pattern: {station_identifier}YYMMDDHHMMSS.{channel}
            pattern = os.path.join(self.data_dir, "**", f"{station_identifier}*.{channel}")
            channel_files = glob.glob(pattern, recursive=True)
            
            if channel_files:
                # Sort files by name to ensure chronological order
                channel_files.sort()
                files[channel] = channel_files
                write_log(f"Found {len(channel_files)} files for channel {channel} using pattern '{station_identifier}*.{channel}'")
            else:
                write_log(f"No files found for channel {channel} using pattern '{station_identifier}*.{channel}' in {self.data_dir}", level="WARNING")
                # Fallback: try the old pattern without station identifier
                fallback_pattern = os.path.join(self.data_dir, "**", f"*.{channel}")
                fallback_files = glob.glob(fallback_pattern, recursive=True)
                if fallback_files:
                    fallback_files.sort()
                    files[channel] = fallback_files
                    write_log(f"Found {len(fallback_files)} files for channel {channel} using fallback pattern '*.{channel}'")
                else:
                    write_log(f"No files found for channel {channel} in {self.data_dir}", level="WARNING")
                    files[channel] = []
        return files
    
    def read_channel_data(self, file_path):
        """
        Reads a single channel file and returns the data as a list of floats.
        
        Args:
            file_path (str): Path to the channel file.
            
        Returns:
            list: List of float values from the file.
        """
        try:
            with open(file_path, 'r') as f:
                data = [float(line.strip()) for line in f if line.strip()]
            return data
        except Exception as e:
            write_log(f"Error reading {file_path}: {e}", level="ERROR")
            return []
    
    def read_specific_channels(self, channels, outfile):
        """
        Reads data for specific channels only.
        
        Args:
            channels (list): List of channel names to read (e.g., ['BX', 'BY'])
            outfile: File object to write output to.
            
        Returns:
            pd.DataFrame: Processed data as a DataFrame with only the specified channels.
        """
        try:
            files = self.get_data_files(specific_channels=channels)
            
            # Check if we have any files
            if not any(files.values()):
                write_log(f"No data files found for channels {channels} in {self.data_dir}", level="ERROR")
                return pd.DataFrame()
            
            # Read data from each specified channel
            channel_data = {}
            min_length = float('inf')
            
            for channel, file_list in files.items():
                if not file_list:
                    continue
                    
                # Read all files for this channel
                all_data = []
                for file_path in file_list:
                    data = self.read_channel_data(file_path)
                    all_data.extend(data)
                
                channel_data[channel] = all_data
                min_length = min(min_length, len(all_data))
            
            # Ensure all channels have the same length
            for channel in channel_data:
                channel_data[channel] = channel_data[channel][:min_length]
            
            # Create DataFrame
            if self.average:
                # Calculate averages for each channel
                avg_data = {}
                for channel, data in channel_data.items():
                    avg_data[channel] = [np.mean(data)]
                df = pd.DataFrame(avg_data)
            else:
                # Create DataFrame with all data
                df = pd.DataFrame(channel_data)
                
                # Check for NaNs and drop rows if necessary
                if df.isnull().values.any():
                    # Count rows with any NaN values
                    n_nans = len(df[df.isnull().any(axis=1)])
                    write_log(f"WARNING: Found {n_nans} rows with NaN values in concatenated data. Dropping these rows.", level="WARNING")
                    df = df.dropna().reset_index(drop=True)
            
            self.data = df
            return df
            
        except Exception as e:
            write_log(f"Error reading data for channels {channels} from {self.data_dir}: {e}", level="ERROR")
            return pd.DataFrame()
    
    def read_all_data(self, outfile):
        """
        Reads all channel files and combines them into a DataFrame.
        
        Args:
            outfile: File object to write output to.
            
        Returns:
            pd.DataFrame: Processed data as a DataFrame.
        """
        return self.read_specific_channels(self.channels, outfile)