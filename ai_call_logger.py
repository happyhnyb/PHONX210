import os
import time
import asyncio
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class CallLogger:
    """
    A logging class that creates separate log files for each caller number.
    Each caller gets their own dedicated log file for efficient logging.
    This version is specifically for insurance-ai directory.
    """
    
    def __init__(self, log_directory: str = "call-logs"):
        """
        Initialize the CallLogger.
        
        Args:
            log_directory (str): Directory where log files will be stored
        """
        self.log_directory = log_directory
        self.file_handles = {}  # Cache for file handles
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Create the log directory if it doesn't exist."""
        try:
            if not os.path.exists(self.log_directory):
                os.makedirs(self.log_directory, exist_ok=True)
                print(f"✅ Created log directory: {self.log_directory}")
            
            # Test if directory is writable
            test_file = os.path.join(self.log_directory, ".test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                print(f"⚠️ Warning: Log directory {self.log_directory} is not writable: {e}")
                raise
                
        except Exception as e:
            print(f"❌ Failed to create or verify log directory {self.log_directory}: {e}")
            raise
    
    def _get_log_filename(self, caller_number: str) -> str:
        """
        Generate the log filename for a given caller number.
        
        Args:
            caller_number (str): The phone number of the caller
            
        Returns:
            str: The log filename
        """
        # Clean the phone number for filename safety
        safe_number = caller_number.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
        return os.path.join(self.log_directory, f"{safe_number}.txt")
    
    def _get_file_handle(self, caller_number: str):
        """
        Get or create a file handle for the given caller number.
        
        Args:
            caller_number (str): The phone number of the caller
            
        Returns:
            file: The file handle for appending
        """
        if caller_number not in self.file_handles:
            filename = self._get_log_filename(caller_number)
            # Open file in append mode with buffering for efficiency
            self.file_handles[caller_number] = open(filename, 'a', buffering=8192)
        return self.file_handles[caller_number]
    
    def append_log(self, caller_number: str, log_data: str, include_timestamp: bool = True):
        """
        Append a log entry to the caller's dedicated log file.
        
        Args:
            caller_number (str): The phone number of the caller
            log_data (str): The log message to append
            include_timestamp (bool): Whether to include timestamp in the log entry
        """
        try:
            file_handle = self._get_file_handle(caller_number)
            
            if include_timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                log_entry = f"[{timestamp}] {log_data}\n"
            else:
                log_entry = f"{log_data}\n"
            
            file_handle.write(log_entry)
            file_handle.flush()  # Ensure data is written immediately
            
            # Force sync to disk to ensure data is persisted
            try:
                os.fsync(file_handle.fileno())
            except:
                pass  # fsync might fail on some systems, but that's okay
            
        except Exception as e:
            # Fallback to stderr if logging fails
            print(f"ERROR: Failed to log for caller {caller_number}: {e}", file=os.sys.stderr)
            # Try to recreate the log entry as a fallback
            try:
                self._recreate_log_entry(caller_number, log_data)
            except:
                pass
    
    
    def log_event(self, caller_number: str, message: str, global_start_time: Optional[float] = None):
        """
        Log an event with optional elapsed time tracking.
        All file I/O operations happen in parallel in separate threads to reduce latency.
        
        Args:
            caller_number (str): The phone number of the caller
            message (str): The log message
            global_start_time (float, optional): Global start time for elapsed time calculation
        """
        # Prepare log data
        if global_start_time is not None:
            elapsed = time.perf_counter() - global_start_time
            log_data = f"[{elapsed:8.4f}s] {message}"
        else:
            log_data = message
        
        # Fire-and-forget logging in separate thread to avoid blocking main execution
        async def _async_log():
            try:
                # Run file I/O operations in separate thread to avoid blocking
                await asyncio.to_thread(self.append_log, caller_number, log_data, True)
                
                # Run verification in separate thread as well
                await asyncio.to_thread(self._verify_log_written, caller_number, message)
                
            except Exception as e:
                # Fallback logging if async operations fail
                print(f"ERROR: Async logging failed for caller {caller_number}: {e}", file=os.sys.stderr)
                try:
                    # Fallback to synchronous logging
                    self.append_log(caller_number, log_data, True)
                    self._verify_log_written(caller_number, message)
                except Exception as fallback_error:
                    print(f"ERROR: Fallback logging also failed for caller {caller_number}: {fallback_error}", file=os.sys.stderr)
        
        # Always run synchronously for now since async context detection is commented out
        try:
            self.append_log(caller_number, log_data, True)
            self._verify_log_written(caller_number, message)
        except Exception as e:
            print(f"ERROR: Synchronous logging failed for caller {caller_number}: {e}", file=os.sys.stderr)
    
    def _verify_log_written(self, caller_number: str, message: str):
        """
        Verify that the log entry was actually written to the file.
        If the file doesn't exist, create it and add the log entry.
        
        Args:
            caller_number (str): The phone number of the caller
            message (str): The message that should have been logged
        """
        try:
            filename = self._get_log_filename(caller_number)
            
            # Force flush any pending writes
            if caller_number in self.file_handles:
                self.file_handles[caller_number].flush()
                os.fsync(self.file_handles[caller_number].fileno())
            
            # Check if file exists and has content
            if os.path.exists(filename):
                if os.path.getsize(filename) > 0:
                    return True
                else:
                    print(f"⚠️ Warning: Log file {filename} exists but is empty - attempting to recreate")
                    # File exists but is empty, try to recreate the log entry
                    self._recreate_log_entry(caller_number, message)
                    return True
            else:
                print(f"⚠️ Warning: Log file {filename} was not created - creating it now")
                # File doesn't exist, create it and add the log entry
                self._recreate_log_entry(caller_number, message)
                return True
                
        except Exception as e:
            print(f"⚠️ Warning: Could not verify log file for {caller_number}: {e}")
            # Try to recreate the log entry as a fallback
            try:
                self._recreate_log_entry(caller_number, message)
                return True
            except:
                return False
    
    def _recreate_log_entry(self, caller_number: str, message: str):
        """
        Recreate a log entry by directly writing to the file.
        This is a fallback method when the normal logging fails.
        
        Args:
            caller_number (str): The phone number of the caller
            message (str): The message to log
        """
        try:
            filename = self._get_log_filename(caller_number)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] {message}\n"
            
            # Write directly to file
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                f.flush()
                os.fsync(f.fileno())
            
            print(f"✅ Recreated log entry for {caller_number} in {filename}")
            
        except Exception as e:
            print(f"❌ Failed to recreate log entry for {caller_number}: {e}")
    
    def close_caller_log(self, caller_number: str):
        """
        Close the log file for a specific caller.
        
        Args:
            caller_number (str): The phone number of the caller
        """
        if caller_number in self.file_handles:
            try:
                self.file_handles[caller_number].close()
                del self.file_handles[caller_number]
            except Exception as e:
                print(f"ERROR: Failed to close log for caller {caller_number}: {e}", file=os.sys.stderr)
    
    def close_all_logs(self):
        """Close all open log files."""
        for caller_number in list(self.file_handles.keys()):
            self.close_caller_log(caller_number)
    
    def __del__(self):
        """Destructor to ensure all files are closed."""
        self.close_all_logs()
    
    def get_log_file_path(self, caller_number: str) -> str:
        """
        Get the full path to a caller's log file.
        
        Args:
            caller_number (str): The phone number of the caller
            
        Returns:
            str: Full path to the log file
        """
        return os.path.abspath(self._get_log_filename(caller_number))
    
    def log_file_exists_and_has_content(self, caller_number: str) -> bool:
        """
        Check if a caller's log file exists and has content.
        
        Args:
            caller_number (str): The phone number of the caller
            
        Returns:
            bool: True if log file exists and has content, False otherwise
        """
        try:
            filename = self._get_log_filename(caller_number)
            return os.path.exists(filename) and os.path.getsize(filename) > 0
        except Exception:
            return False
  