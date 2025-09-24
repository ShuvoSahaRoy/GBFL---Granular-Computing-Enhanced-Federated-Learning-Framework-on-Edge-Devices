import sys
import os
from datetime import datetime
import random
import numpy as np
import torch
import pandas as pd
from config import SEED


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

def setup_logger(save_log=False, log_dir="results/logs"):
    if save_log:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = os.path.join(log_dir, f"experiment_log_{timestamp}.txt")
        log_file = open(log_file_path, "w")
        tee = Tee(sys.stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        return log_file, tee
    else:
        return None, None
    
# Function to process granular ball data for logistic regression
def process_gb_data_lr(train_data_list):
    updated_list = []
    for df in train_data_list:
        df.drop(df.columns[-2], axis=1, inplace=True)
        updated_list.append(df)
    return updated_list


# def check_convergence(value_list, window_size=10, tolerance=0.01):
#     """
#     Check if the last 'window_size' elements in the list have converged.
    
#     Args:
#         value_list (list): List of values (e.g., accuracies over rounds).
#         window_size (int): Number of recent elements to check (default: 10).
#         tolerance (float): Maximum allowed difference for convergence (default: 0.01, i.e., 1%).
    
#     Returns:
#         bool: True if converged (stop), False if not (continue).
#     """
#     # If the list is shorter than the window size, continue
#     if len(value_list) < window_size:
#         return False
    
#     # Get the last 'window_size' elements
#     recent_values = value_list[-window_size:]
    
#     # Calculate the range (max - min) of the recent values
#     value_range = max(recent_values) - min(recent_values)
    
#     # If the range is within tolerance, consider it converged
#     if value_range <= tolerance:
#         return True  # Stop
#     else:
#         return False  # Continue

def check_convergence(value_list, window_size=10, tolerance=0.01):
    if len(value_list) < window_size:
        return False
    recent_values = value_list[-window_size:]
    # Use moving average to smooth fluctuations
    avg = sum(recent_values) / window_size
    # Check if recent values are within tolerance of the average
    max_deviation = max(abs(v - avg) for v in recent_values)
    return max_deviation <= tolerance * avg  # Relative tolerance