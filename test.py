"""
Training script for reinforcement learning experiments.

Created on: Sat May 23, 2020
Author: Giulio
"""

import os
import time
import pandas as pd
from spinup import sac_pytorch as sac
# from spinup import vpg_pytorch as vpg
# from spinup import ddpg_pytorch as ddpg
from MyEnv13 import CustomEnv13
# from MyEnv15 import CustomEnv15

def run_experiments():
    # Define logging directory based on current timestamp
    timestamp = int(time.time())
    logdir = f"/tmp/experiments/{timestamp}"
    
    # Running SAC algorithm on CustomEnv13
    print("Starting SAC training on CustomEnv13...")
    sac(env_fn=CustomEnv13, epochs=800000, seed=2)
    print("SAC training completed.")

    # Uncomment below to run additional experiments
    # print("Starting VPG training on CustomEnv13...")
    # vpg(env_fn=CustomEnv13, epochs=800000, seed=2)
    # print("VPG training completed.")

    # print("Starting DDPG training on CustomEnv13...")
    # ddpg(env_fn=CustomEnv13, epochs=800000, seed=2)
    # print("DDPG training completed.")

    # print("Starting VPG training on CustomEnv15...")
    # vpg(env_fn=CustomEnv15, epochs=800000, seed=2)
    # print("VPG training completed.")

    # print("Starting DDPG training on CustomEnv15...")
    # ddpg(env_fn=CustomEnv15, epochs=800000, seed=2)
    # print("DDPG training completed.")

    return logdir

def evaluate_training(logdir):
    # Load the progress file to check the performance
    progress_file_path = os.path.join(logdir, 'progress.txt')
    if os.path.exists(progress_file_path):
        data = pd.read_table(progress_file_path)
        print("Training evaluation data loaded successfully.")
        # Perform analysis or display last few epochs' data here
        # Example: print(data.tail())
    else:
        print("Progress file not found.")

if __name__ == "__main__":
    log_directory = run_experiments()
    evaluate_training(log_directory)
