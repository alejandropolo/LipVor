import os
import pickle
import logging
import signal
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import yaml
import time
import os
import pandas as pd
import gc
import psutil
## Import torch
import torch
from torch.utils.data import TensorDataset, DataLoader
from neuralsens.partial_derivatives import calculate_second_partial_derivatives_mlp
from sklearn.model_selection import train_test_split
from scipy.spatial import Voronoi
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
import sys
# sys.path.insert(0,'/home/apolo/PhD/LipVorOptimization/Scripts/')
sys.path.insert(0,'../Scripts/')
from Utilities import *
from MultiProcessingVoronoi import *

import Utilities
import LipVor_functions
from LipVor_functions import *
from LipVor_Certification import LipVorMonotoneCertification
from multiprocessing import Pool

from generate_data import generate_data_Neumann
from train import train
from DNN import DNN
import random

## Plotly render
import plotly
from IPython.display import display, HTML

# Configuration
torch.manual_seed(0)
np.random.seed(0)



def signal_handler(sig, frame):
    # logging.info("Received termination signal. Exiting gracefully.")
    exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def load_and_prepare_data():
    """Load and prepare the AutoMPG dataset"""
    df = pd.read_csv('../Data/AutoMPG/auto-mpg.csv')
    df = df[df.horsepower != "?"]
    df = df[[col for col in df if col != 'mpg'] + ['mpg']].dropna()
    df = df.drop('car name', axis=1)
    df['horsepower'] = df['horsepower'].astype(float)
    
    cols_selected = [i for i in range(df.shape[1]-1)]
    X = df.iloc[:, cols_selected].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    
    # Train/test split and normalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=0)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    y_val = scaler.transform(y_val)
    
    return (
        torch.tensor(X_train, dtype=torch.float),
        torch.tensor(X_test, dtype=torch.float),
        torch.tensor(X_val, dtype=torch.float),
        torch.tensor(y_train, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.float),
        torch.tensor(y_val, dtype=torch.float),
        cols_selected
    )

def save_result_pickle(result_data, idx, folder_path):
    """Save result to pickle file with atomic write"""
    temp_file = os.path.join(folder_path, f"temp_Multi_{idx}.pkl")
    final_file = os.path.join(folder_path, f"Results_AutoMPG_Multi_{idx}.pkl")
    
    with open(temp_file, "wb") as f:
        pickle.dump(result_data, f)
    os.rename(temp_file, final_file)

def process_subsquare_seq(model, config, weights, biases, global_lipschitz_constant,
                     X_train_tensor, sub_square, intervals, idx, folder_path):
    """Process a single subsquare with retries and error handling"""
    retry_limit = 3
    attempt = 0
    result_file = os.path.join(folder_path, f"Results_AutoMPG_{idx}.pkl")
    
    while attempt < retry_limit:
        attempt += 1
        try:
            # logging.info(f"Processing subsquare {idx} (attempt {attempt}/{retry_limit})")
            
            # Clear memory before each attempt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process the subsquare
            space_filled, x_reentrenamiento, combo_results = LipVorMonotoneCertification(
                model=model,
                actfunc=config['model_architecture']['actfunc'],
                weights=weights,
                biases=biases,
                monotone_relations=[i for i in config['training']['monotone_relations'] if i != 0],
                variable_index=[i for i, val in enumerate(config['training']['monotone_relations']) if val != 0],
                global_lipschitz_constant=global_lipschitz_constant,
                X_train_tensor=X_train_tensor,
                original_points=sub_square,
                intervals=intervals,
                max_iterations=2,
                n=10,
                epsilon_derivative=0.1,
                epsilon_proyection=1e-5,
                probability=0.0,
                seed=1,
                verbose=0,
                julia=True,
                categorical_indices=[0, 6],
                categorical_values={
                    0: X_train_tensor[:, 0].unique().numpy(),
                    6: X_train_tensor[:, -1].unique().numpy()
                }
            )
            
            # Save results
            with open(result_file, "wb") as f:
                pickle.dump({
                    "combo_results": combo_results,
                    "intervals": intervals
                }, f)
            
            # logging.info(f"Successfully processed subsquare {idx}")
            return True
            
        except Exception as e:
            # logging.error(f"Error processing subsquare {idx} (attempt {attempt}): {str(e)}")
            if os.path.exists(result_file):
                os.remove(result_file)
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # logging.error(f"Failed to process subsquare {idx} after {retry_limit} attempts")
    return False

def process_subsquare(args, model, config, weights, biases, global_lipschitz_constant, 
                     X_train_tensor, categorical_indices, categorical_values, folder_path):
    """Process a single subsquare with retries and checkpointing"""
    sub_square, intervals, idx = args
    
    result_file = os.path.join(folder_path, f"Results_AutoMPG_Multi_{idx}.pkl")
    
    # Skip if already processed
    if os.path.exists(result_file):
        # logging.info(f"Skipping subsquare {idx} - already processed")
        return None
    
    retry_limit = 3
    for attempt in range(retry_limit):
        try:
            # Clear memory before each attempt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            result = LipVorMonotoneCertification(
                model=model,
                actfunc=config['model_architecture']['actfunc'],
                weights=weights,
                biases=biases,
                monotone_relations=[i for i in config['training']['monotone_relations'] if i != 0],
                variable_index=[i for i, val in enumerate(config['training']['monotone_relations']) if val != 0],
                global_lipschitz_constant=global_lipschitz_constant,
                X_train_tensor=X_train_tensor,
                original_points=sub_square,
                intervals=intervals,
                max_iterations=200,
                n=10,
                epsilon_derivative=0.1,
                epsilon_proyection=1e-4,
                probability=0.0,
                seed=1,
                verbose=0,
                julia=True,
                categorical_indices=[0, 6],
                categorical_values={
                    0: X_train_tensor[:, 0].unique().numpy(),
                    6: X_train_tensor[:, -1].unique().numpy()
                }
            )
            
            # Save result immediately
            result_data = (result, intervals)
            save_result_pickle(result_data, idx, folder_path)
            
            # logging.info(f"Successfully processed subsquare {idx}")
            return result
            
        except Exception as e:
            # logging.error(f"Error processing subsquare {idx} (attempt {attempt+1}): {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # logging.error(f"Failed to process subsquare {idx} after {retry_limit} attempts")
    return None

def main():
    # Load and prepare data
    (X_train_tensor, X_test_tensor, X_val_tensor, 
     y_train_tensor, y_test_tensor, y_val_tensor, 
     cols_selected) = load_and_prepare_data()
    
    # Load model and config
    model = torch.load('../Models/AutoMPG_model_penalized_2.pt', weights_only=False)
    # model = torch.load('../Models/AutoMPG_model.pt', weights_only=False)
    with open('configAutoMPG.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config['model_architecture']['layers'][0] = len(cols_selected)
    if config['model_architecture']['actfunc'][0] != 'identity':
        config['model_architecture']['actfunc'].insert(0, 'identity')
    if config['model_architecture']['actfunc'][-1] != 'identity':
        config['model_architecture']['actfunc'].append('identity')
    
    # Calculate Lipschitz constant
    weights, biases = LipVor_functions.get_weights_and_biases(model)
    x = X_train_tensor[0].view(1, -1)
    y = y_train_tensor[0].view(1, -1)
    
    W, Z, O, D, D2, D_accum, Q, H, counter, mlpstr = calculate_second_partial_derivatives_mlp(
        weights, biases, ['identity', 'sigmoid', 'identity'], x, y,
        sens_end_layer=len(config['model_architecture']['actfunc']))
    
    global_lipschitz_constant = LipVor_functions.hessian_bound(
        W=W, 
        actfunc=config['model_architecture']['actfunc'],
        partial_monotonic_variable=[1, 2, 3],
        n_variables=len(cols_selected))

    # logging.info(f"Global Lipschitz constant: {global_lipschitz_constant}")
    # Divide the points into subsquares
    n_subsquares = 2  # Number of subsquares in each dimension
    d = len(cols_selected) # Number of dimensions (assuming 2D space, for example)    
    monotone_relations = [i for i in config['training']['monotone_relations'] if i!=0]
    variable_index = [i for i,val in enumerate(config['training']['monotone_relations']) if val!=0]
    # logging.info(f"Monotone relations: {monotone_relations}")
    # logging.info(f"Variable index of the Monotone relations: {variable_index}")

    # Define categorical indices and categorical values
    categorical_indices = [0,6]
    # Get unique values of column cylinders in X_train_tensor
    unique_cylinders = X_train_tensor[:,0].unique().numpy()
    print('Unique Cylinders: ',unique_cylinders)
    # Get unique values of column origin in X_train_tensor
    unique_origin = X_train_tensor[:,-1].unique().numpy()
    print('Unique Origin: ',unique_origin)
    categorical_values = {0:unique_cylinders,6:unique_origin}


    original_points = generate_random_points(1000, d,categorical_indices=categorical_indices, categorical_values=categorical_values)

    print('Monotone Relations: ',monotone_relations)
    print('Variable Index of the Monotone Relations: ',variable_index)


    # Divide the points into subsquares and get their intervals
    # sub_squares, subsquares_intervals = divide_into_subsquares(original_points, n_subsquares, d,categorical_indices=categorical_indices,categorical_values=categorical_values)

    
    # n_subsquares = 2  # Number of divisions per dimension
    subsquares, intervals = divide_into_subsquares(
        original_points, n_subsquares, len(cols_selected),
        categorical_indices=categorical_indices,
        categorical_values=categorical_values)
    
    # Limit to 32 subsquares if needed
    # subsquares = subsquares[:32]
    # intervals = intervals[:32]

    # Prepare results folder
    folder_path = "../Results/AutoMPG_MultiProcessing"
    os.makedirs(folder_path, exist_ok=True)

    # Execute first the sequential to load the julia code
    total_subsquares = 3
    for idx in range(total_subsquares):
        result_file = os.path.join(folder_path, f"Results_AutoMPG_{idx}.pkl")
        
        # # Skip already processed subsquares
        # if os.path.exists(result_file):
        #     logging.info(f"Skipping subsquare {idx} - already processed")
        #     continue
        
        success = process_subsquare_seq(
            model, config, weights, biases, global_lipschitz_constant,
            X_train_tensor, subsquares[idx], intervals[idx],
            idx, folder_path)
    
    
    # Prepare arguments for multiprocessing
    process_args = [(sub, intv, idx) for idx, (sub, intv) in enumerate(zip(subsquares, intervals))]
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_subsquare,
        model=model,
        config=config,
        weights=weights,
        biases=biases,
        global_lipschitz_constant=global_lipschitz_constant,
        X_train_tensor=X_train_tensor,
        categorical_indices=categorical_indices,
        categorical_values=categorical_values,
        folder_path=folder_path
    )
    
    # Process with multiprocessing
    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_func, process_args),
            total=len(process_args)
        ))
    
    time_elapsed = time.time() - start_time
    # logging.info(f"Processing complete. Time elapsed: {time_elapsed:.2f} seconds")
    print(f"\nProcessing complete. Time elapsed: {time_elapsed:.2f} seconds")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('fork')  # <--- ADD THIS LIN
    main()