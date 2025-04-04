from datetime import datetime
import logging
import os
import yaml
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import importlib
import numpy as np

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add relative paths
sys.path.insert(0, os.path.join(script_dir, '../Scripts/'))

import MonoNN
import Utilities as Utilities
from DNN import DNN
import Scripts.LipVor_functions as LipVor_functions

importlib.reload(LipVor_functions)
importlib.reload(MonoNN)
importlib.reload(Utilities)

from MonoNN import MonoNN

path = '../Scripts/config.yaml'
with open(path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def write_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def train(config, external_points=None, model=None, regression=True):
    """
    Train the neural network model based on the provided configuration.

    Parameters:
    config (dict): Configuration dictionary loaded from YAML.
    external_points (torch.Tensor, optional): External points for training. Default is None.
    model (torch.nn.Module, optional): Predefined model. Default is None.

    Returns:
    MonoNN: Trained model.
    """
    
    # Set random seeds for reproducibility
    np.random.seed(config['training']['seed'])
    torch.manual_seed(config['training']['seed'])

    # Load training and testing data
    X_train_tensor = torch.load('../Data/X_train_data.pt')
    X_test_tensor = torch.load('../Data/X_test_data.pt')
    y_train_tensor = torch.load('../Data/y_train_data.pt')
    y_test_tensor = torch.load('../Data/y_test_data.pt')

    # Create DataLoader for training and validation data
    train_dt = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataload = DataLoader(train_dt, batch_size=64)

    val_dt = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataload = DataLoader(val_dt, batch_size=len(X_test_tensor))

    # Define the model
    if not model:
        model = DNN(config['model_architecture']['layers'], activations=config['model_architecture']['actfunc'])

    if external_points is None and config['training']['delta_external'] > 0:
        raise ValueError('External points are not provided, but delta_external is greater than 0')

    if config['model_architecture']['actfunc'][0] != 'identity':
        config['model_architecture']['actfunc'].insert(0, 'identity')
    if config['model_architecture']['actfunc'][-1] != 'identity' and regression:
        config['model_architecture']['actfunc'].append('identity')
    elif config['model_architecture']['actfunc'][-1] != 'sigmoid' and not regression:
        config['model_architecture']['actfunc'].append('sigmoid')

    ## TODO: Generalize for regression and classification and multi output
    if regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    mlp_model = MonoNN(_model_name="Prueba", _model=model)

    eps = config['training']['epsilon'] if config['training']['epsilon'] is not None else 0.0

    mlp_model.train(train_data=train_dataload, val_data=val_dataload, criterion=criterion, epsilon=eps,
                                 n_epochs=config['training']['n_epochs'], categorical_columns=[], verbose=config['training']['verbose'], n_visualized=1,
                                 monotone_relations=config['training']['monotone_relations'], optimizer_type=config['training']['optimizer_type'],
                                 learning_rate=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'],
                                 delta=config['training']['delta'], patience=config['training']['patience'],
                                 delta_synthetic=config['training']['delta_synthetic'], delta_external=config['training']['delta_external'],
                                 std_growth=config['training']['std_growth'], epsilon_synthetic=config['training']['epsilon_synthetic'],
                                 model_path='./Models/checkpoint_mlp_', external_points=external_points, seed=2023,
                                _early_stopping=config['training']['early_stopping'], delta_early_stopping=config['training']['delta_early_stopping']) 


    if config['training']['plot_history']:
        mlp_model.plot_history()


    return mlp_model

if __name__ == "__main__":
    logger.info("Starting the training script")
    train(config)
    logger.info("Training script completed")