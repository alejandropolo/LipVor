import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import torch
from torch.autograd.functional import jacobian
from torch.optim import SGD, Adam,LBFGS
from torch.nn import BCELoss,MSELoss
import torch.nn as nn
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import accuracy_score
import math
import Utilities as Utilities
import importlib
import sys
sys.path.append('../Scripts/')
sys.path.append('..')
from Scripts.pytorchtools import EarlyStopping

import logging
from datetime import datetime
import torch

importlib.reload(Utilities)


class MonoNN:
    """
    Monotonic Neural Network (MonoNN) class:

        This class enables the training of unconstrained Artificial Neural Networks (ANNs) 
        by imposing a penalization term. The methodology aims to train unconstrained probably partial 
        ε-monotonic ANNs that can later be certified as partial monotonic using the LipVor Algorithm. 
        The idea is to use a modified version of the usual training loss, similar to the approach 
        in Monteiro et al. (2022). The ANN is forced to follow an ε-monotonic relationship at the training 
        data by means of a penalization term. By continuity of the ANN, enforcing an ε-monotonicity constraint 
        at the training data is expected to ensure that the ANN is partially monotonic in a neighborhood of 
        each training point. However, partial monotonicity is only expected to be achieved close to the regions 
        where the penalization is enforced. Therefore, there is no guarantee of obtaining a partial monotonic ANN 
        in the whole domain. Consequently, after training the unconstrained partial ε-monotonic ANNs, the LipVor 
        Algorithm is computed to try to certify partial monotonicity. This methodology allows for the certification 
        of partial monotonicity without needing to use constrained architectures.

    Attributes:
        _model_name (str): The name of the model.
        _model (torch.nn.Module): The neural network model.
        _jacobian (torch.Tensor): The Jacobian matrix of the model.
        _avg_train_losses (list): The average training losses.
        _avg_train_losses_modified (list): The average modified training losses.
        _avg_valid_losses (list): The average validation losses.
    """
    def __init__(self,_model_name,_model):
        self._model_name = _model_name
        self._model = _model
        self._jacobian = None
        self._avg_train_losses = None
        self._avg_train_losses_modified = None
        self._avg_valid_losses = None

    def batch_jacobian(self, input):
            """
            Calculate the Jacobian for all samples in a batch. The result is a tensor of size (n_outputs, batch_size, n_inputs) such that the following vectors
    
            - jacobian[i,j,k]: Returns the Jacobian with respect to output i and variable k, evaluated at sample j
            (f/x_i)'(batch_j)
    
            Args:
                input (torch.Tensor): Input tensor
    
            Returns:
                torch.Tensor: Jacobian matrix
            """
            f_sum = lambda x: torch.sum(self._model(x), axis=0)
            return jacobian(f_sum, input, create_graph=True)
    
    def checkpoint(model, filename):
        torch.save(model.state_dict(), filename)
        
    def resume(model, filename):
        model.load_state_dict(torch.load(filename))
    
    def train(self, train_data, val_data, criterion, n_epochs, categorical_columns=[None], verbose=1, n_visualized=1,
              monotone_relations=[0], optimizer_type='Adam', learning_rate=0.01, delta=0.0, weight_decay=0.0, delta_synthetic=0.0, delta_external=0.0,
              patience=100, model_path='./Models/checkpoint_', std_syntethic=0.0, std_growth=0.0, epsilon=0.0,
              epsilon_synthetic=0.0, external_points=None, seed=None, keep_model=True, _early_stopping=True):
        """
        Train the neural network model with adjusted standard deviation.
    
        Args:
            train_data (DataLoader): Training data loader.
            val_data (DataLoader): Validation data loader.
            criterion (torch.nn.Module): Loss function.
            n_epochs (int): Number of epochs.
            categorical_columns (list, optional): List of categorical columns. Defaults to [None].
            verbose (int, optional): Verbosity level. Defaults to 1.
            n_visualized (int, optional): Number of visualized epochs. Defaults to 1.
            monotone_relations (list, optional): List of monotone relations. Defaults to [0].
            optimizer_type (str, optional): Optimizer type. Defaults to 'Adam'.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            delta (float, optional): Delta value. Defaults to 0.0.
            weight_decay (float, optional): Weight decay. Defaults to 0.0.
            delta_synthetic (float, optional): Delta synthetic value. Defaults to 0.0.
            delta_external (float, optional): Delta external value. Defaults to 0.0.
            patience (int, optional): Patience for early stopping. Defaults to 100.
            model_path (str, optional): Path to save the model. Defaults to './Models/checkpoint_'.
            std_syntethic (float, optional): Standard deviation for synthetic data. Defaults to 0.0.
            std_growth (float, optional): Standard deviation growth. Defaults to 0.0.
            epsilon (float, optional): Epsilon value. Defaults to 0.0.
            epsilon_synthetic (float, optional): Epsilon synthetic value. Defaults to 0.0.
            external_points (torch.Tensor, optional): External points for training. Defaults to None.
            seed (int, optional): Random seed. Defaults to None.
            keep_model (bool, optional): Whether to keep the model. Defaults to False.
            _early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
    
        Returns:
            None
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
    
        train_losses = []
        train_losses_modified = []
        valid_losses = []
        avg_train_losses = []
        avg_train_losses_modified = []
        avg_penalization_losses = []
        avg_valid_losses = []
    
        if _early_stopping:
            print('Using early stopping')
            data_folder = os.path.join("..", "Models")
            os.makedirs(data_folder, exist_ok=True)
            model_path_timestamp = Utilities.add_timestamp(model_path)
            path = model_path_timestamp + '.pt'
            early_stopping = EarlyStopping(path=path, patience=patience, verbose=False)
    
        if optimizer_type == 'SGD':
            optimizer = SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'Adam':
            optimizer = Adam(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'LBFGS':
            optimizer = LBFGS(self._model.parameters(), lr=learning_rate)
    
        n_vars = len(monotone_relations)
        var_mono_crec = Utilities.find_index(monotone_relations, 1)
        var_mono_decrec = Utilities.find_index(monotone_relations, -1)
    
        pbar = tqdm(range(n_epochs + 1))
        for epoch in pbar:
            self._model.train()
            for i, batch in enumerate(train_data):
                if isinstance(batch, list) and len(batch) == 3:
                    batch_input1, batch_input2, targets = batch
                    inputs = (batch_input1, batch_input2)
                else:
                    inputs, targets = batch
                if optimizer_type == 'Adam':
                    optimizer.zero_grad()
                    yhat = self._model(inputs)
                    loss = criterion(yhat, targets)
    
                    if delta > 0:
                        jacob = self.batch_jacobian(inputs)
                        adjusted_loss_crec = torch.sum(torch.relu(-jacob[0, :, var_mono_crec] + epsilon))
                        adjusted_loss_decrec = torch.sum(torch.relu(jacob[0, :, var_mono_decrec] + epsilon))
                    else:
                        adjusted_loss_crec = 0.0
                        adjusted_loss_decrec = 0.0
    
                    if delta_synthetic > 0:
                        with torch.no_grad():
                            synthetic_data = Utilities.generate_synthetic_tensor_categorical(inputs, 100, categorical_columns, mean=0, std=std_syntethic)
                        jacob_synthetic = self.batch_jacobian(synthetic_data)
                        adjusted_loss_crec_synthetic = torch.sum(torch.relu(-jacob_synthetic[0, :, var_mono_crec] + epsilon))
                        adjusted_loss_decrec_synthetic = torch.sum(torch.relu(jacob_synthetic[0, :, var_mono_decrec] + epsilon))
                        loss_modified_synthetic = delta_synthetic * (adjusted_loss_decrec_synthetic + adjusted_loss_crec_synthetic)
                    else:
                        loss_modified_synthetic = 0.0
    
                    if delta_external > 0:
                        jacob_external = self.batch_jacobian(external_points)
                        adjusted_loss_crec_external = torch.sum(torch.relu(-jacob_external[0, :, var_mono_crec] + epsilon))
                        adjusted_loss_decrec_external = torch.sum(torch.relu(jacob_external[0, :, var_mono_decrec] + epsilon))
                        loss_modified_external = delta_external * (adjusted_loss_decrec_external + adjusted_loss_crec_external)
                    else:
                        loss_modified_external = 0.0
    
                    with torch.no_grad():
                        if loss_modified_synthetic < epsilon_synthetic:
                            std_syntethic = std_syntethic + std_growth
                    loss_modified = loss + delta * (adjusted_loss_decrec + adjusted_loss_crec) + loss_modified_synthetic + loss_modified_external
    
                def closure():
                    optimizer.zero_grad()
                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)
    
                    if delta > 0:
                        jacob = self.batch_jacobian(inputs)
                        adjusted_loss_crec = torch.sum(torch.relu(-jacob[0, :, var_mono_crec] + epsilon))
                        adjusted_loss_decrec = torch.sum(torch.relu(jacob[0, :, var_mono_decrec] + epsilon))
                    else:
                        adjusted_loss_crec = 0.0
                        adjusted_loss_decrec = 0.0
    
                    if delta_synthetic > 0:
                        with torch.no_grad():
                            synthetic_data = Utilities.generate_synthetic_tensor_categorical(inputs, 100, categorical_columns, mean=0, std=std_syntethic)
                        jacob_synthetic = self.batch_jacobian(synthetic_data)
                        adjusted_loss_crec_synthetic = torch.sum(torch.relu(-jacob_synthetic[0, :, var_mono_crec] + epsilon))
                        adjusted_loss_decrec_synthetic = torch.sum(torch.relu(jacob_synthetic[0, :, var_mono_decrec] + epsilon))
                        loss_modified_synthetic = delta_synthetic * (adjusted_loss_decrec_synthetic + adjusted_loss_crec_synthetic)
                    else:
                        loss_modified_synthetic = 0.0
    
                    if delta_external > 0:
                        jacob_external = self.batch_jacobian(external_points)
                        adjusted_loss_crec_external = torch.sum(torch.relu(-jacob_external[0, :, var_mono_crec] + epsilon))
                        adjusted_loss_decrec_external = torch.sum(torch.relu(jacob_external[0, :, var_mono_decrec] + epsilon))
                        loss_modified_external = delta_external * (adjusted_loss_decrec_external + adjusted_loss_crec_external)
                    else:
                        loss_modified_external = 0.0
    
                    with torch.no_grad():
                        if loss_modified_synthetic < epsilon_synthetic:
                            std_syntethic = std_syntethic + std_growth
                    loss_modified = loss + delta * (adjusted_loss_decrec + adjusted_loss_crec) + loss_modified_synthetic + loss_modified_external
                    loss_modified.backward()
                    train_losses.append(loss.item())
                    train_losses_modified.append(loss_modified.item())
                    return loss_modified
    
                if optimizer_type == 'LBFGS':
                    optimizer.step(closure)
                else:
                    loss_modified.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                    train_losses_modified.append(loss_modified.item())
    
            self._model.eval()
            with torch.no_grad():
                for batch in val_data:
                    if isinstance(batch, list) and len(batch) == 3:
                        x1, x2, y = batch
                        x = (x1, x2)
                    else:
                        x, y = batch
                    out = self._model(x)
                    loss_val = criterion(out, y)
                    valid_losses.append(loss_val.item())
    
            train_loss = np.average(train_losses)
            train_loss_modified = np.average(train_losses_modified)
            penalization_loss = np.average(train_losses_modified) - np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_train_losses_modified.append(train_loss_modified)
            avg_penalization_losses.append(penalization_loss)
            avg_valid_losses.append(valid_loss)
    
            with torch.no_grad():
                epoch_len = len(str(n_epochs))
                if verbose == 2:
                    if epoch % n_visualized == 0:
                        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                        pbar.set_description(print_msg, end=' ')
                        pbar.set_postfix('Train Loss %f, Train Loss Mod %f, Val Loss %f' % (float(train_loss), float(train_loss_modified), float(valid_loss)))
                        for i in range(n_vars):
                            if i in var_mono_crec:
                                pbar.set_postfix({'Minimum Jacobian x_{}'.format(i + 1): float(jacob[0, :, i].min())})
                            elif i in var_mono_decrec:
                                pbar.set_postfix({'Maximum Jacobian x_{}'.format(i + 1): float(jacob[0, :, i].max())})
                            else:
                                pbar.set_postfix({'Min Jacobian x_{}'.format(i + 1): float(jacob[0, :, i].min()), 'Max Jacobian x_{}'.format(i + 1): float(jacob[0, :, i].max())})
                elif verbose == 1:
                    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                    pbar.set_description(print_msg)
                    pbar.set_postfix({'Train Loss': float(train_loss), 'Train Loss Mod': float(train_loss_modified), 'Val Loss': float(valid_loss)})
                else:
                    if epoch == n_epochs:
                        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]')
                        pbar.set_description(print_msg)
                        pbar.set_postfix({'Train Loss': float(train_loss), 'Train Loss Mod': float(train_loss_modified), 'Val Loss': float(valid_loss)})
    
            train_losses = []
            train_losses_modified = []
            valid_losses = []
    
            if _early_stopping:
                early_stopping(valid_loss, penalization_loss, self._model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch %d" % (epoch))
                    break
    
        if _early_stopping:
            self._model.load_state_dict(torch.load(path))
            if not keep_model:
                os.remove(path)
    
        self._avg_train_losses = avg_train_losses
        self._avg_train_losses_modified = avg_train_losses_modified
        self.avg_penalization_losses = avg_penalization_losses
        self._avg_valid_losses = avg_valid_losses
        
    def plot_history(self,figsize=(10,5)):
        """
        This method generates a plot of the training and validation loss values of the neural network model over the epochs during training.
        The plot also includes a vertical line indicating the epoch with the lowest validation loss, which can be used as an early stopping checkpoint.

        Returns:
            None

        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(range(1,len(self._avg_train_losses)+1),self._avg_train_losses, label='Training Loss')
        ax1.plot(range(1,len(self._avg_valid_losses)+1),self._avg_valid_losses,label='Validation Loss')

        # find position of lowest validation loss
        minposs = self._avg_valid_losses.index(min(self._avg_valid_losses))+1 
        ax1.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss (log)')
        ax1.set_yscale('log')
        ax1.set_xlim(0, len(self._avg_train_losses)+1) # consistent scale
        ax1.grid(True)
        ax1.legend()

        diferences = [a_i - b_i for a_i, b_i in zip(self._avg_train_losses_modified, self._avg_train_losses)]
        print(len(diferences))
        ax2.plot(range(1,len(self._avg_train_losses_modified)+1),diferences, label='Modified Training Loss')
        ax2.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('Loss with penalization')
        # ax2.set_yscale('log')
        ax2.set_xlim(0, len(self._avg_train_losses_modified)+1) # consistent scale
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    
    def predict(self, inputs):
        """
        Make predictions using the model.
    
        Args:
            inputs (torch.Tensor): Input data.
    
        Returns:
            numpy.ndarray: Predictions stored in a numpy array.
        """
        self._model.eval()
        with torch.no_grad():
            predictions = self._model(inputs)
        return predictions
    
    def _plot_results(self, n_column, X_tensor, Y_tensor):
        """
        Plot the output with respect to the variable x(n_column).
    
        Args:
            n_column (int): Column index.
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
        """
        print('------------------ Graphical Representation ------------------')
        plt.scatter(X_tensor.numpy()[:, n_column], Y_tensor.numpy().reshape(-1), color='blue', label='True')
        plt.scatter(X_tensor.numpy()[:, n_column], self.predict(X_tensor).reshape(-1), label='Pred', color='orange')
        plt.legend()
        plt.show()
    
    def _plot_surface_interactive(self, X_tensor, y_tensor, n_var_1, n_var_2):
        """
        Plot an interactive 3D surface of the model's predictions with respect to two variables.
    
        Args:
            X_tensor (torch.Tensor): Input tensor.
            y_tensor (torch.Tensor): Output tensor.
            n_var_1 (int): Index of the first variable.
            n_var_2 (int): Index of the second variable.
        """
        # Get the minimum and maximum values of each variable
        x1_min, _ = torch.min(X_tensor[:, n_var_1], dim=0)
        x1_max, _ = torch.max(X_tensor[:, n_var_1], dim=0)
        x2_min, _ = torch.min(X_tensor[:, n_var_2], dim=0)
        x2_max, _ = torch.max(X_tensor[:, n_var_2], dim=0)
    
        # Create the data grid
        x1, x2 = torch.meshgrid(torch.linspace(x1_min, x1_max, 100), torch.linspace(x2_min, x2_max, 100))
    
        # Make predictions for each point on the grid
        X_grid = torch.stack([x1.reshape(-1), x2.reshape(-1)], axis=1)
        y_grid = self._model(X_grid).reshape(x1.shape).detach().numpy()
    
        # Create the Plotly figure
        fig = go.Figure()
    
        # Add the surface
        fig.add_trace(go.Surface(x=x1, y=x2, z=y_grid, opacity=0.5))
    
        # Add the points evaluated by self._model as spheres
        y_pred = self._model(X_tensor).detach().numpy().squeeze()
        fig.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_pred, mode='markers',
                                   marker=dict(
                                       size=5,
                                       color='blue',
                                       opacity=1.0,
                                       line=dict(
                                           color='black',
                                           width=0.5
                                       ),
                                       symbol='circle'
                                   ),
                                   name='Predicted Values', opacity=0.5
                                   ))
        # Add the real points
        fig.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_tensor[:, 0], mode='markers',
                                   marker=dict(
                                       size=5,
                                       color='orange',
                                       opacity=1.0,
                                       line=dict(
                                           color='black',
                                           width=0.5
                                       ),
                                       symbol='circle'
                                   ),
                                   name='True Values'
                                   ))
        # Customize the figure
        fig.update_layout(
            title='Prediction Surface',
            width=800,
            height=1000,
            legend=dict(
                title=dict(text='Legend'),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="white",
                bordercolor="Black",
                borderwidth=2
            ),
            scene=dict(
                xaxis_title='x1',
                yaxis_title='x2',
                zaxis_title='y',
                xaxis=dict(range=[x1_min, x1_max]),
                yaxis=dict(range=[x2_min, x2_max]),
                aspectmode='cube',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            )
        )
    
        # Show the figure
        fig.show()


    def _plot_jacobian_interactive(self, X_tensor, n_var_1, n_var_2, var_rep):
        """
        Plot an interactive 3D surface of the Jacobian's partial derivatives with respect to two variables.
    
        Args:
            X_tensor (torch.Tensor): Input tensor.
            n_var_1 (int): Index of the first variable.
            n_var_2 (int): Index of the second variable.
            var_rep (int): Variable for which the Jacobian output is represented (Which partial derivatives we want to show).
        """
        # Get the minimum and maximum values of each variable
        x1_min, _ = torch.min(X_tensor[:, n_var_1], dim=0)
        x1_max, _ = torch.max(X_tensor[:, n_var_1], dim=0)
        x2_min, _ = torch.min(X_tensor[:, n_var_2], dim=0)
        x2_max, _ = torch.max(X_tensor[:, n_var_2], dim=0)
    
        # Create the data grid
        x1, x2 = torch.meshgrid(torch.linspace(x1_min, x1_max, 100), torch.linspace(x2_min, x2_max, 100))
    
        # Make predictions for each point on the grid
        X_grid = torch.stack([x1.reshape(-1), x2.reshape(-1)], axis=1)
        y_grid = self.batch_jacobian(X_grid)
        y_grid = y_grid[0, :, var_rep].reshape(x1.shape).detach().numpy()
    
        # Create the Plotly figure
        fig2 = go.Figure()
    
        # Add the surface
        fig2.add_trace(go.Surface(x=x1, y=x2, z=y_grid, opacity=0.5, showscale=False))
    
        # Add the points evaluated by self._model as spheres
        y_pred = self.batch_jacobian(X_tensor)
        y_pred_plot = y_pred[0, :, var_rep].reshape(X_tensor[:, n_var_1].shape).detach().numpy()
        fig2.add_trace(go.Scatter3d(x=X_tensor[:, n_var_1], y=X_tensor[:, n_var_2], z=y_pred_plot, mode='markers',
                                    marker=dict(
                                        size=5,
                                        color='blue',
                                        opacity=1.0,
                                        line=dict(
                                            color='black',
                                            width=0.5
                                        ),
                                        symbol='circle'
                                    ),
                                    name='Prediction'
                                    ))
    
        # Add the plane at z=0
        z_plane = np.zeros_like(y_grid)
        fig2.add_trace(go.Surface(x=x1, y=x2, z=z_plane, opacity=0.5, showscale=False))
    
        # Customize the figure
        fig2.update_layout(
            title=dict(text=r'$ \text{Partial Derivative of the } NN$', pad=dict(t=0)),
            scene=dict(
                xaxis_title='t',
                yaxis_title='x',
                zaxis_title='z',
                xaxis=dict(range=[x1_min, x1_max]),
                yaxis=dict(range=[x2_min, x2_max]),
                aspectratio=dict(x=1, y=1, z=1.0),
                camera_eye=dict(x=-1.5, y=-1.5, z=0.5)
            ),
            width=1000,
            height=800
        )
    
        # Show the figure
        fig2.show()
  