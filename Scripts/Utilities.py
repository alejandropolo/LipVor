import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import random
import time
import shutil
import os
import torch
from torch.autograd import grad
from torch.autograd.functional import hessian


def print_errors(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, log=False, config=None, output_scaler=None):
    """
    Print the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) for training, validation, and test sets.

    Args:
        model (torch.nn.Module): The neural network model.
        X_train_tensor (torch.Tensor): Training input tensor.
        y_train_tensor (torch.Tensor): Training target tensor.
        X_val_tensor (torch.Tensor): Validation input tensor.
        y_val_tensor (torch.Tensor): Validation target tensor.
        X_test_tensor (torch.Tensor): Test input tensor.
        y_test_tensor (torch.Tensor): Test target tensor.
        log (bool, optional): Whether to log the results. Defaults to False.
        output_scaler (sklearn.preprocessing.StandardScaler, optional): Scaler used to inverse transform the outputs. Defaults to None.
        config (dict, optional): Configuration dictionary. Defaults to None.

    Returns:
        None
    """
    device = next(model.parameters()).device  

    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    y_train_pred = model(X_train_tensor)
    y_test_pred = model(X_test_tensor)
    y_val_pred = model(X_val_tensor)

    if output_scaler is not None:
        y_train_tensor = torch.tensor(output_scaler.inverse_transform(y_train_tensor.cpu().numpy())).to(device)
        y_test_tensor = torch.tensor(output_scaler.inverse_transform(y_test_tensor.cpu().numpy())).to(device)
        y_val_tensor = torch.tensor(output_scaler.inverse_transform(y_val_tensor.cpu().numpy())).to(device)
        y_train_pred = torch.tensor(output_scaler.inverse_transform(y_train_pred.detach().cpu().numpy())).to(device)
        y_test_pred = torch.tensor(output_scaler.inverse_transform(y_test_pred.detach().cpu().numpy())).to(device)
        y_val_pred = torch.tensor(output_scaler.inverse_transform(y_val_pred.detach().cpu().numpy())).to(device)

    
    mse_train = torch.mean((y_train_tensor - y_train_pred)**2)
    mse_test = torch.mean((y_test_tensor - y_test_pred)**2)
    mse_val = torch.mean((y_val_tensor - y_val_pred)**2)
    rmse_train = torch.sqrt(mse_train)
    rmse_test = torch.sqrt(mse_test)
    rmse_val = torch.sqrt(mse_val)
    mae_train = torch.mean(torch.abs(y_train_tensor - y_train_pred))
    mae_test = torch.mean(torch.abs(y_test_tensor - y_test_pred))
    mae_val = torch.mean(torch.abs(y_val_tensor - y_val_pred))
    r2_train = r2_score(y_train_tensor.detach().cpu().numpy(), y_train_pred.detach().cpu().numpy())
    r2_test = r2_score(y_test_tensor.detach().cpu().numpy(), y_test_pred.detach().cpu().numpy())
    r2_val = r2_score(y_val_tensor.detach().cpu().numpy(), y_val_pred.detach().cpu().numpy())
    
    print(f"MSE Train: {np.round(mse_train.item(),7)}, MSE Val: {np.round(mse_val.item(),7)}, MSE Test: {np.round(mse_test.item(),7)}")
    print(f"RMSE Train: {np.round(rmse_train.item(),7)}, RMSE Val: {np.round(rmse_val.item(),7)} , RMSE Test: {np.round(rmse_test.item(),7)}")
    print(f"MAE Train: {np.round(mae_train.item(),7)}, MAE Val: {np.round(mae_val.item(),7)}, MAE Test: {np.round(mae_test.item(),7)}")
    print(f"R2 Train: {np.round(r2_train,7)}, R2 Val: {np.round(r2_val,7)}, R2 Test: {np.round(r2_test,7)}")


def find_index(indexes, value):
    """
    Find the indices of a given value in a list.

    Args:
        indexes (list): List of values.
        value (any): Value to find in the list.

    Returns:
        list: List of indices where the value is found.
    """
    indices = []
    for i in range(len(indexes)):
        if indexes[i] == value:
            indices.append(i)
    return indices


def add_scaled_columns(df, columns):
    """
    Add scaled versions of specified columns to a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        columns (list): List of column names to scale.

    Returns:
        pandas.DataFrame: DataFrame with scaled columns added.
    """
    df_copy = df.copy()
    scaler = StandardScaler()
    for column in columns:
        column_name = column + '_scaled'
        scaled_column = scaler.fit_transform(df_copy[column].values.reshape(-1, 1))
        df_copy.loc[:, column_name] = scaled_column
    return df_copy


def get_accuracy(y_true, y_prob):
    """
    Calculate the accuracy score.

    Args:
        y_true (numpy.ndarray): True labels.
        y_prob (numpy.ndarray): Predicted probabilities.

    Returns:
        float: Accuracy score.
    """
    accuracy = accuracy_score(y_true, y_prob > 0.5)
    return accuracy


def generate_synthetic_tensor(original_tensor, n, mean=0, std=1):
    """
    Generate a synthetic tensor by adding noise to randomly selected rows from the original tensor.

    Args:
        original_tensor (torch.Tensor): Original tensor.
        n (int): Number of synthetic samples to generate.
        mean (float, optional): Mean of the noise. Defaults to 0.
        std (float, optional): Standard deviation of the noise. Defaults to 1.

    Returns:
        torch.Tensor: Synthetic tensor.
    """
    synthetic_tensor = torch.empty((n, original_tensor.shape[1]))
    row_indices = torch.randint(low=0, high=original_tensor.shape[0], size=(n,))
    synthetic_tensor[:, :] = original_tensor[row_indices, :]
    noise_tensor = torch.normal(mean=mean, std=std, size=(n, original_tensor.shape[1]))
    synthetic_tensor += noise_tensor
    return synthetic_tensor


def generate_synthetic_tensor_categorical(original_tensor, n, categorical_columns, mean=0.0, std=1.0):
    """
    Generate a synthetic tensor by adding noise to randomly selected rows from the original tensor, with special handling for categorical columns.

    Args:
        original_tensor (torch.Tensor): Original tensor.
        n (int): Number of synthetic samples to generate.
        categorical_columns (list): List of indices of categorical columns.
        mean (float, optional): Mean of the noise. Defaults to 0.0.
        std (float, optional): Standard deviation of the noise. Defaults to 1.0.

    Returns:
        torch.Tensor: Synthetic tensor.
    """
    synthetic_tensor = torch.empty((n, original_tensor.shape[1]))
    row_indices = torch.randint(low=0, high=original_tensor.shape[0], size=(n,))
    synthetic_tensor[:, :] = original_tensor[row_indices, :]
    for column in range(original_tensor.shape[1]):
        if column in categorical_columns:
            val_list = random.choices(original_tensor[:, column], k=n)
            synthetic_tensor[:, column] = torch.tensor(val_list)
        else:
            noise_tensor = torch.normal(mean=mean, std=std, size=(n,))
            synthetic_tensor[:, column] += noise_tensor
    return synthetic_tensor


def add_timestamp(string):
    """
    Add a timestamp to a given string.

    Args:
        string (str): Input string.

    Returns:
        str: String with timestamp appended.
    """
    timestamp = str(int(time.time()))
    return f"{string}_{timestamp}"


def batch_hessian(model, input):
    """
    Calculate the Hessian matrix for all samples in a batch.

    Args:
        model (torch.nn.Module): The neural network model.
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Hessian matrix.
    """
    f_sum = lambda x: torch.sum(model(x), axis=0)
    return hessian(f_sum, input, create_graph=True)


def delete_models(path='./Models/'):
    """
    Delete all models in the specified directory and recreate the directory.

    Args:
        path (str, optional): Path to the directory. Defaults to './Models/'.

    Returns:
        None
    """
    shutil.rmtree(path)
    os.mkdir(path)