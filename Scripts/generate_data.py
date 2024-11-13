import numpy as np
import torch
from sklearn.model_selection import train_test_split
import yaml
import os 
### CARGAR EL YAML DE CONFIGURACIÓN
path = '../Scripts/config.yaml'
with open(path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def generate_data_Neumann(L, k, N_train, N_test, T,seed,boundary_condition=lambda t: t**3):
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Noise
    noise = np.random.normal(0,config['data_load']['noise_constant'],config['data_load']['n_samples'])

    # Initial condition
    initial_condition = lambda x: 0.0

    Nt = 100000         # Number of time steps
    dx = L / (N_test - 1)
    dt = T / Nt

    x_test = np.linspace(0, L, N_test)
    t_test = np.linspace(0, T, N_test)
    u = np.zeros((N_test, Nt))

    # Set initial condition
    u[:, 0] = initial_condition(x_test)

    # Time-stepping loop
    for n in range(Nt):
        for i in range(1, N_test-1):
            u[i, n] = u[i, n-1] + k * dt / dx**2 * (u[i+1, n-1] - 2*u[i, n-1] + u[i-1, n-1])
        
        # Apply boundary condition
        u[0, n] = boundary_condition(n * dt)
        u[-1, n] = boundary_condition(n * dt)
    
    ## Reduce Dimensionality to get an array of shape (N_test, N_test)
    indices = np.linspace(0, len(u[0])-1, N_test, dtype=int)  
    u_test = u[:, indices]  

    T_mesh_test, X_mesh_test = np.meshgrid(t_test, x_test)
    mesh_test = np.hstack((T_mesh_test.reshape(-1, 1), X_mesh_test.reshape(-1, 1)))
    input_mesh_test = torch.tensor(mesh_test).float()
    y_test = torch.tensor(u_test).reshape(len(input_mesh_test), 1).float()

    ######### TRAIN
    Nt = 100000         # Number of time steps
    dx = L / (N_train - 1)
    dt = T / Nt

    x_train = np.linspace(0, L, N_train)
    t_train = np.linspace(0, T, N_train)
    u = np.zeros((N_train, Nt))

    # Set initial condition
    u[:, 0] = initial_condition(x_train)

    # Time-stepping loop
    for n in range(Nt):
        for i in range(1, N_train-1):
            u[i, n] = u[i, n-1] + k * dt / dx**2 * (u[i+1, n-1] - 2*u[i, n-1] + u[i-1, n-1])
        
        # Apply boundary condition
        u[0, n] = boundary_condition(n * dt)
        u[-1, n] = boundary_condition(n * dt)
    
    ## Reduce Dimensionality to get an array of shape (N_train, N_train)
    indices = np.linspace(0, len(u[0])-1, N_train, dtype=int)  
    u_train = u[:, indices]  

    T_mesh_train, X_mesh_train = np.meshgrid(t_train, x_train)
    mesh_train = np.hstack((T_mesh_train.reshape(-1, 1), X_mesh_train.reshape(-1, 1)))
    input_mesh_train = torch.tensor(mesh_train).float()
    y_train = torch.tensor(u_train).reshape(len(input_mesh_train), 1).float()

    # Save the data
    data_folder = "../Data"
    os.makedirs(data_folder, exist_ok=True)
    X_train_data_path = os.path.join(data_folder, "X_train_data.pt")
    y_train_data_path = os.path.join(data_folder, "y_train_data.pt")
    X_test_data_path = os.path.join(data_folder, "X_test_data.pt")
    y_test_data_path = os.path.join(data_folder, "y_test_data.pt")
    X_mesh_data_train_path = os.path.join(data_folder, "X_mesh_train_data.pt")
    T_mesh_data_train_path = os.path.join(data_folder, "T_mesh_train_data.pt")
    X_mesh_data_test_path = os.path.join(data_folder, "X_mesh_test_data.pt")
    T_mesh_data_test_path = os.path.join(data_folder, "T_mesh_test_data.pt")

    torch.save(input_mesh_train, X_train_data_path)
    torch.save(y_train, y_train_data_path)
    torch.save(input_mesh_test, X_test_data_path)
    torch.save(y_test, y_test_data_path)
    torch.save(X_mesh_train, X_mesh_data_train_path)
    torch.save(T_mesh_train, T_mesh_data_train_path)
    torch.save(X_mesh_test, X_mesh_data_test_path)
    torch.save(T_mesh_test, T_mesh_data_test_path)
