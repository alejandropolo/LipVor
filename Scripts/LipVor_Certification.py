import torch
import numpy as np
from scipy.spatial import Voronoi
import importlib
import LipVor_functions

# Load Julia
try:
    from juliacall import Main as jl
    jl.include("../Scripts/JuliaSrc/VorFunctions.jl")
except:
    import warnings
    warnings.warn("Julia could not be loaded. Julia options will not be available.", RuntimeWarning)



def get_intervals(n_variables):
    """
    Get intervals for each variable from user input.

    Parameters:
    n_variables (int): Number of variables.

    Returns:
    list: List of tuples representing the intervals for each variable.
    """
    intervals = []
    for i in range(n_variables):
        x_lim = input(f"Enter the limits for variable x_{i} (format: min,max): ").split(',')
        intervals.append((float(x_lim[0]), float(x_lim[1])))
    return intervals

def LipVorMonotoneCertification(model, actfunc, weights, biases, monotone_relations, variable_index, global_lipschitz_constant,
                X_train_tensor, original_points, intervals, max_iterations=100, n = 1,
                epsilon_derivative=0.1,epsilon_proyection=0.01, probability=0.1, seed=1, verbose=0, julia=False):
    """
    Certify the model using the LipVor algorithm.

    Parameters:
    model (torch.nn.Module): The neural network model.
    actfunc (str): Activation function used in the model.
    weights (list): List of weights of the model.
    biases (list): List of biases of the model.
    monotone_relations (list): List of monotone relations for each variable.
    global_lipschitz_constant (float): Global Lipschitz constant.
    X_train_tensor (torch.Tensor): Training data tensor.
    original_points (np.ndarray): Array of original points.
    intervals (list): List of tuples representing the intervals for each variable.
    n_initial_points (int): Number of initial points to select randomly. Default is 1.
    max_iterations (int): Maximum number of iterations for the LipVor algorithm. Default is 100.
    epsilon_derivative (float): Epsilon value for derivative computation. Default is 0.1.
    epsilon_proyection (float): Epsilon value for projection. Default is 0.01.
    probability (float): Probability value for the LipVor algorithm. Default is 0.1.
    seed (int): Seed for random number generation. Default is 1.
    """
    
    print("""
          _____                _____                _____                    _____                   _______                   _____          
         /\    \              /\    \              /\    \                  /\    \                 /::\    \                 /\    \         
        /::\____\            /::\____\            /::\    \                /::\____\               /::::\    \               /::\    \        
       /:::/    /           /:::/    /           /::::\    \              /:::/    /              /::::::\    \             /::::\    \       
      /:::/    /           /:::/    /           /::::::\    \            /:::/    /              /::::::::\    \           /::::::\    \      
     /:::/    /           /:::/    /           /:::/\:::\    \          /:::/    /              /:::/--\:::\    \         /:::/\:::\    \     
    /:::/    /           /:::/    /           /:::/__\:::\    \        /:::/____/              /:::/    \:::\    \       /:::/__\:::\    \    
   /:::/    /           /:::/    /           /::::\   \:::\    \       |::|    |              /:::/    / \:::\    \     /::::\   \:::\    \   
  /:::/    /           /:::/    /           /::::::\   \:::\    \      |::|    |     _____   /:::/____/   \:::\____\   /::::::\   \:::\    \  
 /:::/    /           /:::/    /           /:::/\:::\   \:::\____\     |::|    |    /\    \ |:::|    |     |:::|    | /:::/\:::\   \:::\____\ 
/:::/____/           /:::/    /           /:::/  \:::\   \:::|    |    |::|    |   /::\____\|:::|____|     |:::|    |/:::/  \:::\   \:::|    |
\:::\    \          /:::/    /            \::/    \:::\  /:::|____|    |::|    |  /:::/    / \:::\    \   /:::/    / \::/   |::::\  /:::|____|
 \:::\    \        /:::/    /              \/_____/\:::\/::::/    /    |::|    | /:::/    /   \:::\    \ /:::/    /   \/____|:::::\/:::/    / 
  \:::\    \       \::/    /                         \::::::/    /     |::|____|/:::/    /     \:::\    /:::/    /          |:::::::::/    /  
   \:::\    \       \/____/                           \::::/    /      |:::::::::::/    /       \:::\__/:::/    /           |::|\::::/    /   
    \:::\    \                                         \::/____/        \:::::::::/____/         \::::::::/    /            |::| \::/____/    
     \:::\    \                                         --               ---------                \::::::/    /             |::|  -|          
      \:::\    \                                                                                   \::::/    /              |::|   |          
       \:::\____\                                                                                   \::/____/               \::|   |          
        \::/    /                                                                                    --                      \:|   |          
         \/____/                                                                                                              \|___|          
                                                                                                                                       
    """)

    ## Fix seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## Filter duplicated values
    original_points = np.unique(original_points, axis=0)
    ## Round original_points
    original_points = np.round(original_points, 3).astype(np.float64)
    inputs = torch.tensor(original_points, dtype=torch.float)
    outputs = []

    x_reentrenamiento = None

    ## Infer number of variables from the dimension of X_train_tensor
    n_variables = X_train_tensor.shape[1]

    # Print the interval and the increasing and decreasing set of variables
    print(f"Intervals: {intervals}")
    print(f"Increasing set of variables: {[i for i in range(len(variable_index)) if monotone_relations[i] == 1]}")
    print(f"Decreasing set of variables: {[i for i in range(len(variable_index)) if monotone_relations[i] == -1]}")

    ## Generate vertices for a hypercube (n-dimensional cube) defined by the given interval
    vertices = LipVor_functions.generate_hypercube_vertices(intervals)
    if not julia:
        intervals_extended = [(x - epsilon_proyection, y + epsilon_proyection) for x, y in intervals]
        vertices_extended = LipVor_functions.generate_hypercube_vertices(intervals_extended)
        ## Compute original Voronoi diagram for the initial points
        original_vor = Voronoi(original_points, incremental=True)

        ## Compute symmetric points
        all_points, _ = LipVor_functions.add_symmetric_points(original_vor, vertices_extended, intervals_extended)

        ## Compute Voronoi diagram with symmetric points (and therefore bounded)
        finite_vor = Voronoi(all_points, incremental=True)

        ## Compute the radios for each point
        _, dict_radios, _, no_points = LipVor_functions.get_lipschitz_radius_neuralsens(
        inputs=inputs, outputs=outputs, weights=weights, biases=biases,
        actfunc=actfunc, global_lipschitz_constant=global_lipschitz_constant,
        monotone_relation=monotone_relations, variable_index=variable_index, n_variables=n_variables, epsilon_derivative=epsilon_derivative)

        ## Check if the space is filled
        space_filled, distances = LipVor_functions.check_space_filled_vectorized(finite_vor, dict_radios, vertices)

        # If the space is not filled, add points to the Voronoi diagram using the LipVor algorithm
        if not space_filled:
            ## Add points to the Voronoi diagram
            importlib.reload(LipVor_functions)
            space_filled, x_reentrenamiento = LipVor_functions.LipVor(
                original_vor=original_vor, original_points=original_points, finite_vor=finite_vor, dict_radios=dict_radios,
                vertices=vertices, distances=distances, model=model, global_lipschitz_constant=global_lipschitz_constant,
                actfunc=actfunc, intervals=intervals, monotone_relations=monotone_relations,
                variable_index=variable_index, n_variables=n_variables, plot_voronoi=False, epsilon_derivative=epsilon_derivative,
                probability=probability, epsilon=epsilon_proyection, max_iterations=max_iterations)
        return space_filled, x_reentrenamiento
        
    else:
        intervals_extended = [(x - epsilon_proyection, y + epsilon_proyection) for x, y in intervals]
        # Compute the center as the mid point of the intervals and adjust it proportionally with epsilon_proyection
        # center = np.mean(intervals, axis=1) + epsilon_proyection/2 * np.sign(np.mean(intervals, axis=1))
        # # Compute the length of the interval (Assuming cuboid domains)
        # l = float(np.abs(intervals[0][1] - intervals[0][0])) * (1 + epsilon_proyection)
        center = np.mean(intervals_extended, axis=1)
        l = float(np.abs(intervals_extended[0][1] - intervals_extended[0][0]))
        
        # Compute the Voronoi Diagram
        # node_list,vertex_list = jl.generate_voronoi_nodes_points(original_points, l, center,False,False)
        node_list,vertex_list = jl.JuliaVoronoi(original_points, l, center,False,False)

        ## Compute the radios for each point
        radius, dict_radios, _, no_points = LipVor_functions.get_lipschitz_radius_neuralsens(
        inputs=inputs, outputs=outputs, weights=weights, biases=biases,
        actfunc=actfunc, global_lipschitz_constant=global_lipschitz_constant,
        monotone_relation=monotone_relations, variable_index=variable_index, n_variables=n_variables, epsilon_derivative=epsilon_derivative)

        space_filled, distances, _ = jl.check_space_filled_julia(node_list,radius,vertex_list)
        
        if not space_filled:
            space_filled, x_reentrenamiento=LipVor_functions.LipVor_Julia(original_points=original_points,radius=radius, distances=distances, vertices=vertices,
                          model=model,actfunc=actfunc, global_lipschitz_constant=global_lipschitz_constant,
                          intervals=intervals,monotone_relations=monotone_relations,variable_index=variable_index,
                          n_variables=n_variables, epsilon_derivative= epsilon_derivative, epsilon_proyection=epsilon_proyection, probability=probability, mode='neuralsens',
                          plot_voronoi=False, max_iterations=max_iterations,n = n, verbose=verbose)
            
            # x_reentrenamiento, finite_vor = LipVor_functions.LipVor(
            #     original_vor=original_vor, original_points=original_points, finite_vor=finite_vor, dict_radios=dict_radios,
            #     vertices=vertices, distances=distances, model=model, global_lipschitz_constant=global_lipschitz_constant,
            #     actfunc=actfunc, intervals=intervals, monotone_relations=monotone_relations,
            #     variable_index=variable_index, n_variables=n_variables, plot_voronoi=False, epsilon_derivative=epsilon_derivative,
            #     probability=probability, epsilon=epsilon_proyection, max_iterations=max_iterations)

        return space_filled, x_reentrenamiento, intervals
    