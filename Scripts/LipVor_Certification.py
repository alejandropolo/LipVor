import torch
import numpy as np
from scipy.spatial import Voronoi
import importlib
import LipVor_functions
from itertools import product

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

def LipVorMonotoneCertification(
    model, actfunc, weights, biases, monotone_relations, variable_index, global_lipschitz_constant,
    X_train_tensor, original_points, intervals, max_iterations=100, n=1, epsilon_derivative=0.1,
    epsilon_proyection=0.01, probability=0.1, seed=1, verbose=0, julia=False,
    categorical_indices=None, categorical_values=None, plot_voronoi=False
):
    """
    Certify the model using the LipVor algorithm for both numerical and categorical variables.

    For categorical variables, we skip Voronoi in those dimensions, computing Voronoi
    only in the numerical subspace for each possible categorical combination.
    """
#     if verbose:
#         print("""
#           _____                _____                _____                    _____                   _______                   _____          
#          /\    \              /\    \              /\    \                  /\    \                 /::\    \                 /\    \         
#         /::\____\            /::\____\            /::\    \                /::\____\               /::::\    \               /::\    \        
#        /:::/    /           /:::/    /           /::::\    \              /:::/    /              /::::::\    \             /::::\    \       
#       /:::/    /           /:::/    /           /::::::\    \            /:::/    /              /::::::::\    \           /::::::\    \      
#      /:::/    /           /:::/    /           /:::/\:::\    \          /:::/    /              /:::/--\:::\    \         /:::/\:::\    \     
#     /:::/    /           /:::/    /           /:::/__\:::\    \        /:::/____/              /:::/    \:::\    \       /:::/__\:::\    \    
#    /:::/    /           /:::/    /           /::::\   \:::\    \       |::|    |              /:::/    / \:::\    \     /::::\   \:::\    \   
#   /:::/    /           /:::/    /           /::::::\   \:::\    \      |::|    |     _____   /:::/____/   \:::\____\   /::::::\   \:::\    \  
#  /:::/    /           /:::/    /           /:::/\:::\   \:::\____\     |::|    |    /\    \ |:::|    |     |:::|    | /:::/\:::\   \:::\____\ 
# /:::/____/           /:::/    /           /:::/  \:::\   \:::|    |    |::|    |   /::\____\|:::|____|     |:::|    |/:::/  \:::\   \:::|    |
# \:::\    \          /:::/    /            \::/    \:::\  /:::|____|    |::|    |  /:::/    / \:::\    \   /:::/    / \::/   |::::\  /:::|____|
#  \:::\    \        /:::/    /              \/_____/\:::\/::::/    /    |::|    | /:::/    /   \:::\    \ /:::/    /   \/____|:::::\/:::/    / 
#   \:::\    \       \::/    /                         \::::::/    /     |::|____|/:::/    /     \:::\    /:::/    /          |:::::::::/    /  
#    \:::\    \       \/____/                           \::::/    /      |:::::::::::/    /       \:::\__/:::/    /           |::|\::::/    /   
#     \:::\    \                                         \::/____/        \:::::::::/____/         \::::::::/    /            |::| \::/____/    
#      \:::\    \                                         --               ---------                \::::::/    /             |::|  -|          
#       \:::\    \                                                                                   \::::/    /              |::|   |          
#        \:::\____\                                                                                   \::/____/               \::|   |          
#         \::/    /                                                                                    --                      \:|   |          
#          \/____/                                                                                                              \|___|          
                                                                                                                                       
#     """)
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_variables = X_train_tensor.shape[1]

    # Check that the categorical indices are valid
    if categorical_indices:
        categorical_indices = list(categorical_indices)
        if any(idx in categorical_indices for idx in variable_index):
            raise ValueError("Monotonic relationships cannot be imposed on categorical variables")
        # Identify the numeric indices, excluding any categorical indices
        numeric_indices = [i for i in range(X_train_tensor.shape[1]) if i not in (categorical_indices or [])]    

    # If there are categorical variables, handle them by iterating over all possible value combinations
    if categorical_indices and categorical_values:
        all_space_filled = True
        all_x_reent = None

        # Generate all combinations of categorical values
        for combo in product(*[categorical_values[i] for i in categorical_indices]):
            # print(f"Processing categorical combination: {combo}")
            # Create a mask to filter points matching this specific combination
            mask = np.ones(len(original_points), dtype=bool)
            for cidx, cval in zip(categorical_indices, combo):
                mask &= (original_points[:, cidx] == cval)

            # Extract and prepare the numeric part of the filtered points
            sub_points = original_points[mask][:, numeric_indices]
            if len(sub_points) == 0:
                continue
            sub_points = np.unique(sub_points, axis=0).astype(np.float64)
            # TODO: Delete this line or understand why is it neccesary this rounding
            # sub_points = np.round(sub_points, 3).astype(np.float64)
            inputs = torch.tensor(original_points[mask], dtype=torch.float)

            # Generate hypercube vertices for both the original intervals and the extended intervals
            vertices = LipVor_functions.generate_hypercube_vertices(intervals)
            intervals_extended = [(x - epsilon_proyection, y + epsilon_proyection) for x, y in intervals]
            vertices_extended = LipVor_functions.generate_hypercube_vertices(intervals_extended)

            # If not using Julia, compute Voronoi and LipVor in Python
            if not julia:
                # Compute the Voronoi diagram for the initial set of points
                original_sub_vor = Voronoi(sub_points, incremental=True)

                # Compute the symmetric poitns
                all_points, _ = LipVor_functions.add_symmetric_points(
                    original_sub_vor, vertices_extended,
                    intervals_extended
                )
                # Compute the Voronoi diagram with the symmetric points
                finite_vor = Voronoi(all_points, incremental=True)

                # Compute the local Lipschitz radii for each region
                _, dict_radios, _, no_points = LipVor_functions.get_lipschitz_radius_neuralsens(
                    inputs=inputs, outputs=[], weights=weights, biases=biases,
                    actfunc=actfunc, global_lipschitz_constant=global_lipschitz_constant,
                    monotone_relation=monotone_relations,
                    variable_index=variable_index,
                    n_variables=n_variables, epsilon_derivative=epsilon_derivative
                )

                # Check if the space is fully covered
                space_filled, distances = LipVor_functions.check_space_filled_vectorized(
                    finite_vor, dict_radios, vertices
                )
                # if space_filled and no_points:
                #     print('The space is filled: {}. Intervals that define the space: {}'.format(space_filled, intervals))
                x_reentrenamiento = None

                # If not filled, refine Voronoi cells to cover the space
                # TODO: Take out this function from the loop of combo as it LipVor function handle the combos
                if not space_filled:
                    space_filled, x_reentrenamiento = LipVor_functions.LipVor(
                        original_vor=original_sub_vor, original_points=original_points[mask],
                        finite_vor=finite_vor, dict_radios=dict_radios, vertices=vertices,
                        distances=distances, model=model,
                        global_lipschitz_constant=global_lipschitz_constant,
                        actfunc=actfunc,
                        intervals=intervals,
                        monotone_relations=monotone_relations,
                        variable_index=variable_index,
                        n_variables=n_variables,
                        epsilon_derivative=epsilon_derivative, probability=probability,
                        epsilon=epsilon_proyection, max_iterations=max_iterations, verbose=verbose,
                        categorical_indices=categorical_indices, categorical_values=categorical_values,
                        plot_voronoi=plot_voronoi
                    )

                all_space_filled &= space_filled
                # if x_reentrenamiento is not None:
                #     if isinstance(all_x_reent, torch.Tensor):
                #         all_x_reent = torch.cat([all_x_reent, x_reentrenamiento], dim=0)
                #     else:
                #         all_x_reent.append(x_reentrenamiento)

                if x_reentrenamiento is not None and x_reentrenamiento.shape[0] > 0:
                    if not torch.is_tensor(x_reentrenamiento):
                        x_reentrenamiento = torch.tensor(x_reentrenamiento, dtype=torch.float)
                    if all_x_reent is None or all_x_reent.numel() == 0:
                        all_x_reent = x_reentrenamiento
                    else:
                        all_x_reent = torch.cat((all_x_reent, x_reentrenamiento), dim=0)

                # Collect information about the current combination
                if 'combo_results' not in locals():
                    combo_results = []
                combo_results.append({
                    'combo': combo,
                    'space_filled': space_filled,
                    'x_reentrenamiento': x_reentrenamiento
                })
            # If using Julia, compute Voronoi and LipVor through the Julia interface
            else:
                # intervals_extended = [(x - epsilon_proyection, y + epsilon_proyection)
                #                       for x, y in intervals]
                center = np.mean(intervals_extended, axis=1)
                l = float(np.abs(intervals_extended[0][1] - intervals_extended[0][0]))

                # Compute Voronoi structures in Julia
                node_list, vertex_list = jl.JuliaVoronoi(sub_points, l, center, False, False)

                # Compute the local Lipschitz radii for each region
                radius, dict_radios, _, no_points = LipVor_functions.get_lipschitz_radius_neuralsens(
                    inputs=inputs, outputs=[], weights=weights, biases=biases,
                    actfunc=actfunc, global_lipschitz_constant=global_lipschitz_constant,
                    monotone_relation=monotone_relations,
                    variable_index=variable_index, n_variables=n_variables,
                    epsilon_derivative=epsilon_derivative
                )

                # Check if the space is fully covered using Julia
                space_filled, distances, _ = jl.check_space_filled_julia(node_list, radius, vertex_list)
                
                
                # if space_filled and no_points:
                #     print('The space is filled: {}. Intervals that define the space: {}'.format(space_filled, intervals))
                x_reentrenamiento = None

                # If not filled, refine Voronoi cells using Julia
                if not space_filled:
                    space_filled, x_reentrenamiento = LipVor_functions.LipVor_Julia(
                        original_points=original_points[mask], radius=radius, distances=distances,
                        vertices=vertices, model=model, actfunc=actfunc,
                        global_lipschitz_constant=global_lipschitz_constant,
                        intervals=intervals,
                        monotone_relations=monotone_relations,
                        variable_index=variable_index,
                        n_variables=n_variables,
                        epsilon_derivative=epsilon_derivative,
                        epsilon_proyection=epsilon_proyection,
                        probability=probability, mode='neuralsens',
                        plot_voronoi=False, max_iterations=max_iterations,
                        n=n, verbose=verbose,
                        categorical_indices=categorical_indices, categorical_values=categorical_values
                    )
                all_space_filled &= space_filled
                # if x_reentrenamiento is not None:
                #     all_x_reent.append(x_reentrenamiento)
                if x_reentrenamiento is not None and x_reentrenamiento.shape[0] > 0:
                    if not torch.is_tensor(x_reentrenamiento):
                        x_reentrenamiento = torch.tensor(x_reentrenamiento, dtype=torch.float)
                    if all_x_reent is None or all_x_reent.numel() == 0:
                        all_x_reent = x_reentrenamiento
                    else:
                        all_x_reent = torch.cat((all_x_reent, x_reentrenamiento), dim=0)
                
                # Collect information about the current combination
                if 'combo_results' not in locals():
                    combo_results = []
                combo_results.append({
                    'combo': combo,
                    'space_filled': space_filled,
                    'x_reentrenamiento': x_reentrenamiento
                })

        # Return combined results for all categorical combinations
        return all_space_filled, all_x_reent, combo_results
    else:
        # If no categorical variables, do the standard approach
        original_points = np.unique(original_points, axis=0).astype(np.float64)
        # original_points = np.round(original_points, 3).astype(np.float64)
        inputs = torch.tensor(original_points, dtype=torch.float)
        x_reentrenamiento = None
        n_variables = X_train_tensor.shape[1]

        vertices = LipVor_functions.generate_hypercube_vertices(intervals)
        if not julia:
            intervals_extended = [(x - epsilon_proyection, y + epsilon_proyection) for x, y in intervals]
            vertices_extended = LipVor_functions.generate_hypercube_vertices(intervals_extended)
            ## Compute original Voronoi diagram for the initial points
            original_vor = Voronoi(original_points, incremental=True)

            ## Compute symmetric points
            all_points, _ = LipVor_functions.add_symmetric_points(original_vor, vertices_extended, intervals_extended)

            finite_vor = Voronoi(all_points, incremental=True)
            _, dict_radios, _, _ = LipVor_functions.get_lipschitz_radius_neuralsens(
                inputs=inputs, outputs=[], weights=weights, biases=biases, actfunc=actfunc,
                global_lipschitz_constant=global_lipschitz_constant, monotone_relation=monotone_relations,
                variable_index=variable_index, n_variables=n_variables, epsilon_derivative=epsilon_derivative
            )
            space_filled, distances = LipVor_functions.check_space_filled_vectorized(finite_vor, dict_radios, vertices)
            if not space_filled:
                importlib.reload(LipVor_functions)
                space_filled, x_reentrenamiento = LipVor_functions.LipVor(
                    original_vor=original_vor, original_points=original_points, finite_vor=finite_vor, dict_radios=dict_radios,
                    vertices=vertices, distances=distances, model=model, global_lipschitz_constant=global_lipschitz_constant,
                    actfunc=actfunc, intervals=intervals, monotone_relations=monotone_relations,
                    variable_index=variable_index, n_variables=n_variables, plot_voronoi=False,
                    epsilon_derivative=epsilon_derivative, probability=probability, epsilon=epsilon_proyection,
                    max_iterations=max_iterations,verbose=verbose
                    # categorical_indices=categorical_indices, categorical_values=categorical_values
                )
            return space_filled, x_reentrenamiento

        else:
            intervals_extended = [(x - epsilon_proyection, y + epsilon_proyection) for x, y in intervals]
            center = np.mean(intervals_extended, axis=1)
            l = float(np.abs(intervals_extended[0][1] - intervals_extended[0][0]))
            node_list, vertex_list = jl.JuliaVoronoi(original_points, l, center, False, False)
            radius, dict_radios, _, _ = LipVor_functions.get_lipschitz_radius_neuralsens(
                inputs=inputs, outputs=[], weights=weights, biases=biases,
                actfunc=actfunc, global_lipschitz_constant=global_lipschitz_constant,
                monotone_relation=monotone_relations, variable_index=variable_index,
                n_variables=n_variables, epsilon_derivative=epsilon_derivative
            )
            space_filled, distances, _ = jl.check_space_filled_julia(node_list, radius, vertex_list)
            if not space_filled:
                space_filled, x_reentrenamiento = LipVor_functions.LipVor_Julia(
                    original_points=original_points, radius=radius, distances=distances, vertices=vertices,
                    model=model, actfunc=actfunc, global_lipschitz_constant=global_lipschitz_constant,
                    intervals=intervals, monotone_relations=monotone_relations, variable_index=variable_index,
                    n_variables=n_variables, epsilon_derivative=epsilon_derivative,
                    epsilon_proyection=epsilon_proyection, probability=probability, mode='neuralsens',
                    plot_voronoi=False, max_iterations=max_iterations, n=n, verbose=verbose
                )
            return space_filled, x_reentrenamiento
    