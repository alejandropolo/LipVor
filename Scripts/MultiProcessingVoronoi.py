import gc
import os
import psutil
from itertools import product
import numpy as np
from scipy.spatial import Voronoi


######## SET OF FUNCTIONS NEEDED #######

def free_memory():
    """Free memory before computation."""
    gc.collect()  # Garbage collection
    process = psutil.Process(os.getpid())
    print(f"Memory usage after cleanup: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# def generate_random_points(num_points, d, intervals=None):
#     """Generate random points in d-dimensional space within given intervals."""
#     if intervals is None:
#         return np.random.uniform(0, 1, (num_points, d))
#     else:
#         points = np.zeros((num_points, d))
#         for i in range(d):
#             low, high = intervals[i]
#             points[:, i] = np.random.uniform(low, high, num_points)
#         return points
    
def generate_random_points(num_points, d, intervals=None, categorical_indices=None, categorical_values=None):
    """Generate random points in d-dimensional space within given intervals and handle categorical indices."""
    if categorical_indices is None:
        categorical_indices = []
    if categorical_values is None:
        categorical_values = {}

    points = np.zeros((num_points, d))
    interval_counter = 0
    for i in range(d):
        if i in categorical_indices:
            # For categorical indices, sample random values from the provided categorical values
            points[:, i] = np.random.choice(categorical_values[i], num_points)
        else:
            # For continuous indices, sample uniformly within the intervals
            if intervals is None:
                points[:, i] = np.random.uniform(0, 1, num_points)
                interval_counter += 1
            else:
                low, high = intervals[interval_counter]
                points[:, i] = np.random.uniform(low, high, num_points)
                interval_counter += 1
    return points

def compute_voronoi(points):
    """Compute the Voronoi diagram for a given set of points."""
    return Voronoi(points)

def divide_into_subsquares_old(points, n, d):
    """Divide the points into n^d subsquares within the unit square [0, 1] x [0, 1] and return the intervals of definition for each subsquare."""
    subsquares = [[] for _ in range(n**d)]
    intervals = []

    # Calculate the intervals for each subsquare
    for i in range(n**d):
        # Convert the flat index back to multi-dimensional indices
        indices = np.unravel_index(i, (n,) * d)
        interval = []
        for idx in indices:
            # Calculate the start and end of the interval for each dimension
            start = idx * (1 / n)
            end = (idx + 1) * (1 / n)
            interval.append((start, end))
        intervals.append(interval)

    for point in points:
        # For each point, we calculate which sub-square it belongs to
        indices = []
        for i in range(d):
            # For each dimension, ensure that boundary points are assigned correctly
            if point[i] == 1:  # Boundary condition (x == 1 or y == 1)
                indices.append(n - 1)
            else:
                # Normalize the point's coordinate and map it to the sub-square grid
                index = min(int(point[i] // (1 / n)), n - 1)
                indices.append(index)
        
        # Convert the indices to a flat index for the subsquare
        subsquare_idx = np.ravel_multi_index(indices, (n,) * d)
        
        # Append the point to the appropriate sub-square
        subsquares[subsquare_idx].append(point)

    # Ensure each subsquare has at least d+2 points
    for i, subsquare in enumerate(subsquares):
        while len(subsquare) < d + 2:
            new_point = generate_random_points(1, d, intervals[i])[0]
            # Check if the new point coincides with any existing point in the subsquare
            if not any(np.array_equal(new_point, existing_point) for existing_point in subsquare):
                subsquare.append(new_point)

    # Assert that each point in the subsquare is within the intervals
    for subsquare, interval in zip(subsquares, intervals):
        for point in subsquare:
            assert np.all([interval[dim][0] <= point[dim] <= interval[dim][1] for dim in range(d)]), f"Point {point} is out of interval {interval}"

    # Return both the subsquares and their intervals
    return [np.array(subsquare) for subsquare in subsquares], intervals

def divide_into_subsquares(points, n, d, categorical_indices=None,categorical_values=None):
    """Divide the points into n^(d-number of categorical indices) subsquares within the unit square [0, 1] x [0, 1] and return the intervals of definition for each subsquare."""
    if categorical_indices is None:
        categorical_indices = []

    # Adjust the dimension to exclude categorical indices
    effective_d = d - len(categorical_indices)
    subsquares = [[] for _ in range(n**effective_d)]
    intervals = []

    # Calculate the intervals for each subsquare
    for i in range(n**effective_d):
        # Convert the flat index back to multi-dimensional indices
        indices = np.unravel_index(i, (n,) * effective_d)
        interval = []
        dim_counter = 0
        for dim in range(d):
            if dim in categorical_indices:
                continue
            # Calculate the start and end of the interval for each dimension
            start = indices[dim_counter] * (1 / n)
            end = (indices[dim_counter] + 1) * (1 / n)
            interval.append((start, end))
            dim_counter += 1
        intervals.append(interval)

    for point in points:
        # For each point, we calculate which sub-square it belongs to
        indices = []
        dim_counter = 0
        for dim in range(d):
            if dim in categorical_indices:
                continue
            # For each dimension, ensure that boundary points are assigned correctly
            if point[dim] == 1:  # Boundary condition (x == 1 or y == 1)
                indices.append(n - 1)
            else:
                # Normalize the point's coordinate and map it to the sub-square grid
                index = min(int(point[dim] // (1 / n)), n - 1)
                indices.append(index)
            dim_counter += 1
        
        # Convert the indices to a flat index for the subsquare
        subsquare_idx = np.ravel_multi_index(indices, (n,) * effective_d)
        
        # Append the point to the appropriate sub-square
        subsquares[subsquare_idx].append(point)

    # Ensure each subsquare has at least effective_d+2 points
    for i, subsquare in enumerate(subsquares):
        categorical_combos = list(product(*[categorical_values[i] for i in categorical_indices]))
        combo_counts = {combo: 0 for combo in categorical_combos}

        # Count existing points for each combo
        for point in subsquare:
            combo = tuple(point[idx] for idx in categorical_indices)
            if combo in combo_counts:
                combo_counts[combo] += 1

        for combo in categorical_combos:
            while combo_counts[combo] < effective_d + 2:
                new_point = generate_random_points(1, d - len(categorical_indices), intervals[i], None, None)[0]
                # Append the categorical values of the combo to the new point in the correct indices
                for j, idx in enumerate(categorical_indices):
                    new_point = np.insert(new_point, idx, combo[j])
                # Check if the new point coincides with any existing point in the subsquare
                if not any(np.array_equal(new_point, existing_point) for existing_point in subsquare):
                    subsquare.append(new_point)
                    combo_counts[combo] += 1
        # while len(subsquare) < effective_d + 2:
        #     new_point = generate_random_points(1, d, intervals[i],categorical_indices,categorical_values)[0]
        #     # Check if the new point coincides with any existing point in the subsquare
        #     if not any(np.array_equal(new_point, existing_point) for existing_point in subsquare):
        #         subsquare.append(new_point)

    # Assert that each point in the subsquare is within the intervals
    for subsquare, interval in zip(subsquares, intervals):
        for point in subsquare:
            dim_counter = 0
            for dim in range(d):
                if dim in categorical_indices:
                    continue
                assert interval[dim_counter][0] <= point[dim] <= interval[dim_counter][1], f"Point {point} is out of interval {interval}"
                dim_counter += 1

    # Return both the subsquares and their intervals
    return [np.array(subsquare) for subsquare in subsquares], intervals
