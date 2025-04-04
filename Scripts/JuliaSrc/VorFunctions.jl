using HighVoronoi
using Random
using Statistics
using PythonCall
using LinearAlgebra

Random.seed!(42)
function JuliaVoronoi(points::AbstractArray, l::Float64, center::AbstractArray, plot_voronoi::Bool = true, py::Bool = false)
    # Redirect stdout and stderr to devnull to suppress all output
    original_stdout = stdout
    original_stderr = stderr
    redirect_stdout(devnull)
    redirect_stderr(devnull)

    

    try
        # Transpose the points
        points = points'

        dim, N = size(points)

        # Assert that the dim is the same as the length of the center
        @assert dim == length(center) "The dimension of the points and the center do not match"

        # Compute Voronoi nodes
        nodes = VoronoiNodes(points)

        # Compute the offset to move the center of the cuboid
        offset = center .- (l / 2)  # Subtract half the edge length from each component of the center

        # Create a domain (cuboid with edge length l and shifted center)
        domain = cuboid(dim; dimensions=l * ones(Float64, dim), periodic=[], offset=offset)

        # Compute Voronoi geometry
        VG = VoronoiGeometry(nodes, domain, integrator=HighVoronoi.VI_GEOMETRY, silence=true)
        vd = VoronoiData(VG)

        # Plot the Voronoi diagram if applicable and plot_voronoi is true
        # if plot_voronoi
        #     if dim == 2
        #         HighVoronoi.draw2D(VG)
        #     elseif dim == 3
        #         HighVoronoi.draw3D(VG)
        #     end
        # end

        # Initialize arrays to store nodes and vertices
        node_list = Vector{Vector{Float64}}()  # Explicitly specify the type
        vertex_list = Vector{Vector{Vector{Float64}}}()  # Explicitly specify the type

        for (i, node) in enumerate(nodes)
            cell_vertices = Vector{Vector{Float64}}()  # Explicitly specify the type
            for vertex in vd.vertices[i]
                push!(cell_vertices, vertex[2])  # Assuming vertex[2] contains coordinates
            end
            push!(node_list, node)
            push!(vertex_list, cell_vertices)
        end

        # Convert Julia arrays to Python objects using PythonCall
        if py
            py_nodes = Py(node_list)
            py_vertices = Py(vertex_list)
            py_nodes = PythonCall.@py "$(py_nodes)"
            py_vertices = PythonCall.@py "$(vertex_list)"
            return node_list, vertex_list, py_nodes, py_vertices
        else
            return node_list, vertex_list
        end

    finally
        # Restore original stdout and stderr
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
    end
end

function compute_furthest_vertices(node_list::Vector{Vector{Float64}}, vertex_list::Vector{Vector{Vector{Float64}}})
    # Initialize arrays to store the furthest vertex and the corresponding distance for each node
    furthest_vertices = Vector{Vector{Float64}}()
    distances = Vector{Float64}()

    # Iterate over each node and its corresponding vertices
    for (i, node) in enumerate(node_list)
        max_distance = -Inf
        furthest_vertex = Vector{Float64}()

        # Iterate over each vertex in the current node's cell
        for vertex in vertex_list[i]
            # Calculate the Euclidean distance between the node and the vertex
            distance = sqrt(sum((node .- vertex).^2))

            # Update the furthest vertex and distance if the current distance is greater
            if distance > max_distance
                max_distance = distance
                furthest_vertex = vertex
            end
        end

        # Store the furthest vertex and its distance for the current node
        push!(furthest_vertices, furthest_vertex)
        push!(distances, max_distance)
    end

    return distances, furthest_vertices
end

function check_space_filled_julia(node_list::Vector{Vector{Float64}}, radius_array::AbstractArray, vertex_list::Vector{Vector{Vector{Float64}}})
    # Initialize arrays to store the furthest vertex and the corresponding distance for each node
    furthest_vertices = Vector{Vector{Float64}}()
    distances = Vector{Float64}()

    # Iterate over each node and its corresponding vertices
    for (i, node) in enumerate(node_list)
        max_distance = -Inf
        furthest_vertex = Vector{Float64}()

        # Iterate over each vertex in the current node's cell
        for vertex in vertex_list[i]
            # Calculate the Euclidean distance between the node and the vertex
            distance = sqrt(sum((node .- vertex).^2))

            # Update the furthest vertex and distance if the current distance is greater
            if distance > max_distance
                max_distance = distance
                furthest_vertex = vertex
            end
        end

        # Store the furthest vertex and its distance for the current node
        push!(furthest_vertices, furthest_vertex)
        push!(distances, max_distance)
    end

    # Check if all distances are within the radius
    if all(distances .< radius_array)
        return true, distances, furthest_vertices
    else
        return false, distances, furthest_vertices
    end
end

function add_new_point_julia(node_list::Vector{Vector{Float64}}, vertex_list::Vector{Vector{Vector{Float64}}}, distances::AbstractArray, radios::AbstractArray, probability::Float64)
    min_covered_points = Inf
    max_radio = -Inf
    min_radio = Inf
    selected_vertex_max = nothing
    selected_vertex_min = nothing

    # Ensure the number of points inside the Voronoi set matches the distances and radios
    @assert length(node_list) == length(distances) == length(radios) "The number of points inside the Voronoi set is not the same as the distances and the radios"

    # Check that not all the radios are greater than the distances
    @assert !all(radios .>= distances) "The Voronoi cell is already filled"

    # Loop over the distances and the radios
    for i in eachindex(distances)
        # Check if the distance is greater than the radio (if not, the Voronoi cell is already filled)
        if distances[i] > radios[i]
            # Extract the point and its corresponding vertices
            point = node_list[i]
            region_vertices = vertex_list[i]

            # Extract the furthest vertex
            furthest_vertex = region_vertices[argmax([norm(vertex - point) for vertex in region_vertices])]

            # Precompute distances between points and the furthest vertex
            distances_to_furthest = [norm(node - furthest_vertex) for node in node_list]

            # Count covered points using vectorized operations
            covered_points = count(distances_to_furthest .< radios[i])

            # Update selected vertices based on the conditions
            if covered_points < min_covered_points || (covered_points == min_covered_points && radios[i] < min_radio)
                min_covered_points = covered_points
                min_radio = radios[i]
                selected_vertex_min = furthest_vertex
            end
            if covered_points < min_covered_points || (covered_points == min_covered_points && radios[i] > max_radio)
                min_covered_points = covered_points
                max_radio = radios[i]
                selected_vertex_max = furthest_vertex
            end
        end
    end

    # Select the vertex based on the probability
    selected_vertex = rand() < probability ? selected_vertex_min : selected_vertex_max

    return selected_vertex  # Return the counter along with the selected vertex
end


function add_n_new_points_julia(node_list::Vector{Vector{Float64}}, vertex_list::Vector{Vector{Vector{Float64}}}, distances::AbstractArray, radios::AbstractArray, probability::Float64, n::Int=1)
    # Collect eligible candidates where distance > radio
    candidates = []
    for i in eachindex(distances)
        if distances[i] > radios[i]
            point = node_list[i]
            region_vertices = vertex_list[i]
            # Find the furthest vertex in the Voronoi region
            furthest_vertex = region_vertices[argmax([norm(vertex - point) for vertex in region_vertices])]
            # Calculate distances from all nodes to this furthest vertex
            distances_to_furthest = [norm(node - furthest_vertex) for node in node_list]
            covered_points = count(d < radios[i] for d in distances_to_furthest)
            push!(candidates, (covered_points, radios[i], furthest_vertex))
        end
    end

    # Adjust n to select all candidates if n == -1
    if n == -1
        n = length(candidates)
    end

    # Ensure n does not exceed the number of eligible candidates
    if n > length(candidates)
        n = length(candidates)
    end

    # Validate input and constraints
    @assert length(node_list) == length(distances) == length(radios) "Input length mismatch"
    @assert !all(radios .>= distances) "All Voronoi cells are filled"
    @assert n <= length(node_list) "n exceeds node count"
    @assert n <= length(candidates) "n exceeds eligible candidates"

    # Determine sorting order for radios based on probability
    sort_order = rand() < probability ? :asc : :desc

    # Sort candidates by covered points (ascending) and radio (ascending/descending)
    sorted_candidates = sort(candidates, lt = function(a, b)
        if a[1] == b[1]
            return sort_order == :asc ? (a[2] < b[2]) : (a[2] > b[2])
        else
            return a[1] < b[1]
        end
    end)

    # Extract the top n vertices
    selected_vertices = [candidate[3] for candidate in sorted_candidates[1:n]]

    return selected_vertices
end