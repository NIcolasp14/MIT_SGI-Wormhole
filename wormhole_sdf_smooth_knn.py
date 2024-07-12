import numpy as np
import polyscope as ps
from sklearn.neighbors import NearestNeighbors

ps.init()

# SDF for a hollow cylinder 
def sdOpenCylinder(p, radius, height):
    radial_dist = np.sqrt(p[:, 0]**2 + p[:, 2]**2) - radius
    vertical_dist = np.abs(p[:, 1]) - height / 2
    return np.maximum(radial_dist, vertical_dist)

# SDF for a semi-cylinder 
def sdSemiCylinder(p, radius, height, shift_x):
    rotated_x = p[:, 2] - shift_x 
    rotated_z = p[:, 0] 
    radial_dist = np.sqrt(rotated_x**2 + p[:, 1]**2) - radius

    uncapped_dist = radial_dist  
    
    no_cap_condition = (rotated_z >= -height / 2) & (rotated_z <= height / 2)
    capped_sdf = np.where(no_cap_condition, uncapped_dist, np.inf)  
    
    capped_sdf = np.where(rotated_x > 0, capped_sdf, np.inf)

    return capped_sdf

# SDF for a cube
def sdBox(p, size):
    return np.max(np.abs(p) - size, axis=1)

# grid setup
grid_size = 100
new_outer_radius = 0.5
height = 3
cube_size = 1.5  

grid_spacing = (2 * (new_outer_radius + 1)) / grid_size
x = np.linspace(-new_outer_radius - 1, new_outer_radius + 1 + cube_size, grid_size)
y = np.linspace(-height / 2 - 1, height / 2 + 1, grid_size)
z = np.linspace(-new_outer_radius - 1, new_outer_radius + 1 + cube_size, grid_size)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

inner_radius = new_outer_radius - grid_spacing

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

outer_cylinder_sdf = sdOpenCylinder(points, new_outer_radius, height)
inner_cylinder_sdf = sdOpenCylinder(points, inner_radius, height + 2)
hollow_cylinder_sdf = np.maximum(outer_cylinder_sdf, -inner_cylinder_sdf)

outer_cube_sdf = sdBox(points, np.array([cube_size, cube_size, cube_size]))
inner_cube_sdf = sdBox(points, np.array([cube_size + 1, cube_size, cube_size + 1]) - grid_spacing)
hollow_cube_sdf = np.maximum(outer_cube_sdf, -inner_cube_sdf)

hollow_cube_sdf = np.maximum(hollow_cube_sdf, -inner_cylinder_sdf)
semi_cylinder_sdf = sdSemiCylinder(points, cube_size, height, cube_size)
combined_sdf = np.minimum(np.minimum(hollow_cylinder_sdf, hollow_cube_sdf), semi_cylinder_sdf)

# Filter points to visualize only the combined structure
surface_threshold = grid_spacing * 0.5
near_surface_indices = np.where(np.abs(combined_sdf) < surface_threshold)[0]
combined_points = points[near_surface_indices]

# Use KNN to form the mesh
k = 5  # Number of neighbors for KNN
nbrs = NearestNeighbors(n_neighbors=k).fit(combined_points)
distances, indices = nbrs.kneighbors(combined_points)

vertices = []
faces = []
vertex_map = {}

# converting the point cloud into a structured mesh using knn
for i, point in enumerate(combined_points): # accessing every point of the whole wormhole shape
    if i not in vertex_map: # checks if we have already processed the point
        vertex_map[i] = len(vertices)
        vertices.append(point)
    
    for j in indices[i]: # neighbors of point i
        if j != i and j in vertex_map: # if the point j is not point i and if j has already been accessed
            for k in indices[j]:
                if k != j and k in vertex_map and k != i: # and if there is a neighbor of j that is not i or j and that has been accessed
                    face = sorted([vertex_map[i], vertex_map[j], vertex_map[k]]) # create a face using the 3
                    if face not in faces: # if it is a new face
                        faces.append(face) # add it

vertices = np.array(vertices)
faces = np.array(faces)

# smooth the mesh by averaging the positions of each vertex's neighbors
for i in range(5):  # number of smoothing iterations
    smoothed_vertices = np.copy(vertices)
    for j, v in enumerate(vertices): # for every vertex
        neighbor_indices = indices[j] # take neighbors calculated by knn
        # smoothing
        smoothed_vertices[j] = np.mean(vertices[neighbor_indices], axis=0) # calculates the new position for vertex i by taking the mean of the positions of its neighbors
    vertices = smoothed_vertices

# Register the mesh with Polyscope
ps.register_surface_mesh("KNN Mesh", vertices, faces)
ps.show()
