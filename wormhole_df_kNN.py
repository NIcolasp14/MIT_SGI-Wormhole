import numpy as np
import polyscope as ps
from sklearn.neighbors import NearestNeighbors

# Initialize Polyscope
ps.init()

# SDF for an open cylinder (no caps)
def sdOpenCylinder(p, radius, height):
    radial_dist = np.sqrt(p[:, 0]**2 + p[:, 2]**2) - radius
    vertical_dist = np.abs(p[:, 1]) - height / 2
    return np.maximum(radial_dist, vertical_dist)

# SDF for a rotated and translated semi-cylinder along the X-axis
def sdSemiCylinder(p, radius, height, shift_x):
    rotated_x = p[:, 2] - shift_x  # z becomes x in the semi-cylinder, translate along x
    rotated_z = p[:, 0]  # x becomes z in the semi-cylinder
    radial_dist = np.sqrt(rotated_x**2 + p[:, 1]**2) - radius
    vertical_dist = np.abs(rotated_z) - height / 2
    uncapped_dist = radial_dist  # Only consider radial distance for SDF
    no_cap_condition = (rotated_z >= -height / 2) & (rotated_z <= height / 2)
    capped_sdf = np.where(no_cap_condition, uncapped_dist, np.inf)  # Use radial distance where within height limits, else infinity
    capped_sdf = np.where(rotated_x > 0, capped_sdf, np.inf)  # Enforce semi-cylinder (only one half)
    return capped_sdf

# SDF for a cube
def sdBox(p, size):
    return np.max(np.abs(p) - size, axis=1)

# Grid setup
grid_size = 100
initial_radius = 2
radius_reduction = 1.5
new_outer_radius = initial_radius - radius_reduction
height = 3
cube_size = 1.5  # Size of the cube
grid_spacing = (2 * (new_outer_radius + 1)) / grid_size
x = np.linspace(-new_outer_radius - 1, new_outer_radius + 1 + cube_size, grid_size)
y = np.linspace(-height / 2 - 1, height / 2 + 1, grid_size)
z = np.linspace(-new_outer_radius - 1, new_outer_radius + 1 + cube_size, grid_size)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

inner_radius = new_outer_radius - grid_spacing

# Calculate the SDFs for the outer and inner open cylinders
outer_cylinder_sdf = sdOpenCylinder(points, new_outer_radius, height)
inner_cylinder_sdf = sdOpenCylinder(points, inner_radius, height + 2)
hollow_cylinder_sdf = np.maximum(outer_cylinder_sdf, -inner_cylinder_sdf)

# Create a hollow cube
outer_cube_sdf = sdBox(points, np.array([cube_size, cube_size, cube_size]))
inner_cube_sdf = sdBox(points, np.array([cube_size + 1, cube_size, cube_size + 1]) - grid_spacing)
hollow_cube_sdf = np.maximum(outer_cube_sdf, -inner_cube_sdf)

# Use the inner cylinder to make holes in the hollow cube
hollow_cube_sdf = np.maximum(hollow_cube_sdf, -inner_cylinder_sdf)

# Create a semi-cylinder, moved by half the cube size
semi_cylinder_sdf = sdSemiCylinder(points, cube_size, height, cube_size)

# Combine the hollow cylinder with the hollow cube and semi-cylinder
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

for i, point in enumerate(combined_points):
    if i not in vertex_map:
        vertex_map[i] = len(vertices)
        vertices.append(point)
    
    for j in indices[i]:
        if j != i and j in vertex_map:
            for k in indices[j]:
                if k != j and k in vertex_map and k != i:
                    face = sorted([vertex_map[i], vertex_map[j], vertex_map[k]])
                    if face not in faces:
                        faces.append(face)

vertices = np.array(vertices)
faces = np.array(faces)

# Smooth the mesh by averaging the positions of each vertex's neighbors
for _ in range(5):  # Number of smoothing iterations
    smoothed_vertices = np.copy(vertices)
    for i, v in enumerate(vertices):
        neighbor_indices = indices[i]
        smoothed_vertices[i] = np.mean(vertices[neighbor_indices], axis=0)
    vertices = smoothed_vertices

# Register the mesh with Polyscope
ps.register_surface_mesh("KNN Mesh", vertices, faces)
ps.show()
