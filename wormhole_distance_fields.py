import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
import polyscope as ps

ps.init()

# SDF for an open cylinder (no caps)
def sdOpenCylinder(p, radius, height):
    radial_dist = np.sqrt(p[:, 0]**2 + p[:, 2]**2) - radius
    vertical_dist = np.abs(p[:, 1]) - height / 2
    return np.maximum(radial_dist, vertical_dist)

# SDF for a rotated and translated semi-cylinder along the X-axis
def sdSemiCylinder(p, radius, height, shift_x):
    # Rotate points around Y-axis to orient the semi-cylinder
    rotated_x = p[:, 2] - shift_x  # z becomes x in the semi-cylinder, translate along x
    rotated_z = p[:, 0]  # x becomes z in the semi-cylinder
    radial_dist = np.sqrt(rotated_x**2 + p[:, 1]**2) - radius

    uncapped_dist = radial_dist  # Only consider radial distance for SDF
    
    # Apply uncapping condition: keep inside the height but do not cap ends
    no_cap_condition = (rotated_z >= -height / 2) & (rotated_z <= height / 2)
    capped_sdf = np.where(no_cap_condition, uncapped_dist, np.inf)  # Use radial distance where within height limits, else infinity
    
    # Enforce semi-cylinder (only one half)
    capped_sdf = np.where(rotated_x > 0, capped_sdf, np.inf)

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
cube_size = 1.5  
grid_spacing = (2 * (new_outer_radius + 1)) / grid_size
x = np.linspace(-new_outer_radius - 1, new_outer_radius + 1 + cube_size, grid_size)
y = np.linspace(-height / 2 - 1, height / 2 + 1, grid_size)
z = np.linspace(-new_outer_radius - 1, new_outer_radius + 1 + cube_size, grid_size)

inner_radius = new_outer_radius - grid_spacing
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

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
sdf = points[np.abs(combined_sdf) < surface_threshold]

# Convert the grid and combined SDF to a PyVista grid
grid = pv.StructuredGrid(xx, yy, zz)
grid['sdf'] = combined_sdf.flatten()

# Threshold the grid to display only the surface
surface = grid.threshold([-surface_threshold, surface_threshold])

# Visualize using PyVista
plotter = pv.Plotter()
plotter.add_mesh(surface, color='w', opacity=0.5)  
plotter.show()
