import numpy as np
import polyscope as ps

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

voxels = combined_sdf.reshape((grid_size, grid_size, grid_size))

# use marching cubes to generate the mesh
from skimage.measure import marching_cubes
verts, faces, normals, values = marching_cubes(voxels, level=0, spacing=(grid_spacing, grid_spacing, grid_spacing))

ps.register_surface_mesh("Wormhole", verts, faces)
ps.show()
