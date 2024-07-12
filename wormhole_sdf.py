import numpy as np
import polyscope as ps

ps.init()


# ################################### SDF for an open cylinder ###################################
def sdOpenCylinder(p, radius, height):
    radial_dist = np.sqrt(p[:, 0]**2 + p[:, 2]**2) - radius # positive if the point lies outside the cylinder radially; negative if the point is inside radially
    vertical_dist = np.abs(p[:, 1]) - height / 2 # positive if the point is above or below the cylinder; negative if the point is within the vertical bounds of the cylinder
    return np.maximum(radial_dist, vertical_dist)

# ############################ SDF for a rotated and translated (along the X-axis) semi-cylinder (surface) ############################
def sdSemiCylinder(p, radius, height, shift_x):
    # necessary rotations
    rotated_x = p[:, 2] - shift_x  # z becomes x in the semi-cylinder, translate along x
    rotated_z = p[:, 0]  # x becomes z in the semi-cylinder

    # SDF
    radial_dist = np.sqrt(rotated_x**2 + p[:, 1]**2) - radius # same as cylinder's
    
    # condition to remove caps
    no_cap_condition = (rotated_z >= -height / 2) & (rotated_z <= height / 2)
    capped_sdf = np.where(no_cap_condition, radial_dist, np.inf)  
    
    # enforce semi-cylinder
    capped_sdf = np.where(rotated_x > 0, capped_sdf, np.inf)

    return capped_sdf

# ############################ SDF for a cube ############################
def sdBox(p, size):
    return np.max(np.abs(p) - size, axis=1) # positive if the point lies outside the cube; negative if the point is inside the cube



# grid setup
grid_size = 100 # resolution of the grid: 100 points in each dimension of the grid (x, y, z), leading to 1,000,000 points in total if the space is filled uniformly
outer_radius = 0.5
height = 3
cube_size = 1.5  

grid_spacing = (2 * (outer_radius + 1)) / grid_size # etermines the distance between adjacent points in the grid
x = np.linspace(-outer_radius - 1, outer_radius + 1 + cube_size, grid_size) # generate linearly spaced points: the range for each dimension ensures that the grid covers the entire area of interest plus some buffer
y = np.linspace(-height / 2 - 1, height / 2 + 1, grid_size)
z = np.linspace(-outer_radius - 1, outer_radius + 1 + cube_size, grid_size)

# didnt work
# x = np.linspace(-outer_radius - 1, outer_radius + 1, grid_size)
# y = np.linspace(-height / 2 - 1, height / 2 + 1, grid_size)
# z = np.linspace(-outer_radius - 1, outer_radius + 1, grid_size)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij') # creates a 3D meshgrid that forms the basis for computing the SDF values 
points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T # converts the meshgrid arrays into a list of 3D points

inner_radius = outer_radius - grid_spacing # or subsract grid_spacing: it is the distance between adjacent points on the grid so it will yield the same result

# SDFs for the outer and inner open cylinders
outer_cylinder_sdf = sdOpenCylinder(points, outer_radius, height)
inner_cylinder_sdf = sdOpenCylinder(points, inner_radius, height + 2)

# hollow cylinder resulted by the maximum negative combination of the outer and the inner cylinder
hollow_cylinder_sdf = np.maximum(outer_cylinder_sdf, -inner_cylinder_sdf)

# SDFs for the outer and inner open cubes
outer_cube_sdf = sdBox(points, np.array([cube_size, cube_size, cube_size]))
inner_cube_sdf = sdBox(points, np.array([cube_size + 1, cube_size, cube_size + 1]) - grid_spacing)

# hollow cube resulted by the maximum negative combination of the outer and the inner cube
hollow_cube_sdf = np.maximum(outer_cube_sdf, -inner_cube_sdf)

# holes resulted by the maximum negative combination of the hollow cube and the inner cylinder
hollow_cube_sdf = np.maximum(hollow_cube_sdf, -inner_cylinder_sdf)

# semi-cylinder, moved by half the cube size
semi_cylinder_sdf = sdSemiCylinder(points, cube_size, height, cube_size)

# combine using the minimum per pair of shapes
combined_sdf = np.minimum(np.minimum(hollow_cylinder_sdf, hollow_cube_sdf), semi_cylinder_sdf)

# filter points to visualize only the combined structure
surface_threshold = grid_spacing * 0.5
combined_points = points[np.abs(combined_sdf) < surface_threshold]


ps.register_point_cloud("Wormhole", combined_points)
ps.get_point_cloud("Wormhole").add_scalar_quantity("SDF", combined_sdf[np.abs(combined_sdf) < surface_threshold], enabled=True)
ps.show()
