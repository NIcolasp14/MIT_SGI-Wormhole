import numpy as np
import polyscope as ps


# Initialize polyscope
ps.init()


# Parameters for the bent plane
num_u = 100  # resolution along the width
num_v = 90   # resolution along the bend
width = 10   # total width of the plane
radius = 5   # radius of the half-circle bend
extension_length = 10  # length of each straight extension, adjusted for clear backward extension (to  extend the semi-circle, it fell behind it)
hole_radius = 1.0  # radius of the "circular" hole in the grid units

# Parameters for the cylinder
cylinder_radius_top = 1 
cylinder_radius_bottom = 1
cylinder_height = 2 * radius + 0.4 # make the height to connect both planes, the addition of 0.4 was empirical
cylinder_segments = 30 # cylinder height and circumference resolution



def is_inside_circle(x, z, circle_center_x, circle_center_z, circle_radius):
    """Function to check if a point is inside a circle in the xz plane."""
    return (x - circle_center_x) ** 2 + (z - circle_center_z) ** 2 <= circle_radius ** 2


def create_bent_space(num_u, num_v, width, radius, extension_length, hole_radius):
    """ 
    Function to create bent space. Use the parameters in the beginning to adjust the geometric parameters
    of the bent space. The space consists of a semi-circular bent plane with planar extensions from both 
    ends and circular holes placed at center positions on the extensions. The function calculates vertices for:

    1. A semi-circle based on the radius and the first third of num_v segments.
    2. Straight extensions on both sides of the semi-circle using the remaining segments of num_v.
    3. "Circular" holes on the extensions based on the specified hole radius.

    Each vertex is computed using polar coordinates for the curved parts and linear interpolations 
    for the extensions. The resulting mesh vertices and faces are structured to form a 3D model suitable
    for visualization or further processing.

    Parameters:
    - num_u (int): The number of divisions along the width of the bent plane, determining the resolution.
    - num_v (int): The total number of divisions along the length of the bend and the extensions.
    - width (float): The total width of the bent plane.
    - radius (float): The radius of the semi-circle part of the bent plane.
    - extension_length (float): The length of the straight extensions from each end of the semi-circle.
    - hole_radius (float): The radius of the circular holes to be subtracted from the bent plane.
    
    Returns:
    - A tuple containing two elements:
      1. vertices (np.array): An array of 3D coordinates for each vertex of the mesh.
      2. faces (list of lists): A list where each sublist contains indices of vertices that form a face.

    Example usage:
    >>> vertices, faces = create_bent_space(100, 90, 10, 5, 10, 0.5)
    """

    vertices = []
    faces = []

    # Choose the number of vertices in the bend and extensions
    num_bend = num_v // 3  # 1/3 for the bend
    num_extension = num_v - num_bend  # 1/3 for each extension


    ########### semi-circular bend: dimensions yz ###########
    for i in range(num_bend):
        angle = np.pi * (i / (num_bend - 1))  # angle from 0 to pi and I normalize/divide by num_bend - 1 to distribute the vertices across 0 and pi
        
        # we use the circle's formula in 3D (cosa+cosb, restricted to pi)
        y = radius * np.cos(angle) - radius  # Y-coordinates from 0 downward to -2*radius. Remove -radious to get a restaurant booth :).
        z = radius * np.sin(angle)  # Z extension to bring bend forward
               
        ########### making it 3D: dimension x ###########
        for j in range(num_u): # num_u is the resolution across the width
            u = j / (num_u - 1)  # u ranges from 0 to 1 across the width, so that we multiply by width and place vertices across the whole width 
            x = (u - 0.5) * width  # x ranges from -width/2 to width/2, to align with plane
            vertices.append((x, y, z)) # so for every vertice in yz add all of the neigboring vertices in x to make the shape 3D
   

    ########### bottom planar extension: dimensions yz ###########
    for i in range(num_extension):
        y = -2 * radius  # constant Y-coordinate making plane parallel to floor and also placed at the bottom of the semi circle to connect with it
        z = -i / (num_extension - 1) * extension_length 
        # Z-coordinates: explanation
        # place the vertices across the length of the extension planes. 
        # it normalises by the number of the total vertices
        # it places them across the extension_length
        
        ########### making it 3D: dimension x ###########
        for j in range(num_u): 
            u = j / (num_u - 1)  
            x = (u - 0.5) * width  
            vertices.append((x, y, z)) 
   

    ########### upper planar extension: dimensions yz ###########
    for i in range(num_extension):
        y = 0  # constant Y-coordinate for upper extension aligned with the top edge of the semi-circle
        z = -i / (num_extension - 1) * extension_length  # Z-coordinates, similar to bottom plane
        
        ########### making it 3D: dimension x ###########
        for j in range(num_u):
            u = j / (num_u - 1)  
            x = (u - 0.5) * width  
            vertices.append((x, y, z))


    # Define the circle centers for holes in both planes
    bottom_hole_center_x = 0
    bottom_hole_center_z = -extension_length / 2 # negative because all orientations are assumed negative
    upper_hole_center_x = 0
    upper_hole_center_z = -extension_length / 2



    ########### quad faces, semi-circle ########### 
    for i in range(num_bend - 1): # we stop one row short because the last row is included by the second to last row
        for j in range(num_u - 1): # same
            x_center = (j + 0.5) / num_u * width - width / 2 
            # (j + 0.5) / num_u this term finds the center of the quad instead of its boundary.
            # because we have range from -width / 2 to -width / 2

            z_center = radius * np.sin(np.pi * (i + 0.5) / (num_bend - 1))
            # np.sin(np.pi * (i + 0.5) / (num_bend - 1)): the sin of the centers of the quad faces of the semi circle
            
            if is_inside_circle(x_center, z_center, bottom_hole_center_x, bottom_hole_center_z, hole_radius):
                continue # no holes will be found here
            idx1 = i * num_u + j # calculates the index of a vertex in a 1D array that represents a 2D mesh grid, i changes row, j iterates columns. num_v simulate changing a row in the hypothetical 2D array, but we have 1D
            idx2 = idx1 + num_u # the vertex one place up from idx1 in the mesh grid, one rown down in the data structure
            idx3 = idx1 + 1 # the vertex next to idx1
            idx4 = idx2 + 1 # the vertex next to idx2
            faces.append([idx1, idx2, idx4]) # counter clockwise
            faces.append([idx1, idx4, idx3]) # counter clockwise

    ########### quad faces, bottom plane ########### 
    base_idx = num_bend * num_u  # Starting index for bottom extension in 1D array that represents 2D mesh
    for i in range(num_extension - 1):
        for j in range(num_u - 1):
            x_center = (j + 0.5) / num_u * width - width / 2
            z_center = - (i + 0.5) / (num_extension - 1) * extension_length    
            if is_inside_circle(x_center, z_center, bottom_hole_center_x, bottom_hole_center_z, hole_radius):
                continue  # Skip faces that would be in the hole
            idx1 = base_idx + i * num_u + j
            idx2 = idx1 + num_u
            idx3 = idx1 + 1
            idx4 = idx2 + 1
            faces.append([idx1, idx2, idx4]) # counter clockwise (light blue, else dark blue)
            faces.append([idx1, idx4, idx3]) # counter clockwise

    ########### quad faces, upper plane ########### 
    offset = num_v * num_u # Starting index for upper extension in 1D array that represents 2D mesh
    for i in range(num_extension - 1):
        for j in range(num_u - 1):
            x_center = (j + 0.5) / num_u * width - width / 2
            z_center = - (i + 0.5) / (num_extension - 1) * extension_length    
            if is_inside_circle(x_center, z_center, upper_hole_center_x, upper_hole_center_z, hole_radius):
                continue  # Skip faces that would be in the hole 
            idx1 = offset + i * num_u + j
            idx2 = idx1 + num_u
            idx3 = idx1 + 1
            idx4 = idx2 + 1
            faces.append([idx1, idx2, idx4]) # counter clockwise
            faces.append([idx1, idx4, idx3]) # counter clockwise

    return np.array(vertices), np.array(faces)

########### cylinder ########### 
def create_wormhole(radius_top, radius_bottom, height, segments):
    cylinder_vertices = []
    cylinder_faces = []

    # Create vertices for the cylinder
    for i in range(segments + 1):
        t = i / segments # cylinder's height resolution
        y = t * height 
        current_radius = radius_top  # in case we want to parametrize it

        for j in range(segments):
            angle = 2 * np.pi * j / segments # cylinder's circumference resolution

            # for each cylinder's height segment we calculate the cyinder's circle
            x = current_radius * np.cos(angle)
            z = current_radius * np.sin(angle)

            cylinder_vertices.append((x, y, z))


    # Create faces for the cylinder
    for i in range(segments):
        for j in range(segments):
            idx1 = i * segments + j # current vertex
            idx2 = idx1 + segments # vertex above idx1
            idx3 = idx1 + 1 if j < segments - 1 else i * segments # horizontal slice: they if statement makes sure it goes around, finishing the rotation.
            idx4 = idx2 + 1 if j < segments - 1 else (i + 1) * segments # vertical slice

            if i < segments - 1:
                cylinder_faces.append([idx1, idx2, idx4]) # counter clockwise
                cylinder_faces.append([idx1, idx4, idx3]) # counter clockwise

    return np.array(cylinder_vertices), np.array(cylinder_faces)





# Create the bent space
vertices, faces = create_bent_space(num_u, num_v, width, radius, extension_length, hole_radius)

# Create the cylinder
cylinder_vertices, cylinder_faces = create_wormhole(cylinder_radius_top, cylinder_radius_bottom, cylinder_height, cylinder_segments)

# Adjust the cylinder position to connect the holes
# Empirically adjust the y-position to place the base on the bottom plane
cylinder_vertices[:, 1] -= 2 * radius #+ (cylinder_height / 2 - 0.2) # if we remove 2 as a coeef of radious this works too
# I substract the (2*)radius of the bent space (semi-circle) to align the center of it with the center of the cylinder, because the cylinder
# was generated with the bottom base alligned with the center of the semi-circle.

# Empirically align the cylinder with the z-axis holes
cylinder_vertices[:, 2] -= extension_length / 2  
# Similar to radius above

# Combine the vertices and faces from the plane and cylinder
all_vertices = np.vstack([vertices, cylinder_vertices])
all_faces = np.vstack([faces, cylinder_faces + len(vertices)]) # :+ len(vertices) adjusting the indeces after stacking vertices

# Register the combined mesh in Polyscope
ps.register_surface_mesh("Wormhole", all_vertices, all_faces)


# show the visualization in polyscope
ps.show()


