from mayavi import mlab
import numpy as np

def compute_normal(v1, v2, v3):
    """Compute the normal vector of a triangle given its vertices."""
    a = np.array(v2) - np.array(v1)
    b = np.array(v3) - np.array(v1)
    return np.cross(a, b)

def plot_face_and_normal(vertices, color):
    """Plot a triangular face and its normal vector."""
    x, y, z = zip(*vertices)
    triangles = [(0, 1, 2)]  # the triangle formed by the three vertices
    mlab.triangular_mesh(x, y, z, triangles, color=color, opacity=0.6)

    # Calculate the centroid and normal
    centroid = np.mean(vertices, axis=0)
    normal = compute_normal(*vertices)

    # Normalize the normal for visualization purposes
    normalized_normal = normal / np.linalg.norm(normal) * 0.5  # Scale for visibility

    # Plot the normal vector
    mlab.quiver3d(centroid[0], centroid[1], centroid[2],
                  normalized_normal[0], normalized_normal[1], normalized_normal[2],
                  color=color, scale_factor=1, mode='arrow')

vertices = [
        [0.7, 0.7, 0.7],  # origin
        [1, 0, 0],  # x-axis direction from origin
        [0, 1, 0],  # y-axis direction from origin
        [0, 0, 1]   # z-axis direction from origin
    ]

# Create the three faces of the corner
faces = [
    [vertices[0], vertices[3], vertices[1]],  # XZ plane
    [vertices[0], vertices[1], vertices[2]],  # XY plane
    [vertices[0], vertices[2], vertices[3]]   # YZ plane
]

mlab.figure(size=(800, 800))

# Plot the faces and their normal vectors
plot_face_and_normal(faces[0], (1, 0, 0))  # red
plot_face_and_normal(faces[1], (0, 1, 0))  # green
plot_face_and_normal(faces[2], (0, 0, 1))  # blue

mlab.show()

