import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_tetrahedron_with_normals():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define vertices of the tetrahedron
    v = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, 1/np.sqrt(12), np.sqrt(2/3)]
    ])
    
    # Define faces using vertex indices (adjusting the order)
    faces = [[v[0], v[1], v[2]], 
             [v[0], v[3], v[1]], 
             [v[0], v[2], v[3]], 
             [v[1], v[3], v[2]]]

    # Draw tetrahedron
    ax.add_collection3d(Poly3DCollection(faces, facecolor='cornflowerblue', linewidths=1, edgecolor='k'))

    # Compute and draw normals for each face
    for face in faces:
        # Using the cross product to compute the normal
        normal = -1*np.cross(face[1] - face[0], face[2] - face[0])
        # Normalize the normal
        normal /= np.linalg.norm(normal)
        # Calculate face centroid to position the normal arrow
        centroid = np.mean(face, axis=0)
        # Offset the centroid outwards slightly to ensure the normal vectors are visible
        offset_centroid = centroid + normal * 0.1
        # Draw the normal as a quiver (arrow)
        ax.quiver(offset_centroid[0], offset_centroid[1], offset_centroid[2], 
                  normal[0], normal[1], normal[2], 
                  length=0.5, normalize=True, color="red", zorder =10)

    # Set the aspect ratio and show the plot
    # ax.set_xlim([-0.5, 1])
    # ax.set_ylim([-0.5, 1])
    # ax.set_zlim([-0.5, 1])
    # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.show()

draw_tetrahedron_with_normals()