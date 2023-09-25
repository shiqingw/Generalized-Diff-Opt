import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D

def get_line(point1, point2):
    line = Line3D([point1[0], point2[0]],
              [point1[1], point2[1]],
              [point1[2], point2[2]], linewidth=3, color='black', linestyle='solid')
    return line

def compute_normal(v0, v1, v2):
    # Calculate the vectors between the three points
    a = np.array(v1) - np.array(v0)
    b = np.array(v2) - np.array(v0)
    
    # Compute the cross product of the two vectors
    normal = np.cross(a, b)
    
    # Normalize the vector
    normal = normal / np.linalg.norm(normal)
    return normal

class Arrow3D:
    def __init__(self, ax, x, y, z, dx, dy, dz, color='black'):
        self.ax = ax

        arrow_length = np.linalg.norm([dx, dy, dz])
        arrow_prop = 0.2  # proportion of arrowhead
        head_length = arrow_length * arrow_prop
        tail_length = arrow_length - head_length

        # Calculate the end point of the arrow tail
        x_tail = x + dx * tail_length / arrow_length
        y_tail = y + dy * tail_length / arrow_length
        z_tail = z + dz * tail_length / arrow_length

        # Draw the tail using a line
        ax.plot([x, x_tail], [y, y_tail], [z, z_tail], color, linewidth=5, zorder=1)

        # Define a function to create an orthogonal vector
        def orthogonal_vector(v):
            if v[0] == 0 and v[1] == 0:
                if v[2] == 0:
                    raise ValueError('Zero vector')
                return [0, 1, 0]
            return [-v[1], v[0], 0]

        # Generate points for the base of the cone
        n = 50
        circle_points = []
        radius = head_length * 0.2
        direction = np.array([dx, dy, dz]) / arrow_length
        ortho = orthogonal_vector(direction)
        ortho /= np.linalg.norm(ortho)
        cross = np.cross(direction, ortho)
        for i in np.linspace(0, 2*np.pi, n):
            point = [x_tail, y_tail, z_tail]
            point += radius * (np.cos(i) * ortho + np.sin(i) * cross)
            circle_points.append(point)

        # Draw the cone using triangular facets
        arrow_tip = [x+dx, y+dy, z+dz]
        for i in range(n):
            triangle = [arrow_tip, circle_points[i], circle_points[(i+1)%n]]
            ax.add_collection3d(Poly3DCollection([triangle], color=color, zorder=1))


# Create a 3D plot
fig = plt.figure(figsize=(6, 6), dpi=200)
ax = fig.add_subplot(111, projection='3d',computed_zorder=False)

# Define the vertices of the corner
N = 5
offset = 0
angles = np.linspace(0 + offset, 2*np.pi + offset, N, endpoint=False)
vertices = [[0,0,2]] + [[np.cos(angle), np.sin(angle), 0] for angle in angles]

# Create the three faces of the corner
faces = [
    [vertices[0], vertices[1], vertices[2]],  
    [vertices[0], vertices[2], vertices[3]],  
    [vertices[0], vertices[3], vertices[4]],
    [vertices[0], vertices[4], vertices[5]],
    [vertices[0], vertices[5], vertices[1]],
]

colors = ['tab:blue', 'tab:red', 'tab:green', "tab:orange", "tab:purple"]
for i in range(len(colors)):
# Plot the faces
    face = faces[i]
    ax.add_collection3d(Poly3DCollection([face], facecolors=colors[i], linewidths=2, edgecolors=None, alpha=.7, zorder=1))
    centroid = np.mean(face, axis=0)
    normal = compute_normal(*face)
    arrow = Arrow3D(ax, *centroid, *(normal*0.6), color=colors[i])

ax.add_line(get_line(vertices[0], vertices[1]))
ax.add_line(get_line(vertices[0], vertices[2]))
ax.add_line(get_line(vertices[0], vertices[3]))
ax.add_line(get_line(vertices[0], vertices[4]))
ax.add_line(get_line(vertices[0], vertices[5]))

# Set the tick size for each axis
tickfontsize = 20
ax.xaxis.set_tick_params(size=tickfontsize)
ax.yaxis.set_tick_params(size=tickfontsize)
ax.zaxis.set_tick_params(size=tickfontsize)
ax.tick_params(labelsize=tickfontsize)

ax.grid(False)
ax.axis('off')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set the view angle (elevation and azimuth)
ax.view_init(elev=77, azim=-100)  # Adjust these angles
# Show the plot
plt.gca().set_aspect('equal')
plt.tight_layout()
# plt.show()
plt.savefig("five_faces_corners.png")
