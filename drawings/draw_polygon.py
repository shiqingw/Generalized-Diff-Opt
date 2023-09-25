import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Example usage
N = 6
# Generating sample vertices for a regular N-sided polygon
offset = 2*np.pi/4
angles = np.linspace(0 + offset, 2*np.pi + offset, N, endpoint=False)
vertices = [(np.cos(angle), np.sin(angle)) for angle in angles]
fig = plt.figure(figsize=(6, 6), dpi=200)
ax = fig.add_subplot(111)
# Extract x and y coordinates of vertices for easier plotting
x = [vert[0] for vert in vertices]
y = [vert[1] for vert in vertices]

# Fill the polygon with cornflowerblue color
ax.fill(x, y, color='tab:blue', alpha = 0.6)

# Draw solid edges for all but the last segment and compute the normals
for i in range(len(vertices)):
    start = vertices[i]
    end = vertices[(i + 1) % len(vertices)]  # wrap-around to 0 for the last vertex

    if i==len(vertices)-1:
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linestyle=':', linewidth=3.0)
    else:
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linestyle='-', linewidth=3.0)
    
    # Compute direction vector for the edge
    edge_dir = np.array([end[0] - start[0], end[1] - start[1]])
    
    # Compute the normal by rotating edge_dir by 90 degrees clockwise
    normal = np.array([edge_dir[1], -edge_dir[0]])
    
    # Normalize the normal to a desired length (e.g., 0.2)
    normal_length = 0.2
    normalized_normal = normal / np.linalg.norm(normal) * normal_length
    
    # Compute midpoint of the edge for positioning the normal arrow
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    
    # Draw the normal arrow
    if i<len(vertices)-1:
        ax.arrow(midpoint[0], midpoint[1], normalized_normal[0], normalized_normal[1], 
                head_width=0.05, head_length=0.1, fc='red', ec='red', linewidth=4.0)

ax.set_aspect('equal')
ax.grid(False)
ax.axis('off')
ax.set_xticks([])
ax.set_yticks([])
# plt.show()
plt.savefig("polygon.png")