import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, distance
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def compute_normal(elev, azim):
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    
    x = np.cos(azim_rad) * np.cos(elev_rad)
    y = np.sin(azim_rad) * np.cos(elev_rad)
    z = np.sin(elev_rad)
    
    return (x, y, z)

# Define φ as the golden ratio
phi = (1 + math.sqrt(5)) / 2

# Generate the initial points
points = []

# Points for (0, ±1, ±3φ)
points.extend([(0, sign1, sign2 * 3 * phi) for sign1 in [1, -1] for sign2 in [1, -1]])

# Points for (±1, ±(2 + φ), ±2φ)
points.extend([(sign1, sign2 * (2 + phi), sign3 * 2 * phi) for sign1 in [1, -1] for sign2 in [1, -1] for sign3 in [1, -1]])

# Points for (±φ, ±2, ±(2φ + 1))
points.extend([(sign1 * phi, sign2 * 2, sign3 * (2 * phi + 1)) for sign1 in [1, -1] for sign2 in [1, -1] for sign3 in [1, -1]])

points = np.array(points)

# Function to check if a permutation is even
def is_even(permutation):
    inversions = 0
    for i, j in itertools.combinations(permutation, 2):
        if i > j:
            inversions += 1
    return inversions % 2 == 0

# Generate even permutations
even_permutations = []
for point in points:
    for perm in itertools.permutations([0, 1, 2]):
        if is_even(perm):
            even_permutations.append(point[list(perm)])

even_permutations = np.array(even_permutations)

# Calculate the pairwise distances
distances = distance.cdist(even_permutations, even_permutations)
unique_distances = sorted(np.unique(distances))

# Identify edge length as the smallest non-zero unique distance
edge_length = unique_distances[1]

# Create a 3D plot
fig = plt.figure(figsize=(5, 5), dpi=200)
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
elev=26
azim=38
normal_vector = compute_normal(elev, azim)
d = 0
if_cover = lambda point: point[0] * normal_vector[0] + point[1] * normal_vector[1] + point[2] * normal_vector[2] < d

for i, point1 in enumerate(even_permutations):
    for j, point2 in enumerate(even_permutations):
        if abs(distances[i, j] - edge_length) < 0.01 and i != j:
            if if_cover(point1) or if_cover(point2):
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], linewidth=2, color='black', linestyle="solid", zorder=1)
            else:
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], linewidth=2, color='black', linestyle="solid", zorder=3)

# Calculate the convex hull
hull = ConvexHull(even_permutations)
polygons = []
for simplex in hull.simplices:
    vertices = even_permutations[simplex]
    polygons.append(vertices)
ax.add_collection3d(Poly3DCollection(polygons, facecolors='tab:blue', alpha=0.5, zorder=2))


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
ax.view_init(elev=elev, azim=azim)  # Adjust these angles
# Show the plot
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("c60.png")
# plt.show()
