import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import time
import torch

def get_line(point1, point2):
    line = Line2D([point1[0], point2[0]],
              [point1[1], point2[1]], linewidth=2, color='black', linestyle='--')
    return line

def to_np(x):
    return x.cpu().detach().numpy()

############################################################################################################
# Hyperparameters
result_dir = Path.joinpath(Path(__file__).parent, 'square_circle')
if not result_dir.exists():
    result_dir.mkdir()

points1 = np.array([[-1,-1],[1,-1],[1,0.7],[-1,1]], dtype=np.float64)
R = 4
N = 100
thetas = np.linspace(0, 0.5*np.pi, N)
circle_centers = np.array([R*np.cos(thetas), R*np.sin(thetas)]).T
dist = 10
plot_x_lim = [-6,6]
plot_y_lim = [-6,6]
# solver_args={"solve_method": "SCS", "eps": 1e-7, "max_iters": 2000}
solver_args={"solve_method": "ECOS", "max_iters": 1000} # Cannot use eps for ECOS

############################################################################################################
# create records
record = {}
record['square_points'] = np.zeros((N, points1.shape[0], points1.shape[1]))
record['circle_centers'] = circle_centers
record['alpha_sol'] = np.zeros(N)
record['p_sol'] = np.zeros((N, 2))
record['square_points_alpha_grad'] = np.zeros((N, points1.shape[0], points1.shape[1]))
record['circle_center_alpha_grad'] = np.zeros_like(circle_centers)
record['square_points_p1_grad'] = np.zeros((N, points1.shape[0], points1.shape[1]))
record['circle_center_p1_grad'] = np.zeros_like(circle_centers)
record['square_points_p2_grad'] = np.zeros((N, points1.shape[0], points1.shape[1]))
record['circle_center_p2_grad'] = np.zeros_like(circle_centers)

############################################################################################################
# Build problem
nv = 2
nc1 = 4
_p = cp.Variable(nv)
_alpha = cp.Variable(1, pos=True)

_A1 = cp.Parameter((nc1, nv))
_b1 = cp.Parameter(nc1)
_circle_center = cp.Parameter(nv)

obj = cp.Minimize(_alpha)
cons = [_A1 @ _p + _b1 <= _alpha, cp.power(cp.norm(_p - _circle_center, p=2),2) <= _alpha]
# cons = [_A1 @ _p + _b1 <= _alpha, cp.norm(_p - _circle_center, p=2) <= _alpha]

problem = cp.Problem(obj, cons)
assert problem.is_dpp()
assert problem.is_dcp(dpp = True)

cvxpylayer = CvxpyLayer(problem, parameters=[_A1, _b1, _circle_center], variables=[_alpha, _p], gp=False)

############################################################################################################
# Loop over position of circle center
for ii in range(circle_centers.shape[0]):
    if ii % 20 == 0:
        print('==> {:04d} out of {:04d} '.format(ii, circle_centers.shape[0]))
    circle_center = circle_centers[ii]
    time1 = time.time()
    pixel_coords1 = torch.tensor(points1, requires_grad=True)
    x1 = pixel_coords1[:,0]
    y1 = pixel_coords1[:,1]
    A1_val = -torch.vstack((y1-torch.roll(y1,-1), torch.roll(x1,-1)-x1)).T
    b1_val = y1*torch.roll(x1,-1) - torch.roll(y1,-1)*x1
    norm = torch.linalg.vector_norm(A1_val, dim=1)
    A1_val = A1_val / norm.view(-1,1)
    b1_val = b1_val / norm + 1.0

    time2 = time.time()
    circle_center_val = torch.tensor(circle_center, requires_grad=True)

    time3 = time.time()
    alpha_sol, p_sol = cvxpylayer(A1_val, b1_val, circle_center_val, solver_args=solver_args)
    time4 = time.time()

    # print(alpha_sol, p_sol)
    # print(pixel_coords1.grad)
    # print(circle_center_val.grad)

    # print time taken in each step
    # print('Time taken to create A1 and b1: ', time2 - time1)
    # print('Time taken to create A2 and b2: ', time3 - time2)
    # print('Time taken to solve the problem: ', time4 - time3)

    ############################################################################################################
    # Create plot
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot()
    alpha_sol_np = to_np(alpha_sol)
    p_sol_np = to_np(p_sol)

    ############################################################################################################
    # Square 1
    center_position = np.sum(points1, axis=0)/4
    shape_range = np.array([dist,dist])
    N = 4000
    cx, cy = center_position
    sx, sy = shape_range
    dx = np.linspace(cx-sx, cx+sx, N)
    dy = np.linspace(cy-sy, cy+sy, N)
    X,Y = np.meshgrid(dx,dy)
    n_ineq = 4
    A_val = A1_val.detach().numpy()
    b_val = b1_val.detach().numpy()
    Z = np.zeros((A_val.shape[0], X.shape[0], X.shape[1]))
    for i in range(A_val.shape[0]):
        Z[i] = A_val[i, 0]*X + A_val[i, 1]*Y + b_val[i]
    in_inds = np.all(Z <= alpha_sol_np, axis=0) 
    points = np.hstack((X[in_inds].reshape(-1,1), Y[in_inds].reshape(-1,1)))
    hull = ConvexHull(points)
    polygon = plt.Polygon(points[hull.vertices], closed=True, fill="tab:blue", edgecolor=None, alpha=0.7, zorder=2)
    ax.add_patch(polygon)

    ax.add_line(get_line(points1[0], points1[1]))
    ax.add_line(get_line(points1[1], points1[2]))
    ax.add_line(get_line(points1[2], points1[3]))
    ax.add_line(get_line(points1[3], points1[0]))

    ############################################################################################################
    # Circle with center at circle_center and radius alpha_sol
    circle = plt.Circle(circle_center, np.sqrt(alpha_sol_np), fill="tab:blue", edgecolor=None, alpha=0.7, zorder=2)
    ax.add_patch(circle)
    circle = plt.Circle(circle_center, 1, fill=None, edgecolor='black', linewidth=2, linestyle='--', zorder=2)
    ax.add_patch(circle)

    ############################################################################################################
    # Adjust plot
    plt.plot(p_sol_np[0], p_sol_np[1],'ro') 
    tickfontsize = 20
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.xlim(plot_x_lim)
    plt.ylim(plot_y_lim)

    plt.gca().set_aspect('equal')
    plt.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(Path.joinpath(Path(__file__).parent, 'square_circle', 'square_circle_{:04d}.png'.format(ii)), dpi=200)
    plt.close()

    ############################################################################################################
    # Save records
    record['square_points'][ii] = to_np(pixel_coords1)
    record['circle_centers'][ii] = circle_center
    record['alpha_sol'][ii] = alpha_sol_np
    record['p_sol'][ii] = p_sol_np

    alpha_sol.backward(retain_graph=True)
    record['square_points_alpha_grad'][ii] = to_np(pixel_coords1.grad)
    record['circle_center_alpha_grad'][ii] = to_np(circle_center_val.grad)

    pixel_coords1.grad.zero_()
    circle_center_val.grad.zero_()
    p_sol[0].backward(retain_graph=True)
    record['square_points_p1_grad'][ii] = to_np(pixel_coords1.grad)
    record['circle_center_p1_grad'][ii] = to_np(circle_center_val.grad)

    pixel_coords1.grad.zero_()
    circle_center_val.grad.zero_()
    p_sol[1].backward(retain_graph=True)
    record['square_points_p2_grad'][ii] = to_np(pixel_coords1.grad)
    record['circle_center_p2_grad'][ii] = to_np(circle_center_val.grad)

############################################################################################################
# Save records using pickle
import pickle
with open(Path.joinpath(Path(__file__).parent, 'square_circle', '00record.pkl'), 'wb') as f:
    pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
print("==> Saved records. Done!")    
