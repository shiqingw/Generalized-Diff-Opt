import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Use pickle to load records
result_dir = Path.joinpath(Path(__file__).parent, 'square_circle_sqrt')
with open(Path.joinpath(result_dir, '00record.pkl'), 'rb') as f:
    record = pickle.load(f)

tickfontsize = 20
labelsize = 20
# plt.rcParams['text.usetex'] = True

# Plot the alpha_sol
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['alpha_sol'], label=r'$\alpha_{\mathrm{sol}}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$\alpha_{\mathrm{sol}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00alpha_sol.png'), dpi=200)

# Plot the p_sol
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['p_sol'][:,0], label=r'$p_{\mathrm{sol},x}$')
ax.plot(record['p_sol'][:,1], label=r'$p_{\mathrm{sol},y}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$p_{\mathrm{sol}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00p_sol.png'), dpi=200)

# Plot the circle_centers
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['circle_centers'][:,0], label=r'$c_{x}$')
ax.plot(record['circle_centers'][:,1], label=r'$c_{y}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$c_{\mathrm{sol}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00circle_centers.png'), dpi=200)

# Plot the points1
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['square_points'][:,0,0], label=r'$p_{1,x}$')
ax.plot(record['square_points'][:,0,1], label=r'$p_{1,y}$')
ax.plot(record['square_points'][:,1,0], label=r'$p_{2,x}$')
ax.plot(record['square_points'][:,1,1], label=r'$p_{2,y}$')
ax.plot(record['square_points'][:,2,0], label=r'$p_{3,x}$')
ax.plot(record['square_points'][:,2,1], label=r'$p_{3,y}$')
ax.plot(record['square_points'][:,3,0], label=r'$p_{4,x}$')
ax.plot(record['square_points'][:,3,1], label=r'$p_{4,y}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$p_{1}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00square_points.png'), dpi=200)

# Plot the circle_center_alpha_grad
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['circle_center_alpha_grad'][:,0], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial c_{x}}$')
ax.plot(record['circle_center_alpha_grad'][:,1], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial c_{y}}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial c_{\mathrm{sol}}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00circle_center_alpha_grad.png'), dpi=200)

# Plot the circle_center_p1_grad
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['circle_center_p1_grad'][:,0], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial c_{x}}$')
ax.plot(record['circle_center_p1_grad'][:,1], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial c_{y}}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$\frac{\partial p_{\mathrm{sol}}}{\partial c_{\mathrm{sol}}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00circle_center_p1_grad.png'), dpi=200)

# Plot the circle_center_p2_grad
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['circle_center_p2_grad'][:,0], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial c_{x}}$')
ax.plot(record['circle_center_p2_grad'][:,1], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial c_{y}}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$\frac{\partial p_{\mathrm{sol}}}{\partial c_{\mathrm{sol}}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00circle_center_p2_grad.png'), dpi=200)

# Plot the square_points_alpha_grad
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['square_points_alpha_grad'][:,0,0], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{1,x}}$')
ax.plot(record['square_points_alpha_grad'][:,0,1], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{1,y}}$')
ax.plot(record['square_points_alpha_grad'][:,1,0], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{2,x}}$')
ax.plot(record['square_points_alpha_grad'][:,1,1], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{2,y}}$')
ax.plot(record['square_points_alpha_grad'][:,2,0], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{3,x}}$')
ax.plot(record['square_points_alpha_grad'][:,2,1], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{3,y}}$')
ax.plot(record['square_points_alpha_grad'][:,3,0], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{4,x}}$')
ax.plot(record['square_points_alpha_grad'][:,3,1], label=r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{4,y}}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$\frac{\partial \alpha_{\mathrm{sol}}}{\partial p_{1}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00square_points_alpha_grad.png'), dpi=200)

# Plot the square_points_p1_grad
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['square_points_p1_grad'][:,0,0], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{1,x}}$')
ax.plot(record['square_points_p1_grad'][:,0,1], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{1,y}}$')
ax.plot(record['square_points_p1_grad'][:,1,0], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{2,x}}$')
ax.plot(record['square_points_p1_grad'][:,1,1], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{2,y}}$')
ax.plot(record['square_points_p1_grad'][:,2,0], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{3,x}}$')
ax.plot(record['square_points_p1_grad'][:,2,1], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{3,y}}$')
ax.plot(record['square_points_p1_grad'][:,3,0], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{4,x}}$')
ax.plot(record['square_points_p1_grad'][:,3,1], label=r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{4,y}}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$\frac{\partial p_{\mathrm{sol},x}}{\partial p_{1}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00square_points_p1_grad.png'), dpi=200)

# Plot the square_points_p2_grad
fig = plt.figure(figsize=(8,6), dpi=200)
ax = fig.add_subplot(111)
ax.plot(record['square_points_p2_grad'][:,0,0], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{1,x}}$')
ax.plot(record['square_points_p2_grad'][:,0,1], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{1,y}}$')
ax.plot(record['square_points_p2_grad'][:,1,0], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{2,x}}$')
ax.plot(record['square_points_p2_grad'][:,1,1], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{2,y}}$')
ax.plot(record['square_points_p2_grad'][:,2,0], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{3,x}}$')
ax.plot(record['square_points_p2_grad'][:,2,1], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{3,y}}$')
ax.plot(record['square_points_p2_grad'][:,3,0], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{4,x}}$')
ax.plot(record['square_points_p2_grad'][:,3,1], label=r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{4,y}}$')
ax.set_xlabel('Iteration', fontsize = labelsize)
ax.set_ylabel(r'$\frac{\partial p_{\mathrm{sol},y}}{\partial p_{1}}$', fontsize = labelsize)
ax.legend(loc='upper right', fontsize = labelsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.tight_layout()
plt.savefig(Path.joinpath(result_dir, '00square_points_p2_grad.png'), dpi=200)


