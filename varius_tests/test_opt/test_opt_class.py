import cvxpy as cp
import numpy as np
import sympy
from sympy import lambdify, Matrix, hessian, diff, Function
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from DiffOptimization.DiffOptHelper import DiffOptHelper
from cvxpylayers.torch import CvxpyLayer
import torch
import time

def is_symmetric(matrix, tol=1e-8):
    """
    Check if a matrix is symmetric.

    Parameters:
        matrix (ndarray): The input matrix.
        tol (float): The numerical tolerance for equality check.

    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if the matrix is equal to its transpose within the given tolerance
    return np.allclose(matrix, matrix.T, atol=tol)

nv = 2
nc1 = 1
_p = cp.Variable(nv)
_alpha = cp.Variable(1, pos=True)

_A1 = cp.Parameter((nc1, nv))
_b1 = cp.Parameter(nc1)
_circle_center = cp.Parameter(nv)

obj = cp.Minimize(_alpha)

cons = [cp.sum_squares(_p - _circle_center) <= _alpha, _A1 @ _p + _b1 <= _alpha]
# cons = [cp.power(cp.norm(_p - _circle_center, p=2),2) <= _alpha, _A1 @ _p + _b1 <= _alpha]
# cons = [cp.norm(_p - _circle_center, p=2) <= _alpha, _A1 @ _p + _b1 <= _alpha]

problem = cp.Problem(obj, cons)
assert problem.is_dpp()
assert problem.is_dcp(dpp = True)

cvxpylayer = CvxpyLayer(problem, parameters=[_A1, _b1, _circle_center], variables=[_alpha, _p], gp=False)
circle_center_np = np.array([2.,0])
A1_val_np = np.array([[1.,0]])
b1_val_np = np.array([1.])

A1_val = torch.tensor(A1_val_np, requires_grad=True)
b1_val = torch.tensor(b1_val_np, requires_grad=True)
circle_center_val = torch.tensor(circle_center_np, requires_grad=True)
solver_args = {"solve_method": "ECOS"}

time1 = time.time()
alpha_sol, p_sol = cvxpylayer(A1_val, b1_val, circle_center_val, solver_args=solver_args)
alpha_sol.backward()
# p_sol[1].backward()
time2 = time.time()

print(alpha_sol, p_sol)
print(A1_val.grad, b1_val.grad, circle_center_val.grad)
print("Time elapsed: ", time2 - time1)
print("#############################################")


###############################
_A1.value = A1_val_np
_b1.value = b1_val_np
_circle_center.value = circle_center_np
time1 = time.time()
problem.solve(solver=cp.SCS, requires_grad=True)
time2 = time.time()
print("Time for SCS solving: ", time2 - time1)

_alpha.gradient = 1
_p.gradient = np.array([0,0])
time3 = time.time()
problem.backward()
time4 = time.time()
print(_alpha.value, _p.value)
print(_A1.gradient, _b1.gradient, _circle_center.gradient)
print("Time for CVXPY gradient: ", time4 - time3)
print("#############################################")

###############################
px, py, alpha, A1x, A1y, b1, cx, cy = sympy.symbols('px py alpha A1x A1y b1 cx cy', real=True) 
# cons = [sympy.sqrt((-cx + px)**2 + (-cy + py)**2), A1x*px + A1y*py + b1]
cons = [(-cx + px)**2 + (-cy + py)**2, A1x*px + A1y*py + b1]
p_vars = [px, py]
theta_vars = [A1x, A1y, b1, cx, cy]
diff_helper = DiffOptHelper(problem, cons, p_vars, theta_vars)
print('Constraints in sympy:', cons)
dual_val = np.array([problem.constraints[i].dual_value for i in range(len(problem.constraints))]).squeeze()
alpha_val = _alpha.value
p_val = np.array(_p.value)
theta_val = np.array([A1_val_np[0,0], A1_val_np[0,1], b1_val_np[0], circle_center_np[0], circle_center_np[1]])
# print('Dual values:', dual_val)
# print('Alpha value:', alpha_val)
# print('p value:', p_val)
# print('theta value:', theta_val)

# print("#############################################")
# time1 = time.time()
# grad_alpha, grad_p, grad_dual = diff_helper.get_gradient(alpha_val, p_val, theta_val, dual_val)
# time2 = time.time()
# print("Time elapsed: ", time2 - time1)
# print('Gradient of alpha:', grad_alpha)
# print('Gradient of p:', grad_p)
# print('Gradient of dual:', grad_dual)

print("#############################################")
time1 = time.time()
grad_alpha, grad_p, grad_dual, hessian_alpha, hessian_p, hessian_dual = diff_helper.get_gradient_and_hessian(alpha_val, p_val, theta_val, dual_val)
time2 = time.time()
print("Time for getting gradient and hessian: ", time2 - time1)
print('Gradient of alpha:', grad_alpha)
print('Gradient of p:', grad_p)
print('Gradient of dual:', grad_dual)

print("#############################################")
# Sensitivity analysis
epsilon = 1e-2
_A1.value = A1_val_np + epsilon
_b1.value = b1_val_np + epsilon
_circle_center.value = circle_center_np + epsilon
problem.solve(solver=cp.SCS, requires_grad=True)
_alpha.gradient = 1 # gradient of the objective w.r.t _alpha
_p.gradient = np.array([0,0]) # gradient of the objective w.r.t _p
problem.backward()
print(_alpha.value, _p.value)
print(_A1.gradient, _b1.gradient, _circle_center.gradient)

print("#############################################")
dual_val = np.array([problem.constraints[i].dual_value for i in range(len(problem.constraints))]).squeeze()
alpha_val = _alpha.value
p_val = np.array(_p.value)
theta_val = np.array([A1_val_np[0,0], A1_val_np[0,1], b1_val_np[0], circle_center_np[0], circle_center_np[1]]) + epsilon
grad_alpha_new, grad_p_new, grad_dual_new = diff_helper.get_gradient(alpha_val, p_val, theta_val, dual_val)

print('Gradient of alpha:', grad_alpha_new)
print('Gradient of p:', grad_p_new)
print('Gradient of dual:', grad_dual_new)

print("#############################################")
grad_alpha_new_expected = grad_alpha + epsilon * np.ones((1, len(theta_val))) @ hessian_alpha 
grad_p_new_expected = grad_p + epsilon * np.einsum('ij,ijk->ik', np.ones((2, len(theta_val))), hessian_p)
grad_dual_new_expected = grad_dual + epsilon * np.einsum('ij,ijk->ik', np.ones((2, len(theta_val))), hessian_dual)
print("grad alpha new expected:", grad_alpha_new_expected)
print("grad alpha new actual:", grad_alpha_new)
print("grad p new expected:", grad_p_new_expected)
print("grad p new actual:", grad_p_new)
print("grad dual new expected:", grad_dual_new_expected)
print("grad dual new actual:", grad_dual_new)