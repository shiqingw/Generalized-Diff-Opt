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

nv = 2
nc1 = 1
_p = cp.Variable(nv)
_alpha = cp.Variable(1, pos=True)

_A1 = cp.Parameter((nc1, nv))
_b1 = cp.Parameter(nc1)
_circle_center = cp.Parameter(nv)

obj = cp.Minimize(_alpha)

# cons = [cp.sum_squares(_p - _circle_center) <= _alpha, _A1 @ _p + _b1 <= _alpha]
# cons = [cp.power(cp.norm(_p - _circle_center, p=2),2) <= _alpha, _A1 @ _p + _b1 <= _alpha]
cons = [cp.norm(_p - _circle_center, p=2) <= _alpha, _A1 @ _p + _b1 <= _alpha]

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
_alpha.gradient = 1
_p.gradient = np.array([0,0])
problem.backward()
time2 = time.time()
print(_alpha.value, _p.value)
print(_A1.gradient, _b1.gradient, _circle_center.gradient)
print("Time elapsed: ", time2 - time1)
print("#############################################")

###############################
px, py, alpha, A1x, A1y, b1, cx, cy = sympy.symbols('px py alpha A1x A1y b1 cx cy', real=True) 
cons = [sympy.sqrt((-cx + px)**2 + (-cy + py)**2), A1x*px + A1y*py + b1]
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
time1 = time.time()
grad_alpha, grad_p, grad_dual = diff_helper.get_gradient(alpha_val, p_val, theta_val, dual_val)
time2 = time.time()
print("Time elapsed: ", time2 - time1)
print('Gradient of alpha:', grad_alpha)
print('Gradient of p:', grad_p)
print('Gradient of dual:', grad_dual)