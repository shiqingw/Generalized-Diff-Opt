import cvxpy as cp
import numpy as np
import sympy
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

cons = [_A1 @ _p + _b1 <= _alpha, cp.power(cp.norm(_p - _circle_center, p=2),2) <= _alpha]
# cons = [_A1 @ _p + _b1 <= _alpha, cp.norm(_p - _circle_center, p=2) <= _alpha]

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
time2 = time.time()

print(alpha_sol, p_sol)
print(A1_val.grad, b1_val.grad, circle_center_val.grad)
print("Time elapsed: ", time2 - time1)

###############################
_A1.value = A1_val_np
_b1.value = b1_val_np
_circle_center.value = circle_center_np
time1 = time.time()
problem.solve(solver=cp.SCS, requires_grad=True)
_alpha.gradient = 1
_p.gradient = np.zeros(nv)
problem.backward()
time2 = time.time()
print(_alpha.value, _p.value)
print(_A1.gradient, _b1.gradient, _circle_center.gradient)
print("Time elapsed: ", time2 - time1)

###############################
p = sympy.Symbol('p')
alpha = ((sympy.sqrt(5+4*p)-1)/2)**2
J_alpha = sympy.diff(alpha, p)
H_alpha = sympy.hessian(alpha, [p])
print(alpha.subs(p, circle_center_np[0]).evalf())
print(J_alpha.subs(p, circle_center_np[0]).evalf())
print(H_alpha.subs(p, circle_center_np[0]).evalf())