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

p1, p2, theta1, theta2, theta3 = sympy.symbols('p1 p2 theta1 theta2 theta3')
p_vars = [p1, p2]
theta_vars = [theta1, theta2, theta3]
cons1 = theta1*p1**2 + theta2*p2**2 + theta3*p1*p2
fake_prob = cp.Problem(cp.Minimize(0))

diff_helper = DiffOptHelper(fake_prob, [cons1], p_vars, theta_vars)
p_val = np.array([1, 2])
theta_val = np.array([1, 2, 3])
time_start = time.time()
diff_helper.constraints_dict[0]["value"](p_val, theta_val)
diff_helper.constraints_dict[0]["dp"](p_val, theta_val)
diff_helper.constraints_dict[0]["dpdp"](p_val, theta_val)
diff_helper.constraints_dict[0]["dtheta"](p_val, theta_val)
diff_helper.constraints_dict[0]["dpdtheta"](p_val, theta_val)
time_end = time.time()
print("Time elapsed for value: ", time_end - time_start)


print(diff_helper.constraints_dict[0]["value"](p_val, theta_val))
print(diff_helper.constraints_dict[0]["dp"](p_val, theta_val))
print(diff_helper.constraints_dict[0]["dpdp"](p_val, theta_val))
print(diff_helper.constraints_dict[0]["dtheta"](p_val, theta_val))
print(diff_helper.constraints_dict[0]["dpdtheta"](p_val, theta_val))
