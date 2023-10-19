import sympy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from DiffOptimization.TransformHelper import TransformHelper
import numpy as np
import time


x, y = sympy.symbols('x y')
states = [x, y]
transformed_states = [x*y]

helper = TransformHelper(states, transformed_states)
states_np = np.array([2, 1])
external_jacobian = np.array([4])
external_hessian = np.array([2])

N = 1
for i in range(N):
    start = time.time()
    print(helper.get_transformed_states_from_states(states_np))
    print(helper.get_transformation_jacobian(states_np))
    print(helper.get_transformation_hessian(states_np))
    print(helper.get_backward_jacobian(states_np, external_jacobian))
    print(helper.get_backward_hessian(states_np, external_jacobian, external_hessian))
    end = time.time()

print("Time taken for {} iterations: {} seconds".format(N, end-start))
print("Average time taken: {} seconds".format((end-start)/N))
