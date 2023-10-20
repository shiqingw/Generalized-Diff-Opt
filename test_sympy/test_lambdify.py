import sympy
from sympy import lambdify, hessian, diff
import numpy as np

x, y, z = sympy.symbols('x y z')
expr = x**2 + 4*x*y + 4*y**2
tensor_3d = np.empty((3, 3, 3), dtype=object)
for i in range(3):
    for j in range(3):
        for k in range(3):
            tensor_3d[i, j, k] = expr

func = lambdify([x,y,z],tensor_3d, "numpy")
