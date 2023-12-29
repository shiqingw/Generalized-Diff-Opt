import numpy as np
from sympy import symbols, Function, Matrix, diff, lambdify

# Define the symbols
x, y, z = symbols('x y z')

# Define p as a function of x, y, z
p = x**2 + y**2 + z**2
print(type(p))
p_matrix = Matrix([x**2 + y**2 + z**2, x**2 + y**2 + z**2])
print(p_matrix.jacobian([x, y, z]))