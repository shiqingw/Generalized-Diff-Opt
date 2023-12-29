import numpy as np
from sympy import symbols, Function, Matrix, diff, lambdify

# Define the symbols
x, y, z = symbols('x y z')

# Define p as a function of x, y, z
p = Function('p')(x, y, z)

# Assume some expression for F, for example, F = p*x + p*y + p*z
F = p*x + p*y + p*z

# Compute the gradient of F
grad_F = Matrix([diff(F, var) for var in (x, y, z)])

# Lambdify the gradient function
grad_F_func = lambdify((p, x, y, z, Matrix([diff(p, var) for var in (x, y, z)])), grad_F, 'numpy')

# Assume some values
p_value = np.array(1.0)
x_value = np.array(2.0)
y_value = np.array(3.0)
z_value = np.array(4.0)
grad_p_value = np.array([0.1, 0.2, 0.3])

# Compute the gradient of F
grad_F_value = grad_F_func(p_value, x_value, y_value, z_value, grad_p_value)

# Now grad_F_value contains the gradient of F with respect to x, y, z
print(grad_F_value)
