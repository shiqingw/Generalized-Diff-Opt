import sympy
from sympy import lambdify, hessian, diff

x, y, z = sympy.symbols('x y z')
expr = x**2 + 4*x*y + 4*y**2
print(diff(expr, [x,y]))
# func = lambdify([[x, y], [z]], expr, 'numpy')
# h_func = lambdify([[x, y], [z]], hessian(expr, [x,y]), 'numpy')
# print(func([1,1],[100]))
# print(h_func([1,1],[100]))