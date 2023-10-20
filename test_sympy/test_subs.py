from sympy import symbols, sin, Function, diff

x, y, z = symbols('x y z')
expr = x**2 + sin(y)

new_expr = expr.subs([[x,y], [y,z]])
print(new_expr)  # Output will be: x**2 + z
