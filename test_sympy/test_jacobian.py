from sympy import symbols, Matrix, diff, Function

x, y = symbols('x y')
p = Function('p')(x,y)
f = Matrix([p, p**2])
# vars = Matrix([x, y])
J = f.jacobian([x, y])
print(J)
