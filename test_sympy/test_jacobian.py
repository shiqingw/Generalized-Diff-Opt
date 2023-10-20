from sympy import symbols, Matrix, diff

x, y = symbols('x y')
f = Matrix([x**2, x + y])
vars = Matrix([x, y])
J = f.jacobian(vars)

print("Jacobian:", J)

A = Matrix([
    [x**2, x+y],
    [x*y, y**2]
])

# Compute the Jacobian of each column
J1 = A[:, 0].jacobian(vars)
J2 = A[:, 1].jacobian(vars)

print("Jacobian of first column:", J1)
print("Jacobian of second column:", J2)

print(diff(A, x))