from sympy import Matrix, symbols, diff, simplify

# Define the variable
x = symbols('x')

# # Define the matrix M
M = Matrix([[x, 1, 2], [1, x, 3], [2, 3, x]])

dMinv_dx = diff(M.inv(), x)
print(dMinv_dx)

# # Compute the derivative of M with respect to x
# dM_dx = diff(M, x)

# # Compute the derivative of M^-1 with respect to x using the formula
# dMinv_dx = -M.inv() * dM_dx * M.inv()

# # Simplify the expression
# dMinv_dx_simplified = simplify(dMinv_dx)

# # Output the simplified expression for d(M^-1)/dx
# print(dMinv_dx_simplified)
