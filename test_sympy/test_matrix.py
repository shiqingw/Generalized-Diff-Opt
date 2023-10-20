from sympy import zeros, ones

# Create a 3x3 zero matrix
zero_matrix_3x3 = zeros(3, 3)
print("3x3 zero matrix:")
print(zero_matrix_3x3)

zero_matrix_3x3[0:2,0:2] += ones(2,2)
print(zero_matrix_3x3)