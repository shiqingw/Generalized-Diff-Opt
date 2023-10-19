import cvxpy as cp

# Create two scalar optimization variables.
x = cp.Variable()
y = cp.Variable()

# Define constraints
constraints = [x + y <= 1, x - y >= 1]

# Define objective
objective = cp.Minimize(x**2 + y**2)

# Form and solve problem.
prob = cp.Problem(objective, constraints)
print(isinstance(prob, cp.problems.problem.Problem))
assert False
prob.solve()

# The optimal dual variable (Lagrange multiplier) for
# a constraint is stored in constraint.dual_value.
print("optimal (x + y <= 1) dual variable", constraints[0].dual_value)
print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
print("optimal value:", prob.value)

# Access the optimal values of the variables.
optimal_x = x.value
optimal_y = y.value

# Evaluate the constraints at the optimal solution.
constraint_values = [constraint.expr.value for constraint in constraints]
print(constraint_values)