from sympy.abc import x, y, a
from sympy.geometry.util import idiff
from sympy import lambdify
circ = x**2 + y**2 - 4
print(lambdify([x, y], idiff(circ, y, x), "numpy"))


print(idiff(circ, y, x, 2).simplify())