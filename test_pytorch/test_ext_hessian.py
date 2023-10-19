import torch
from torch.autograd.functional import jacobian, hessian
import time
from functools import partial

def f(x):
    return x**2

# Initialize x as a tensor with requires_grad=True so that gradients can be computed
x = torch.tensor([2.0, 3.0], requires_grad=True)

time1 = time.time()
jacobian_matrix = jacobian(f, x)
time2 = time.time()
print(jacobian_matrix)
print("Time taken for jacobian: ", time2-time1)

time3 = time.time()
hessian_matrix = hessian(lambda x: f(x)[1], x)
time4 = time.time()
print(hessian_matrix)
print("Time taken for jacobian: ", time4-time3)