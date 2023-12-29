import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.functional import jacobian

# Step 1: Forward pass
pixel_coords1 = torch.tensor([[580.0915, 276.8857],
                              [701.8141, 277.6409],
                              [706.8201, 390.0564],
                              [574.7156, 390.0147]], requires_grad=True)

def f(pixel_coords1):
    x1 = pixel_coords1[:,0]
    y1 = pixel_coords1[:,1]
    return -torch.vstack((y1-torch.roll(y1,-1), torch.roll(x1,-1)-x1)).T
    # return pixel_coords1**2

external_grad_A1_val = torch.tensor([[1,2.1],
                                     [3,4.4],
                                     [5,6.3],
                                     [7,8.9]])  

N = 500
times = []
for i in range(N):
    time1 = time.time()
    jacb = jacobian(f,pixel_coords1)
    prod = torch.einsum('ijkl,ij->kl', jacb, external_grad_A1_val)
    time2 = time.time()
    times.append(time2-time1)

# The gradient for 'a' is now in a.grad
print(np.mean(times))
print(prod)
# plt.hist(times, bins=50)
# plt.show()