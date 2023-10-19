import torch
import time
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Forward pass
pixel_coords1 = torch.tensor([[580.0915, 276.8857],
                              [701.8141, 277.6409],
                              [706.8201, 390.0564],
                              [574.7156, 390.0147]], requires_grad=True)

x1 = pixel_coords1[:,0]
y1 = pixel_coords1[:,1]
A1_val = -torch.vstack((y1-torch.roll(y1,-1), torch.roll(x1,-1)-x1)).T
# A1_val = pixel_coords1**2

# Step 2: External gradient for b
external_grad_A1_val = torch.tensor([[1,2.1],
                                     [3,4.4],
                                     [5,6.3],
                                     [7,8.9]])  


# Step 3: Backward pass
N = 500
times = []
for i in range(N):
    if pixel_coords1.grad is not None:
        pixel_coords1.grad.zero_()
    time1 = time.time()
    A1_val.backward(external_grad_A1_val, retain_graph=True)
    time2 = time.time()
    times.append(time2-time1)

# The gradient for 'a' is now in a.grad
print(pixel_coords1.grad)
print(np.mean(times))
# plt.hist(times, bins=50)
# plt.show()