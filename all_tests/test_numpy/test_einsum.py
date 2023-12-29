import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
print(np.einsum('ij,ijk->ik', A, B))