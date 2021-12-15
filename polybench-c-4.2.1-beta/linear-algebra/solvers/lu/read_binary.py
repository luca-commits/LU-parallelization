import struct
from functools import partial
import numpy as np
import scipy.linalg
import sys

np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2)

n = 16

init_matrix = np.fromfile('lu_init.out', dtype=np.float64, count=n**2).reshape(n, n)

print(scipy.linalg.lu(init_matrix))

lu_matrix = np.fromfile('lu.out', dtype=np.float64, count=n**2).reshape(n, n)

l_matrix = np.tril(lu_matrix, k=-1) + np.eye(n, n)
u_matrix = np.triu(lu_matrix)

pi_elems = np.fromfile('pi.out', dtype=np.int32, count=n)
p_matrix = np.zeros((n, n))

for i in range(n):
    p_matrix[i, pi_elems[i]] = 1

a_matrix = np.matmul(p_matrix.T, np.matmul(l_matrix, u_matrix))

print("Initial matrix:")
print(init_matrix)
print("P matrix:")
print(p_matrix)
print("LU matrix:")
print(lu_matrix)
print("Output matrix:")
print(a_matrix)
print("Difference:")
print(init_matrix - a_matrix)

err = np.linalg.norm(a_matrix - init_matrix)

print(f"Mean squared error: {err}")