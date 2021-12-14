import struct
from functools import partial
import numpy as np
import scipy.linalg
import sys

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

n = 40

initilal_matrix_file = open('lu_init.out', encoding='latin1')

init_elems = []

for chunk, i in zip(iter(partial(initial_matrix_file.read, 8), b''), range(n**2)):
    init_elements.append(struct.unpack('d', bytes(chunk, encoding='latin1')))

init_matrix = np.matrix(init_elements)
init_matrix = init_matrix.reshape(n, n)


final_matrix_file = open('lu.out', encoding='latin1')

final_elems = []

for chunk, i in zip(iter(partial(final_matrix_file.read, 8), b''), range(n**2)):
    final_elements.append(struct.unpack('d', bytes(chunk, encoding='latin1')))

final_matrix = np.matrix(final_elements)
final_matrix = matrix.reshape(n, n)

#pi_file= open('lu.out', encoding='latin1')
#
#pi_elems = []
#
#for chunk, i in zip(iter(partial(pi_file.read, 8), b''), range(n**2)):
#    pi_elems.append(struct.unpack('d', bytes(chunk, encoding='latin1')))
#
#pi_vector = np.matrix(pi_elems)
#pi_vector = pi_vector.reshape(n, n)
print(np.linalg.norm(matrix))



print(matrix)
