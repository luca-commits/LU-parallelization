import struct
from functools import partial
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

n = 40

f = open('lu.out', encoding='latin1')

elements = []

for chunk, i in zip(iter(partial(f.read, 8), b''), range(n**2)):
    elements.append(struct.unpack('d', bytes(chunk, encoding='latin1')))

matrix = np.matrix(elements)
matrix = matrix.reshape(n, n)

print(matrix)