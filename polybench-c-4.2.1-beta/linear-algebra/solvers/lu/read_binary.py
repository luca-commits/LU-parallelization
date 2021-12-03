import struct
from functools import partial

f = open('lu.out', encoding='latin1')

for chunk in iter(partial(f.read, 8), b''):
    print(struct.unpack('d', bytes(chunk, encoding='latin1')))